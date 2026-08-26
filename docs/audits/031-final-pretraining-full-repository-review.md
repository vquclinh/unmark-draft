# Audit 031 - Final Pretraining Full Repository Review

Date: 2026-08-25

Mode: REVIEW ONLY. No production code, tests, scripts, configs, requirements,
proposal text, decisions, freeze file, prior audits, or scientific artifacts
were modified. This file is the only repository file created by this prompt.

Final verdict:

FINAL FULL-REPOSITORY PRE-TRAIN REVIEW BLOCKED -- MATERIAL ISSUE(S) REQUIRE HUMAN DECISION BEFORE TRAINING

Reason: five BLOCKER findings remain in the current repository. They are
concrete, reachable failures or campaign-integrity failures in the real Stage-1
launch/resume/artifact/provenance path.

## 1. Audit purpose

This audit is the final complete pre-training repository-wide review of the
UNMARK research project before Stage-1 scientific training. The review standard
was hostile and independent: prior audits, tests, comments, freeze metadata, and
documentation were treated as historical evidence rather than authority. Where
possible, behavior was derived from current executable source.

The audit explicitly did not repair defects.

## 2. Exact HEAD and repository start state

Mandatory start commands were run first.

```
git rev-parse HEAD
55aa4064780b37626bcae7eef83c504a96fcc51f

git status --short
<no output>

git diff --cached --name-status
<no output>

git diff --check
<no output>
```

`git log --oneline --decorate -10` began with:

```
55aa406 (HEAD -> main, origin/main, origin/HEAD) final training-launch readiness review
855a1d3 fix bug before training stage-1
479fac5 stage-1 configuration stage
649ad74 fix the persistent cuda
a84cf7e local tests
ac20cfb record fresh CUDA pretrain verification evidence
3c3489b implement Stage 1 pretrain execution contracts
3a5368c docs: record final Stage 1 pretrain blockers
0588b72 docs: close Stage 1 pretrain smoke gate
2363e33 Fix Stage 1 smoke CLI completion contract
```

Expected documentation HEAD in the prompt was
`855a1d3c9179477680153b093546f872b8846b2a`; the actual HEAD is
`55aa4064780b37626bcae7eef83c504a96fcc51f`.

Starting repository status was clean and nothing was staged.

## 3. Relationship to 479fac5 and 855a1d3

The prompt required verification of the historical claim that the scientific
code/configuration at `479fac5bb5fb7be4518e8ed36162c137700851ed` and
`855a1d3c9179477680153b093546f872b8846b2a` is byte-identical.

Observed:

```
git diff --name-status 479fac5bb5fb7be4518e8ed36162c137700851ed 855a1d3c9179477680153b093546f872b8846b2a
M       docs/audits/030-pretrain-repository-wide-audit.md
```

The complete diff contents were inspected. They modify only Audit 030
documentation. No executable scientific code, tests, proposal, decisions,
freeze file, configs, requirements, or training path changed between those two
commits.

The additional delta from expected `855a1d3...` to current HEAD was also
checked:

```
git diff --name-status 855a1d3c9179477680153b093546f872b8846b2a HEAD
M       docs/audits/030-pretrain-repository-wide-audit.md
```

The complete diff contents were inspected. Current HEAD `55aa406...` again
changes only Audit 030 documentation, adding the launch-readiness review. The
scientific implementation at `479fac5...`, `855a1d3...`, and `55aa406...` is
byte-identical.

Classification: MINOR documentation/provenance mismatch, not a scientific code
delta.

## 4. Local `.venv` verification

Before Python/test execution, the required local environment checks were run:

```
pwd
/mnt/vquclinh/PROJECT-CMAKE/UNMARK-DRAFT/unmark-draft

test -x .venv/bin/python
<exit 0>

.venv/bin/python -c "import sys; print(sys.executable); print(sys.prefix)"
/mnt/vquclinh/PROJECT-CMAKE/UNMARK-DRAFT/unmark-draft/.venv/bin/python
/mnt/vquclinh/PROJECT-CMAKE/UNMARK-DRAFT/unmark-draft/.venv

.venv/bin/python -m pip --version
pip 26.2.1 from /mnt/vquclinh/PROJECT-CMAKE/UNMARK-DRAFT/unmark-draft/.venv/lib64/python3.14/site-packages/pip (python 3.14)
```

Additional read-only metadata:

```
Python 3.14.5
pytest==9.1.1
PyYAML==6.0.3
transformers=NOT INSTALLED
tokenizers=NOT INSTALLED
sentencepiece=NOT INSTALLED
safetensors=NOT INSTALLED
torch=NOT INSTALLED
accelerate=NOT INSTALLED
```

The executable and prefix resolve inside the repository-local `.venv`.

## 5. Review scope

The complete repository tree was inspected with `git ls-files`, `rg --files`,
targeted `rg`, `nl`, `sed`, and local tests. Review coverage included:

- `unmark-proposal.md`, `docs/spec/decisions.md`,
  `docs/spec/stage1-final-freeze.json`, and `tests/test_stage1_final_freeze.py`.
- All `docs/audits/*.md` in chronological order as historical evidence.
- Stage-1 implementation under `unmark/stage1/`.
- Stage-1 model/objective/adapter implementation under `unmark/modeling/`.
- Stage-1 scripts, especially `scripts/stage1_runner.py`.
- Stage-1 relevant tests under `tests/`.
- Stage-6 preparation, checkpointing, manifest, corpus verification, and
  parallel preparation code.
- Requirements files and local `.venv` metadata.
- Repository-wide searches for training steps, resume/checkpoint entry points,
  official UIT-VSFC TEST references, RNG, TODO/FIXME/HACK/XXX, broad exception
  swallowing, `strict=False`, Python `hash()`, time/UUID, and alternate
  optimizer paths.

The official UIT-VSFC TEST was not opened, inspected, screened, evaluated, or
passed to any command.

## 6. Repository architecture and operational pipeline

The repository now contains multiple historical stages:

- Orthography core: Unicode/canonicalization/decomposition/recomposition under
  `unmark/orthography/`.
- Deterministic corruption: `unmark/corruption/` and Stage-1 wrappers in
  `unmark/stage1/contracts.py`.
- Linguistic inventory: pinned Vietnamese syllable inventory access under
  `unmark/linguistics/` and `unmark/stage1/preflight.py`.
- Historical pre-G1/G-1 evaluation and probes under `unmark/evaluation/`,
  `unmark/gates/`, and `scripts/b3*`.
- Stage-6 corpus preparation under `scripts/stage1_runner.py`,
  `unmark/stage1/corpus.py`, `chunking.py`, `lengths.py`, `parallel.py`,
  `manifest.py`, and `checkpoint.py`.
- Stage-1 scientific execution under `scripts/stage1_runner.py`,
  `unmark/stage1/execute.py`, `trainer.py`, `objective.py`, `optim.py`,
  `sampler.py`, `preparation.py`, `device.py`, and `selection.py`.

Intended operational order:

1. `prepare-corpus` verifies pinned UVW shard bytes and opened contamination
   references, splits documents, chunks without truncation, writes
   `chunks.jsonl`, `manifest.json`, and `COMPLETE.json`.
2. `lr-pilot` verifies the prepared corpus, runs three candidates at
   `r = 1.0` and seed `21230`, and writes `lr_pilot.json`.
3. `r-phase1` reads `lr_pilot.json`, freezes the selected LR, runs five
   candidates over the locked `r` grid at seed `21230`, and writes
   `r_phase1.json`.
4. `final-main` reads `lr_pilot.json` and `r_phase1.json`, runs three final
   seeds `36930`, `7309`, `5993`, and writes `final_main.json`.
5. Each nominal run starts with a fresh adapter, shared immutable frozen
   PhoBERT, fresh objective, fresh optimizer, fresh sampler, and per-run
   checkpoint namespace.

Current reality: the Stage-1 training command path cannot reach its first
scientific update because `execute_stage()` raises `NameError` before
`train_run()` is called (Finding 031-B1).

## 7. Five-way consistency matrix

| Axis | Proposal | Decisions | Freeze | Production code | Tests | Verdict |
|---|---|---|---|---|---|---|
| Backbone | `vinai/phobert-base` | D-B3B0-007, D-S1B-015/017 | pinned | `protocol.py:30-32`, `execute.py:71-94` | freeze/runner/model tests | MATCH |
| Backbone revision | `01daacda68afe13d83023d16ec647239e344a1e6` | D-B3B0-007 | pinned | `protocol.py:31`, runner revision check | freeze/runner tests | MATCH |
| Hidden size | 768 | D-B4B/D-S1B | pinned | `protocol.py:32`, `execute.py:91-93` | runner tests | MATCH |
| Frozen encoder | frozen weights, eval behavior | D-S1B-017 | pinned | `execute.py:88-90`, `adapter.py:544-577`, `trainer.py:201-232` | model/runner/runtime tests, CUDA evidence | MATCH |
| Precision | FP32 only | D-S1B-015 | pinned | `device.py:88-138`, no AMP calls in Stage-1 | device/runner tests, CUDA evidence | MATCH, runtime-gated |
| Adapter architecture | tone/letter embeddings, fusion, LayerNorm, gate | D-B4A/B4B, D-S1B-016/017 | 3,551,232 params | `adapter.py:106-280`, `protocol.py:34` | adapter/model tests | MATCH |
| Adapter init | deterministic CPU init from run seed domain | D-S1B-016 | pinned | `initialisation.py:29-57`, `protocol.py:217-230`, `execute.py:295-314` | run independence tests | MATCH in source; B1 prevents actual launch |
| Corpus identity | UVW-2026 | D-S1B-002, D-S1B-009..013 | pinned | `protocol.py:40-42`, `corpus.py:133-254` | corpus/checkpoint tests | MATCH |
| Shard order | train -> validation -> test concat | D-S1B-002 | pinned | `protocol.py:42-46`, `corpus.py:229-254` | corpus tests | MATCH |
| Split | document split before chunking, 5000 dev docs | D-S1B-009..013 | pinned | `corpus.py:558+`, runner stage 5/6 order | corpus/chunking tests | MATCH |
| Chunking | max length 256, no truncation, overflow fail | D-S1B-009..013 | pinned | `chunking.py`, `manifest.py`, `execute.py:47` | chunking/lengths/prep tests | MATCH |
| Text policy | preserve source text, no silent normalization/drop | D-S1B-009..013 | pinned | `chunking.py`, `manifest.py`, Stage-6 checkpoint | source-coordinate tests | MATCH |
| Prepared artifact identity | manifest + chunks + COMPLETE hashes | Audit 030 F1 | pinned | `checkpoint.py:894-989`, runner `_verified_corpus` | corpus verification tests | MATCH |
| Contamination | opened TRAIN/VALIDATION only | D-S1B-001/002 | pinned | `corpus.py:402-444`, runner args only opened refs | contamination/sealing tests | MATCH |
| Official TEST | sealed/unused | D-S1B-001 | pinned | no Stage-1 TEST arg; manifest requires false | sealing/runner tests | MATCH |
| Training corruption | p~U(0,1), seed 35422, redraw per visit, PI_STRIP=0.25 | D-S1B-003 | pinned | `contracts.py`, `protocol.py:89`, `trainer.py:636-648` | corruption/sampler tests | MATCH |
| Validation corruption | FULL/P50/P100/STRIP_ALL, seed 19225, run seed independent | D-S1B-001/005 | pinned | `validation.py`, `protocol.py:161-197` | validation/runner tests | MATCH |
| Batch/schedule | batch 128, accumulation 1, eval/checkpoint 500 | D-S1B-004 | pinned | `protocol.py:152-155`, `trainer.py:680-710` | schedule/checkpoint tests | MATCH |
| Budget | initial 20k, hard 40k, one continuation | D-S1B-004/008 | pinned | `selection.py:115-147`, `trainer.py:532-545` | budget tests | REGRESSED in real resume/orchestration, B2/B3 |
| Optimizer | AdamW, betas, eps, no amsgrad, const LR, no warmup/clipping | D-S1B-004 | pinned | `optim.py:70-101`, `trainer.py:597-600` | optimizer tests, torch skipped locally | MATCH in source |
| Weight decay grouping | weights 0.01, embeddings/bias/LN 0 | D-S1B-004 | pinned | `optim.py:19-67` | runner/torch tests | MATCH |
| Campaign | 3 LR + 5 r + 3 final = 11 nominal | D-S1B-004/005 | pinned | `selection.py:227-259`, runner | schedule tests | MATCH |
| Selection | score worst-case, tie d_clean then earlier update/lower LR/lower r | D-S1B-004 | pinned | `selection.py:81-212` | selection tests | MATCH internally |
| Artifact handoff | downstream stages consume prior selections | D-S1B-004/008 | partially recorded | `_load_selection()` only validates stage/protocol | tests do not attack full handoff | BLOCKER, B4 |
| Git provenance | repository identity must be meaningful | D-S1B-008/history | stale freeze head | Stage-6 derives HEAD; training accepts caller field | Stage-6 tests only | BLOCKER, B5 |

Five-way consistency verdict: the frozen scientific constants and core
mathematical/data implementation are internally consistent, but the real
training launch/resume/artifact/provenance path is not consistent with the
frozen execution contract.

## 8. Historical findings ledger

Prior audits were read as historical evidence. Current status was verified
against current source and tests, not inherited from prior "closed" labels.

| Historical finding | Original issue | Repair/current source evidence | Current regression evidence | Current status |
|---|---|---|---|---|
| Audit 001 B1, blanket ML import ban | Local import tests prevented later ML modules | Lazy import boundary now localizes torch/transformers to run paths | `tests/test_stage1_runner_contract.py` import checks | CLOSED/SUPERSEDED |
| Audit 001 B2, signature false fail | Case/punctuation restoration signature over-strict | Strict and rewrite signatures separated | restore utility tests | CLOSED |
| Audits 003/004 B2 corruption determinism | Corruption example key/eligibility unclear | deterministic keyed corruption, schema version, no Python `hash()` in production | corruption tests | CLOSED |
| Audit 005 B3A inventory provenance | Vietnamese syllable inventory not pinned or redistributable | `preflight.InventoryIdentity`, gitignored cache, checksum pin | inventory/preflight tests | CLOSED |
| Audits 006-010 PhoBERT/VnCoreNLP/tokenizer | Input policy and external revisions unclear | RAW_BASE selected, tokenizer/backbone pinned, VnCoreNLP not Stage-1 path | input/probe/runner tests and docs | CLOSED/SUPERSEDED |
| Audits 011-013 RAW_BASE alignment | Token/channel projection and unknown-id behavior uncertain | `prepare_example`, `project_text`, alignment/channel tests | Stage-1 projection tests | CLOSED |
| Audits 014-017 adapter/frozen encoder | Adapter architecture, position ids, eval behavior, gradients | `adapter.py`, `objective.py`, `trainer.py` gates | model/runner/runtime tests and historical CUDA evidence | CLOSED |
| Audits 018-019 Stage-1 objective/data path | Values and inventory provenance still open | D-S1B and freeze lock values; inventory required before science | freeze/preflight tests | CLOSED |
| Audits 020-027 pre-G1/UIT-VSFC role | TEST role and runner boundaries | Stage-1 excludes official TEST; pre-G1 roles remain distinct | sealing/runner tests | CLOSED for Stage-1 |
| Audit 028 config review | TEST contamination contradiction, L_clean floor, lambda collapse, SD tie | D-S1B fixes in decisions/protocol/selection | final freeze and selection tests | CLOSED |
| Audit 029 Stage-6 prep | chunking pin/performance/resume concerns | durable Stage-6 checkpoint, prepared corpus verification, no truncation | Stage-6 checkpoint/corpus tests | CLOSED |
| Audit 030 F1 prepared corpus verification | Training trusted manifest declarations | runner verifies `COMPLETE.json` and re-hashed artifacts before model | corpus verification tests | CLOSED |
| Audit 030 F2 PI_STRIP duplicate literals | Configuration duplication risk | `contracts.PI_STRIP` imports protocol value | freeze/contract tests | CLOSED |
| Audit 030 F3 training checkpoint persistence | Checkpoint helpers existed but were not invoked | Checkpoint writing now in `trainer.py:696-710`; resume load wired | Tests inspect persistence, but real reader is incompatible | REGRESSED/PARTIALLY CLOSED, B2/B3 |
| Audit 030 F4 corpus materialization | Full corpus materialized before model | Still materializes, but accepted by measured launch evidence and model load order | measurement evidence | CLOSED for launch acceptability |
| Audit 030 F5 Stage-6 counters blind under workers | Observability gap with parallel prep | Parallel output equality and timing counters added | parallel/prep tests | CLOSED/SUPERSEDED |
| Audit 030 F6 stale compiled PDF | `unmark-proposal.pdf` can be stale vs editable proposal | PDF still present; editable MD treated as authoritative | Documentation-only | STILL OPEN, DOCUMENTATION-ONLY |
| Audit 030 §V provenance round-trip | `RunProvenance(**to_dict())` TypeError masked verification | `require_match()` compares artifact dict to fresh identity | provenance tests | CLOSED for RunProvenance |
| Audit 030 §W inventory provisioning | Scientific run could miss inventory | `verify_scientific_inputs()` before model | preflight tests | CLOSED |
| Audit 030 §X truncation wiring | Validation/training truncation policy could drift | shared `TRUNCATION` and explicit pass-through | validation preparation tests | CLOSED |
| Audit 030 §Y CUDA/device boundary | CPU/device mismatch risk | `device.py` enforces CUDA/determinism; batch follows model | device tests and CUDA evidence | CLOSED, runtime-gated |
| Audit 030 §AA smoke CLI completion | Smoke handler read missing parser field | shared `_prepared_corpus_inputs()` | CLI tests | CLOSED |
| Audit 030 §AD cross-candidate leakage | Adapter/objective/optimizer state leaked across candidates | fresh adapter/objective inside schedule loop, shared frozen encoder only | run independence tests | CLOSED in source; B1 blocks runtime |
| Audit 030 §AF performance | Serial per-batch prep too slow | persistent spawn pool accepted by benchmark/evidence | parallel prep tests/evidence | CLOSED |
| Audit 030 §AF.4 CUDA resume evidence | Synthetic CUDA resume equivalence accepted | Evidence still useful for optimizer/device pieces | Does not cover production ValidationPoint reader or cap recovery | PARTIALLY VALID, INVALIDATED for production resume |
| Audit 030 §AI logical cuda vs cuda:0 | Harness compared logical vs concrete CUDA devices incorrectly | Device identity exclusions clarified | CUDA resume evidence | CLOSED |
| Audit 030 §AL-B1 objective construction | `objective_cls` unbound in `execute_stage` | Current source still uses stale name | no test catches; tests pin stale symbol | STILL OPEN, B1 |
| Audit 030 §AL-B2 stage handoff | selection artifacts under-validated | Current `_load_selection()` still checks only stage/protocol | no end-to-end artifact attack test | STILL OPEN, B4 |
| Audit 030 §AL-B3 repository HEAD | training provenance caller-supplied | Current runner still passes optional `args.repository_head` | Stage-6 tests do not cover training | STILL OPEN, B5 |
| Audit 030 §AL-B4 continuation crash resume | resumed stage uses initial cap for continued checkpoints | Current code still ignores checkpoint cap in first resumed call | no production crash-resume test | STILL OPEN, B3 |

Historical closure verdict: most historical scientific/data/model blockers
remain closed. The current repository is deliberately pre-§AL-repair and still
contains the launch-readiness blockers. The checkpoint/resume subsystem has an
additional current production incompatibility beyond §AL-B4.

## 9. Complete real CLI-to-optimizer call trace

Real scientific Stage-1 commands are handled by `scripts/stage1_runner.py`:

```
main()
  -> run_lr_pilot() / run_r_phase1() / run_final_main()
  -> _verified_corpus()
      -> verify_prepared_corpus()
  -> _load_selection() for downstream stages
  -> _execute()
      -> unmark.stage1.execute.execute_stage()
          -> verify_scientific_inputs()
          -> require_deterministic_cublas_workspace()
          -> resolve_scientific_device()
          -> enforce_numerical_policy()
          -> verify_numerical_policy()
          -> current_fingerprint()
          -> load_prepared_chunks()
          -> build_backbone()
          -> construct held-out validation batches
          -> create persistent PreparationPool
          -> for each PlannedRun:
              -> RunProvenance
              -> fresh_adapter()
              -> UnmarkEncoder(shared frozen encoder, fresh adapter)
              -> Stage1Objective(...)
              -> train_run()
                  -> verify_model_contract()
                  -> build_optimizer()
                  -> DeterministicSampler()
                  -> optional checkpoint restore
                  -> update 0 validation
                  -> sampler.next_batch()
                  -> preparation_pool.prepare() or serial prepare
                  -> collate + batch_to_device
                  -> objective(batch)
                  -> optimizer.zero_grad(set_to_none=True)
                  -> loss.backward()
                  -> gradient_report()
                  -> optimizer.step()
                  -> validation/checkpoint/budget
```

The single Stage-1 production scientific optimizer step is
`unmark/stage1/trainer.py:677`. The corresponding backward call is
`unmark/stage1/trainer.py:668`.

Alternate/bypass paths:

- `scripts/stage1_runner.py smoke` is structurally no-update: it constructs no
  optimizer and calls no backward path.
- `prepare-corpus` is Stage-6 CPU preparation and has no optimizer.
- `scripts/stage1_pretrain_measurements.py` is a no-step benchmark/measurement
  tool and explicitly excludes optimizer steps.
- `unmark/evaluation/preg1_head.py` contains a separate pre-G1 head optimizer
  path, not Stage-1 adapter training.
- Pytest fixtures include TEST-ONLY `optimizer.step()` and `backward()` calls.
  These are not scientific Stage-1 training.

Current path verdict: intended single route exists, but the real Stage-1
execution path fails before `train_run()` because `objective_cls` is unbound in
`execute_stage()`.

## 10. Static bug/refactor review

Repository-wide static inspection found the following material bugs:

1. `execute_stage()` imports `Stage1Objective` but constructs
   `objective = objective_cls(...)`; `objective_cls` is not bound in that
   function.
2. `ValidationPoint.to_dict()` writes derived `score`, but production resume
   uses `ValidationPoint(**p)`, which cannot accept `score`.
3. Stage-level resume always calls the first `train_run()` leg with
   `cap=INITIAL_MAX_UPDATES`, ignoring the checkpoint's recorded cap, so a
   checkpoint from the 20k->40k continuation cannot be reconstructed after a
   process death.
4. Downstream selection artifact loading validates only `stage` and
   `protocol_version`.
5. Stage-1 training provenance records a caller-supplied optional
   `--repository-head`, unlike Stage-6 which derives actual `git rev-parse HEAD`.

Other static observations:

- `strict=False` appears only in comments/documentation about a previous
  checkpoint defect, not in current production adapter restore.
- Broad `except Exception` blocks in Stage-1 execution are for non-load-bearing
  measurement/version reporting or wrap-and-reraise worker failures.
- TODO/FIXME/HACK/XXX search found one real TODO in
  `requirements/experiment.txt:24-29` about future environment splitting. It is
  not a Stage-1 launch blocker under the current frozen transformer pin.
- Production Stage-1 randomness does not use Python `hash()`. One TEST-ONLY
  resume fixture uses `hash()` in a synthetic tensor generator.

## 11. Checkpoint/resume end-to-end review

Writer path:

- Validation points are built by validation code as `ValidationPoint` objects.
- `checkpoint_payload()` serializes `points` with
  `[p.to_dict() if hasattr(p, "to_dict") else dict(p) for p in points]`
  (`trainer.py:324`).
- `ValidationPoint.to_dict()` writes `update`, `distances`, `d_clean`, and
  derived `score` (`selection.py:72-78`).
- Payload also writes schema version, provenance, adapter-only state,
  optimizer state, global update, sampler state, cap, budget flag, and
  execution fingerprint.
- `save_training_checkpoint()` publishes last and best checkpoint via
  temp-write, flush, fsync, atomic replace, and directory fsync.

Reader path:

- `load_training_checkpoint()` loads last or best with `torch.load(...,
  map_location="cpu")`.
- `verify_checkpoint()` checks required keys, schema version, and
  `RunProvenance.require_match()`.
- `train_run()` strictly restores adapter state, optimizer state, optimizer
  parameter identity, optimizer state device, sampler state, and global update.
- It then reconstructs validation history with
  `[ValidationPoint(**p) for p in resume.get("points", [])]`.

Writer/reader incompatibility:

```
ValidationPoint.to_dict() -> {"update", "distances", "d_clean", "score"}
ValidationPoint(**p)      -> accepts only update, distances, d_clean
```

Read-only probe result:

```
TypeError: ValidationPoint.__init__() got an unexpected keyword argument 'score'
```

Impact:

- Any real checkpoint written after update 0/500/... cannot be resumed by
  production `train_run()`.
- The in-process 20k->40k continuation also reloads the checkpoint and therefore
  hits the same reader.
- Existing tests write/read checkpoint payloads, but they pass raw point dicts
  around or compare dictionaries; they do not execute the production
  `ValidationPoint(**p)` reader on a writer-emitted `ValidationPoint.to_dict()`.

Checkpoint/resume verdict: BLOCKED. Adapter/optimizer/sampler/execution
restore logic is otherwise well structured, but the validation-history schema
breaks legitimate production resume and continuation.

## 12. 20k->40k state-machine review

Derived source state machine:

- Fresh run starts at `global_update = 0`.
- Update 0 validation occurs before any optimizer step.
- Training steps to `cap`, validating and checkpointing every 500 updates.
- At `cap=20_000`:
  - if selected checkpoint is before 20k, stop;
  - if selected checkpoint is exactly 20k, set `result.cap = 40_000` and
    `result.continued = True`.
- In-process continuation reloads the run checkpoint and calls `train_run()` a
  second time with `cap=result.cap`.
- At `cap=40_000`:
  - if selected checkpoint is before 40k, stop;
  - if selected checkpoint is exactly 40k, stop and mark budget limited.

Crash/resume cases:

- Crash before first checkpoint: stage resume has no checkpoint and restarts the
  run from scratch. This is acceptable because no scientific checkpoint was
  committed.
- Crash after first checkpoint but before 20k: current production resume fails
  because of the `ValidationPoint` `score` reader bug.
- Crash exactly at 20k before continuation: current production resume fails
  for the same reason.
- In-process 20k continuation: current production continuation fails for the
  same reason when it reloads the 20k checkpoint.
- Crash after a continuation checkpoint above 20k: even after the
  `ValidationPoint` reader bug is repaired, `run_lr_pilot()`/`run_r_phase1()`/
  `run_final_main()` will call the first `train_run()` leg with
  `cap=INITIAL_MAX_UPDATES` (`execute.py:325-342`). `train_run()` ignores
  `resume["cap"]` and initializes `RunResult(cap=cap)` from the caller
  (`trainer.py:560-622`). A checkpoint with `global_update > 20_000` is therefore
  resumed under a 20k cap, which cannot reconstruct the 40k leg.
- No source-supported path intentionally creates a third 60k/80k continuation;
  `budget_decision()` only permits 20k and 40k caps and rejects other caps.

20k->40k verdict: BLOCKED. The mathematical budget rule is correctly encoded in
`selection.py`, but production resume/orchestration cannot robustly reconstruct
the continuation leg.

## 13. Nominal-run independence

Source review:

- `selection.py` schedules 3 LR candidates, 5 `r` candidates, and 3 final-main
  seeds. `total_planned_runs()` asserts 11.
- LR and `r` selection candidates use `SELECTION_SEED = 21230`.
- Final-main seeds are `36930`, `7309`, and `5993`.
- `execute_stage()` loads tokenizer/frozen encoder once per stage, then creates
  a fresh adapter inside the schedule loop (`execute.py:295`), wraps it in a new
  `UnmarkEncoder`, and constructs a new objective.
- `trainer.py` creates a fresh optimizer and sampler for each `train_run()`.
- Per-run checkpoint namespace is `output_dir/run-<label>/_checkpoint`.
- Shared objects are frozen encoder, tokenizer, validation prepared batches,
  classifier/inventory, and the persistent preparation pool. These are intended
  immutable or operationally pure with respect to scientific state.
- Frozen encoder state hash is checked after each nominal run
  (`execute.py:97-124`, `execute.py:373-374`).

Verdict: source structure preserves nominal-run independence, but current B1
prevents this code path from being exercised in real training.

## 14. Frozen PhoBERT/objective/gradient isolation

Evidence:

- `build_backbone()` sets every encoder parameter `requires_grad_(False)` and
  calls `encoder.eval()` (`execute.py:87-90`).
- `UnmarkEncoder.train()` keeps the adapter in caller mode while forcing the
  encoder back to eval (`adapter.py:562-577`).
- `UnmarkEncoder.forward()` requires frozen encoder eval before every forward
  (`adapter.py:544-557`, `adapter.py:623-629`).
- `verify_model_contract()` rejects encoder trainable parameters and encoder
  training mode (`trainer.py:201-232`).
- `gradient_report()` rejects any frozen encoder gradients
  (`trainer.py:268-289`).
- Optimizer construction is passed only `named_parameters()` with
  `requires_grad` from `objective.unmark_encoder` (`trainer.py:597-600`).
- Checkpoints persist `adapter.state_dict()` only (`trainer.py:699-701`), not
  the frozen encoder.

Verdict: frozen encoder/objective/gradient isolation is consistent in source
and supported by tests/historical CUDA evidence. It is not the current blocker.

## 15. Optimizer review

Source behavior:

- `build_optimizer()` constructs AdamW lazily over supplied trainable adapter
  parameters only (`optim.py:70-101`).
- It refuses frozen parameters (`optim.py:82+`).
- Decay group: adapter fusion/gate weight matrices at weight decay `0.01`.
- Exempt group: bias, LayerNorm, tone embedding, and letter embedding at weight
  decay `0.0`.
- AdamW uses betas `(0.9, 0.999)`, epsilon `1e-8`, `amsgrad=False`, constant LR.
- Stage-1 protocol encodes no warmup, no clipping, and accumulation 1.
- `require_optimizer_parameter_identity()` checks no foreign, duplicate, or
  missing adapter parameters (`trainer.py:342-373`).
- `require_optimizer_state_device()` checks real optimizer state tensors are on
  the training device while allowing scalar CPU Adam `step` tensors
  (`trainer.py:375-405`).

Verdict: optimizer contract matches proposal/decisions/freeze/code/tests in
source. Local torch-dependent tests skipped because local `.venv` has no torch;
historical CUDA evidence remains relevant for runtime behavior but not for the
current launch/resume blockers.

## 16. Randomness and determinism

Scientific RNG consumers:

- Corruption uses deterministic keyed BLAKE2b-style decisions via schema,
  seed, sample id, canonical text, and unit index. No global RNG is used.
- Training sampler uses stable hash ranking over
  `stage1-order|seed|visit|chunk_id` (`sampler.py:38-43`) and checkpointed
  `(seed, visit, position, corpus_digest)`.
- Adapter initialization uses `torch.random.fork_rng(devices=[])` and
  `torch.default_generator.manual_seed(init_seed)` on CPU only
  (`initialisation.py:29-57`).
- Validation corruption uses dedicated validation seed `19225`, independent of
  run seed.
- LR/r selection candidates intentionally share run seed `21230`, hence share
  H0 initialization by design.

Non-scientific time/RNG:

- Timestamps and `datetime.now()` appear in diagnostic/probe scripts and older
  evaluation tooling, not Stage-1 scientific training identity.
- `time.monotonic()`/`perf_counter()` is used for progress/timing only.
- TEST-ONLY fixtures use `hash()` in a miniature tensor generator; production
  Stage-1 does not.

Verdict: deterministic scientific randomness is correctly isolated in source.
No worker-side scientific RNG consumption was found.

## 17. CUDA and numerical contract

Source guarantees:

- `require_deterministic_cublas_workspace()` sets or verifies deterministic
  `CUBLAS_WORKSPACE_CONFIG` before CUDA initialization (`device.py:43-65`).
- `resolve_scientific_device()` requires CUDA and refuses CPU fallback
  (`device.py:68-85`).
- `enforce_numerical_policy()` enables deterministic algorithms, cuDNN
  deterministic mode, disables cuDNN benchmark, sets float32 matmul precision
  to `highest`, and disables CUDA TF32 (`device.py:88-107`).
- `verify_numerical_policy()` re-checks the policy (`device.py:109-141`).
- `ExecutionFingerprint.RESUME_BLOCKING` includes backend, GPU name, compute
  capability, torch/CUDA/cuDNN versions, deterministic flags, cuBLAS workspace,
  matmul precision, and TF32 state (`device.py:171-183`).

Local review did not rerun CUDA because torch/CUDA are unavailable in `.venv`.
Historical CUDA evidence remains necessary for runtime acceptance.

Verdict: CUDA/numerical source contract is strong, but current training is still
blocked before runtime evidence can license launch.

## 18. Data preparation and Stage-6 review

Source behavior:

- Corpus pin checks dataset identity, revision, shard names, byte sizes, and
  SHA-256 before reading rows (`corpus.py:133+`).
- Concatenation follows locked `train.parquet`, `validation.parquet`,
  `test.parquet` order and refuses duplicate document ids (`corpus.py:229-254`).
- Contamination screen accepts only already opened UIT-VSFC derived train and
  official validation references; any other reference key, including official
  TEST, is refused (`corpus.py:402-444`).
- Document split occurs before chunking; dev count and split seed are locked.
- Chunking enforces max length 256 with overflow failure and no truncation.
- Stage-6 checkpoint identity derives actual repository HEAD using
  `resolve_repository_head()` (`stage1_runner.py:194-198`,
  `checkpoint.py:335+`).
- `COMPLETE.json` binds relative artifacts by size and SHA-256, and
  `verify_prepared_corpus()` re-hashes `chunks.jsonl` and `manifest.json`
  before Stage-1 training can load data (`checkpoint.py:894-989`).
- Manifest compatibility requires `official_test_used=false` and
  `official_test_screened=false` (`manifest.py:221-272`).

Verdict: Stage-6 prepared-corpus identity and data safety remain strong. Stage-1
cannot silently accept a different completed corpus through the official runner
without failing the prepared-corpus verification.

## 19. Parallel preparation review

Stage-1 online preparation:

- `PREPARATION_WORKERS = 8` and start method `"spawn"` are explicit in
  `preparation.py`.
- Worker initializer loads the pinned tokenizer and verified inventory in each
  fresh interpreter.
- The main process owns sampler advancement, batch membership, optimizer,
  objective, CUDA tensors, and checkpoints.
- `ProcessPoolExecutor.map()` returns results in input order
  (`preparation.py:232`).
- No prefetch/lookahead is encoded; one sampler batch is prepared synchronously.
- Worker exceptions are wrapped and re-raised loudly, not silently converted to
  serial fallback.
- Pool lifecycle shuts down with `wait=True`.

Stage-6 parallel chunking:

- `parallel.py` is CPU-only preparation before scientific CUDA training.
- It preserves document order and bounded in-flight work; output equality for
  1/2/4/8 workers is tested.

Local test note: sandboxed multiprocessing could not bind a forkserver socket
for `tests/test_stage1_parallel.py`; rerunning that suite outside the sandbox
passed.

Verdict: parallel preparation semantics are sound in source/tests and not a
current blocker.

## 20. Selection and stage-artifact handoff

Internal selection rules:

- `select_checkpoint()` requires update 0, rejects duplicate updates, and
  selects lowest worst-case score, then lower clean distance, then earliest
  update (`selection.py:81-93`).
- `select_learning_rate()` requires exactly the locked LR grid at `r=1.0` and
  breaks ties by score, clean distance, then lower LR (`selection.py:180-191`).
- `select_r()` requires exactly the locked r grid at the frozen LR and breaks
  ties by score, clean distance, then smaller `r` (`selection.py:194-212`).

Artifact consumer attack matrix for `_load_selection()`:

| Attack | Current behavior |
|---|---|
| wrong stage | REFUSES |
| wrong protocol version | REFUSES |
| wrong repository HEAD | PERMITS |
| wrong corpus digest | PERMITS |
| wrong model/revision | PERMITS |
| wrong inventory | PERMITS |
| stale artifact from another campaign | PERMITS if stage/protocol match |
| selected value not in locked candidate grid | PERMITS |
| missing/duplicated candidates | PERMITS in downstream consumers |
| selected metric edited | PERMITS |
| candidate metric edited | PERMITS unless writer-side selection is rerun |

The pure selection functions are strict, but downstream CLI handoff does not
reconstruct candidates and rerun them. This creates an incompatible assumption:
the writer encodes candidate evidence, while the reader trusts only the selected
scalar.

Verdict: BLOCKED by Finding 031-B4.

## 21. Git and repository provenance

Stage-6:

- `prepare-corpus` derives actual HEAD through `resolve_repository_head()` and
  embeds it in `CheckpointIdentity`.
- Stage-6 checkpoints refuse identity mismatches.

Stage-1 training:

- `_corpus_consumer()` accepts optional `--repository-head`, default `None`
  (`scripts/stage1_runner.py:579-580`).
- `_execute()` passes `args.repository_head` to `execute_stage()`
  (`scripts/stage1_runner.py:505-515`).
- `execute_stage()` records that value in `RunProvenance` and stage artifacts
  (`execute.py:280-288`, `execute.py:389-394`).
- No current code compares the provided value to actual `git rev-parse HEAD`.

This is more than a declaration weakness: a real Stage-1 run can execute one
checkout while recording another SHA or `None`, and resume will then compare
against the false value because it is part of checkpoint provenance.

Verdict: BLOCKED by Finding 031-B5.

## 22. CLI/operator executability

Source-supported command forms:

```
.venv-colab/bin/python scripts/stage1_runner.py prepare-corpus \
  --corpus-root <uvw-root> \
  --output-dir <prepared-corpus> \
  [--uitvsfc-derived-train <opened-train-csv>] \
  [--uitvsfc-official-validation <opened-validation-csv>] \
  [--revision <locked-sha>] \
  [--prepare-workers <n>] \
  [--checkpoint-dir <stage6-checkpoint-dir>]

.venv-colab/bin/python scripts/stage1_runner.py lr-pilot \
  --prepared-corpus <prepared-corpus> \
  [--completion-dir <complete-dir>] \
  --output-dir <lr-output> \
  --cache-root <cache-root> \
  [--resume] \
  [--revision <locked-sha>] \
  [--repository-head <sha>]

.venv-colab/bin/python scripts/stage1_runner.py r-phase1 \
  --prepared-corpus <prepared-corpus> \
  [--completion-dir <complete-dir>] \
  --output-dir <r-output> \
  --cache-root <cache-root> \
  --lr-artifact <lr-output/lr_pilot.json> \
  [--resume] \
  [--revision <locked-sha>] \
  [--repository-head <sha>]

.venv-colab/bin/python scripts/stage1_runner.py final-main \
  --prepared-corpus <prepared-corpus> \
  [--completion-dir <complete-dir>] \
  --output-dir <final-output> \
  --cache-root <cache-root> \
  --lr-artifact <lr-output/lr_pilot.json> \
  --r-artifact <r-output/r_phase1.json> \
  [--resume] \
  [--revision <locked-sha>] \
  [--repository-head <sha>]
```

Fresh commands refuse an existing output directory. Resume commands require the
output directory to exist.

The first real LR command cannot reach training construction successfully under
current source because `execute_stage()` uses an unbound `objective_cls`.

Operator/CLI verdict: BLOCKED.

## 23. Output/checkpoint/disk safety

Positive evidence:

- Stage-6 uses durable prefix checkpointing, immutable shard publication,
  manifest/complete markers, temp writes, fsync, atomic replace, and verification.
- Stage-1 training checkpoint names are per run:
  `training-checkpoint-last.pt` and `training-checkpoint-best.pt`.
- Per-run namespaces prevent candidate checkpoint collisions.
- Fresh output directories are immutable unless explicit `--resume` is used.
- Stage-1 training checkpoints store adapter-only state and optimizer/sampler
  state, not full frozen encoder state.
- Stage result artifacts record `raw_text_persisted=false`.

Non-blocking weaknesses:

- `REQUIRED_CHECKPOINT_KEYS` omits writer field `budget_limited`, so the schema
  does not require all writer state it claims to persist.
- Checkpoint payload writes `budget_limited=result.budget_limited` before
  `resolve_budget()` is called, so the final checkpoint at a hard cap can carry
  stale budget-limited metadata. Current reader does not consume that field, so
  this is an observability/schema strictness issue rather than a current
  scientific state corruption path.

Verdict: disk publication mechanics are strong, but recovery is blocked by
checkpoint reader/state-machine bugs.

## 24. UIT-VSFC TEST sealing

Repository-wide UIT-VSFC searches distinguished official TEST from pytest and
generic software-test terminology.

Findings:

- Stage-1 runner has no official TEST CLI argument.
- `prepare-corpus` accepts only `--uitvsfc-derived-train` and
  `--uitvsfc-official-validation`.
- `screen_contamination()` refuses unlisted reference keys and explicitly keeps
  `official_test_screened=false`.
- Manifest compatibility requires `official_test_used=false` and
  `official_test_screened=false`.
- Selection uses only Stage-1 held-out unlabeled distances and no downstream
  labels or official TEST scores.
- No command in this review opened or passed official TEST.

Verdict: official UIT-VSFC TEST remains SEALED / UNUSED.

## 25. Test-quality review

The test suite is large and useful, but it currently gives false confidence on
two critical production paths:

- Some launch-readiness tests are AST/source-text tests. In particular,
  `tests/test_stage1_run_independence.py` expects an `objective_cls` call inside
  the schedule loop, which helped preserve the stale unbound symbol rather than
  verifying that it is bound.
- Checkpoint/resume tests verify helper-level payload persistence and synthetic
  resume equivalence. They pass raw dictionaries or compare payload dictionaries
  rather than driving production `train_run()` resume through
  `ValidationPoint(**p)` on writer-emitted validation points.
- CUDA resume equivalence is valuable for optimizer/device state behavior but
  does not cover the real Stage-1 checkpoint validation-history reader or the
  stage-level 20k->40k crash-resume cap reconstruction.
- CLI tests verify parser/handler shape and smoke no-step behavior, but they do
  not execute the real `execute_stage()` construction path to the point where
  `objective_cls` is resolved.

Verdict: tests are extensive, but the gaps above are material and explain why
the current blockers survive a green full suite.

## 26. Cross-file and cross-module conflict review

Question: Are there two files/modules that each appear correct independently
but make incompatible assumptions about the same data/state/schema?

Answer: yes.

Concrete conflicts:

1. `selection.ValidationPoint.to_dict()` writes an artifact schema with derived
   `score`, while `trainer.train_run()` assumes checkpoint point dictionaries
   are constructor-compatible with `ValidationPoint`.
2. `execute_stage()` imports `Stage1Objective` but uses the older `objective_cls`
   name retained from `build_objective()`/smoke code.
3. `selection.py` strictly validates candidate grids, but
   `scripts/stage1_runner.py` downstream handoff reads only `selected` scalars
   from prior artifacts and does not rerun the strict selection validators on
   the artifact evidence.
4. `checkpoint.py`/Stage-6 treats actual git HEAD as identity, while
   `scripts/stage1_runner.py`/Stage-1 training treats repository HEAD as an
   optional caller declaration.
5. `trainer.checkpoint_payload()` writes `cap`, but `train_run()` stage-level
   resume uses the caller's cap and ignores the checkpoint cap, so the 40k
   continuation leg cannot be reconstructed after process death.

Cross-file conflict verdict: BLOCKED.

## 27. Requirements/environment review

Requirements:

- `requirements/dev.txt` intentionally installs only local lightweight tools:
  `PyYAML` through `base.txt` and `pytest>=8,<10`.
- `requirements/experiment.txt` intentionally excludes torch because the Colab
  runtime supplies CUDA PyTorch via `--system-site-packages`.
- `transformers==4.57.6` is pinned for experiment use.
- `sentencepiece>=0.2,<0.3`, `safetensors>=0.4`, and `accelerate>=0.30` are
  listed for Colab/runtime support.

Local `.venv`:

- Python 3.14.5, pytest 9.1.1, PyYAML 6.0.3.
- No torch, transformers, tokenizers, sentencepiece, safetensors, or accelerate.

Environment verdict: local `.venv` is consistent with the repository's
lightweight local policy. Stage-1 scientific runtime still depends on the
authoritative CUDA/Colab environment, not this local environment.

## 28. Historical CUDA evidence validity

Known historical accepted runtime evidence included Python 3.13.15,
torch 2.11.0+cu128, CUDA 12.8, Transformers 4.57.6, RTX PRO 6000 Blackwell,
compute capability 12.0, deterministic policy, H0 hashes, frozen-encoder mode
gate, zero-update real-data probe, checkpoint/resume probe, synthetic CUDA
resume equivalence, persistent spawn acceptance, and integrated no-step
benchmark.

Current classification:

| Evidence | Current classification | Reason |
|---|---|---|
| Deterministic CUDA/FP32 policy evidence | STILL VALID for unchanged source | `device.py` scientific policy unchanged since the audited code |
| H0 hashes and adapter init evidence | STILL VALID for unchanged source | scientific code unchanged from `479fac5...`; init source matches |
| Frozen encoder mode/gradient gate | STILL VALID for unchanged source | source gates remain present |
| Zero-update real-data probe | STILL VALID for no-step model/data boundary | scientific code unchanged; no optimizer step |
| Persistent spawn preparation acceptance | STILL VALID for preparation semantics | `preparation.py` unchanged; local tests pass |
| Integrated no-step benchmark | STILL VALID for performance context | documentation-only commits since evidence |
| Checkpoint/resume probe | PARTIALLY VALID | useful for optimizer/device/sampler primitives, but not production `ValidationPoint` reader or continuation cap |
| Synthetic CUDA resume equivalence | INVALIDATED for production resume claim | it does not exercise real `train_run()` checkpoint point reconstruction |
| Full launch readiness | MUST RE-RUN BEFORE TRAINING after repairs | current source cannot launch and cannot resume correctly |

No expensive rebenchmark is required merely because current HEAD added
documentation-only Audit 030 changes. Runtime reacceptance is required after
any repair that changes the production launch/resume path.

## 29. Tests actually executed

All Python commands used `.venv/bin/python`.

Read-only checkpoint schema probe:

```
.venv/bin/python -B -c "from unmark.stage1.selection import ValidationPoint; ..."
```

Result: reproduced `TypeError: ValidationPoint.__init__() got an unexpected
keyword argument 'score'`. No training or optimizer step occurred.

Targeted Stage-1 suite:

```
.venv/bin/python -B -m pytest -q -p no:cacheprovider \
  tests/test_stage1_final_freeze.py \
  tests/test_stage1_pretrain_audit.py \
  tests/test_stage1_run_independence.py \
  tests/test_stage1_run_independence_runtime.py \
  tests/test_stage1_runner_contract.py \
  tests/test_stage1_runner_cli_contract.py \
  tests/test_stage1_training_resume_state.py \
  tests/test_stage1_training_resume.py \
  tests/test_stage1_cuda_resume_equivalence.py \
  tests/test_stage1_schedule.py \
  tests/test_stage1_corpus_verification.py \
  tests/test_stage1_corpus.py \
  tests/test_stage1_checkpoint.py \
  tests/test_stage1_prepare_cli.py \
  tests/test_stage1_parallel_preparation.py \
  tests/test_stage1_corruption_scope.py \
  tests/test_stage1_device_contract.py \
  tests/test_stage1_device_contract_runtime.py \
  tests/test_stage1_validation_preparation.py \
  tests/test_stage1_validation_measurement.py \
  tests/test_stage1_inventory_preflight.py \
  tests/test_stage1_measurement_contract.py \
  tests/test_stage1_lengths.py \
  tests/test_stage1_parallel.py \
  tests/test_stage1_chunking.py \
  tests/test_stage1_source_coordinates.py \
  tests/test_stage1_non_vietnamese_orthography.py \
  tests/test_corruption.py
```

Sandbox result: `1411 passed, 6 skipped, 7 failed in 105.48s`. All seven
failures were `PermissionError: [Errno 1] Operation not permitted` when
sandboxed multiprocessing/forkserver attempted to bind a local UNIX socket in
`tests/test_stage1_parallel.py`.

Rerun of the failing multiprocessing suite outside the sandbox:

```
.venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_stage1_parallel.py
```

Result: `18 passed in 12.58s`.

Full lightweight suite outside the sandbox:

```
.venv/bin/python -B -m pytest -q -p no:cacheprovider
```

Result: `3622 passed, 103 skipped in 134.81s`.

Skips: expected local/runtime skips for unavailable heavy ML dependencies and
CUDA-gated tests in the local `.venv`.

No real Stage-1 command (`lr-pilot`, `r-phase1`, or `final-main`) was run. No
real scientific optimizer step was intentionally executed.

## 30. Findings

### 031-B1 BLOCKER - Stage-1 `execute_stage()` cannot construct the objective

Source:

- `unmark/stage1/execute.py:186` imports `Stage1Objective`.
- `unmark/stage1/execute.py:299` calls `objective = objective_cls(...)`.
- No `objective_cls` name is bound in `execute_stage()`.

Concrete reachable failure:

The first `lr-pilot` scientific run verifies inputs, loads data/model, creates a
fresh adapter, then raises `NameError` at objective construction before
`train_run()` is called. The first real LR command cannot reach the scientific
optimizer path.

Why it matters:

This is a direct execution failure in the Stage-1 training path.

Existing tests:

Not caught. `tests/test_stage1_run_independence.py` source-inspects the loop and
requires the stale `objective_cls` call, so it preserves the refactor error.
CLI tests do not execute this construction path.

Runtime evidence:

Historical CUDA/no-step evidence does not cover current `execute_stage()`
construction through this line.

### 031-B2 BLOCKER - Production checkpoints cannot be resumed because validation point writer/reader schemas disagree

Source:

- `unmark/stage1/selection.py:72-78` writes `score` in
  `ValidationPoint.to_dict()`.
- `unmark/stage1/trainer.py:324` serializes validation points with
  `p.to_dict()`.
- `unmark/stage1/trainer.py:621` reconstructs with `ValidationPoint(**p)`,
  whose constructor has no `score` field.

Concrete reachable failure:

Any real training checkpoint written after validation contains points with
`score`. On resume, production `train_run()` raises:

```
TypeError: ValidationPoint.__init__() got an unexpected keyword argument 'score'
```

The same failure also breaks the in-process 20k->40k continuation because
`execute_stage()` reloads the first-leg checkpoint before the second leg.

Why it matters:

It materially breaks legitimate checkpoint recovery and the locked continuation
mechanism.

Existing tests:

Not caught. Tests persist points but use raw dictionaries or compare payloads
without passing writer-emitted points through the production reader.

Runtime evidence:

Historical resume evidence is invalid for this production reader.

### 031-B3 BLOCKER - 20k->40k continuation checkpoint cannot be reconstructed after process death

Source:

- Fresh/resume stage call passes `cap=INITIAL_MAX_UPDATES` in the first
  `train_run()` call (`unmark/stage1/execute.py:325-342`).
- `train_run()` initializes `RunResult(cap=cap)` from the caller and ignores
  `resume["cap"]` (`unmark/stage1/trainer.py:560-622`).
- `budget_decision()` rejects selected updates beyond the caller cap
  (`unmark/stage1/selection.py:128-129`).

Concrete reachable failure:

If a run continues past 20k and the process dies after a checkpoint at, for
example, update 20,500, the next `--resume` stage invocation enters the first
leg with cap 20k even though the checkpoint belongs to the 40k leg. The resumed
run cannot continue correctly and can raise because selected update exceeds the
20k cap or stop under the wrong budget state.

Why it matters:

This breaks many-hour recovery for the only allowed continuation and can prevent
completion of a legitimate 40k run.

Existing tests:

Not caught. Current budget tests cover pure decision functions; resume tests do
not cover stage-level crash resume from the continuation leg.

Runtime evidence:

Historical CUDA resume equivalence does not cover the stage-level 20k->40k
state machine.

### 031-B4 BLOCKER - Downstream stage handoff artifacts are under-validated

Source:

- `_load_selection()` checks only file existence, `stage`, and
  `protocol_version` (`scripts/stage1_runner.py:475-487`).
- `run_r_phase1()` trusts `pilot["selected"]["learning_rate"]`
  (`scripts/stage1_runner.py:446-454`).
- `run_final_main()` trusts selected LR and r from artifacts, checking only that
  r-phase1's selected LR equals the pilot-selected LR
  (`scripts/stage1_runner.py:457-472`).

Concrete reachable failure:

A stale, foreign, hand-edited, wrong-corpus, wrong-HEAD, wrong-inventory, or
wrong-grid artifact with matching `stage` and `protocol_version` can drive
`r-phase1` or `final-main`. The consumer does not verify that the selected
value came from the locked candidates under the current corpus/model/inventory
identity.

Why it matters:

This can create incorrect campaign results while every downstream command
appears to run under the frozen protocol.

Existing tests:

Not caught. Pure selection validators are strict, but the real artifact
consumer does not call them on loaded artifact evidence.

Runtime evidence:

No historical runtime evidence covers malformed/stale selection handoff.

### 031-B5 BLOCKER - Stage-1 training repository HEAD provenance is optional and caller-supplied

Source:

- `--repository-head` defaults to `None` and is described as recorded
  provenance (`scripts/stage1_runner.py:579-580`).
- `_execute()` passes `args.repository_head` unchanged into `execute_stage()`
  (`scripts/stage1_runner.py:505-515`).
- `execute_stage()` records it in `RunProvenance` and stage artifacts
  (`unmark/stage1/execute.py:280-288`, `unmark/stage1/execute.py:389-394`).
- Stage-6 contrasts with this by deriving actual HEAD via
  `resolve_repository_head()` (`scripts/stage1_runner.py:194-198`).

Concrete reachable failure:

An operator can run one checkout while recording another commit SHA or `None`.
The checkpoint resume gate then compares against that declared value rather
than the executing tree. A scientifically meaningful Stage-1 artifact can
therefore misidentify the code that produced it.

Why it matters:

Repository identity is part of reproducibility and checkpoint compatibility.
This is a concrete provenance/integrity failure for the planned campaign.

Existing tests:

Not caught for Stage-1 training. Stage-6 tests cover derived HEAD in the
preparation checkpoint identity, but the training path does not share that
contract.

Runtime evidence:

Historical CUDA evidence records runtime behavior, not actual-vs-declared
repository identity for training commands.

### 031-M1 MINOR - Current HEAD and freeze repository-head metadata are documentation-stale

Current HEAD is `55aa406...`, not the prompt-expected `855a1d3...`. The delta is
documentation-only. The freeze file still records repository head
`649ad741b8e737e7a108e71a47b818bf8ea991b2` under
`docs/spec/stage1-final-freeze.json`. Since scientific code is byte-identical
across `479fac5...`, `855a1d3...`, and `55aa406...`, this is not a training
path blocker, but the metadata is stale.

### 031-M2 MINOR - README is stale relative to current repository state

`README.md:5-6` still says the repository is at Phase 0/G-1 and that nothing
from UNMARK itself is implemented yet. Stage-1/Stage-6 code now exists. This is
documentation-only because proposal/decisions/freeze and code are more
authoritative for this review.

### 031-M3 MINOR - Checkpoint schema and budget flag strictness are incomplete

`checkpoint_payload()` writes `budget_limited`, but
`REQUIRED_CHECKPOINT_KEYS` omits it (`trainer.py:315-339`). The writer also
passes `result.budget_limited` into checkpoint payload before `resolve_budget()`
sets the final budget state (`trainer.py:696-708`). Current resume does not use
that field, so this is not the active recovery blocker, but it is a schema and
observability defect.

### 031-M4 MINOR - `load_prepared_chunks()` treats any non-`train` partition as dev

`unmark/stage1/execute.py:61` routes rows to train only when
`row["partition"] == "train"` and otherwise to dev. The official runner verifies
the prepared corpus first, so malformed partitions should not reach this helper
through the production path. As a local helper contract, however, it is less
fail-closed than the manifest/Stage-6 schema.

### 031-M5 MINOR - Tests mask production launch/resume failures

The full suite passes, but source-inspection and helper-level tests miss the
real `execute_stage()` construction failure and the production checkpoint
reader failure. This is test-quality debt, not a separate production behavior
beyond B1/B2/B3.

### 031-I1 INFORMATIONAL - Local `.venv` is intentionally ML-free

Torch, transformers, tokenizers, sentencepiece, safetensors, and accelerate are
not installed locally. This matches local policy. Scientific runtime evidence
must come from the authorized CUDA environment.

### 031-I2 INFORMATIONAL - No real scientific optimizer step was executed

This review did not run `lr-pilot`, `r-phase1`, or `final-main`, and did not
intentionally execute a real scientific optimizer step. Any optimizer steps
reachable under pytest are TEST-ONLY fixtures.

### 031-I3 INFORMATIONAL - Stage-1 and Stage-6 parallelism have different roles

Stage-1 online preparation uses explicit spawn workers and no prefetch.
Stage-6 chunking is CPU-only and may use multiprocessing differently. This is
not a scientific mismatch because Stage-6 output identity is verified by
artifact hashes and equality tests.

### 031-I4 INFORMATIONAL - Official UIT-VSFC TEST references remain sealed

Repository references to TEST are either explicit sealing statements, manifest
false fields, or software-test terminology. No route to official TEST was found
in Stage-1 production.

Finding counts:

- BLOCKER: 5
- MAJOR: 0
- MINOR: 5
- INFORMATIONAL: 4

## 31. Current-state synthesis

Scientifically frozen:

- Backbone `vinai/phobert-base` at revision
  `01daacda68afe13d83023d16ec647239e344a1e6`.
- Hidden size 768, FP32, frozen encoder, recursive eval behavior.
- Adapter architecture and trainable parameter count 3,551,232.
- Adapter initialization derived from run seed only.
- UVW-2026 corpus revision
  `a0a79294e4568137e25828bb3f2a4cde8546e1fb`, fixed shard order, fixed
  document split, max length 256, no truncation, overflow fail.
- Contamination scope excludes official TEST.
- Corruption seed, validation seed, PI_STRIP, per-visit redraw semantics.
- AdamW optimizer parameters and no warmup/clipping/accumulation.
- Campaign grids, selection seed, final seeds, and 11 nominal runs.
- Worst-case validation selection and one 20k->40k continuation rule.

Exact implementation that would train if launch were repaired:

`scripts/stage1_runner.py` would call `execute_stage()`, which would verify the
prepared corpus, inputs, CUDA/numerical policy, load frozen PhoBERT, share it
immutably across nominal runs, create a fresh adapter/objective/optimizer/sampler
per nominal run, use the persistent spawn preparation pool for per-batch
corruption/tokenization, run `loss.backward()`, check gradients, and perform the
single production scientific `optimizer.step()` in `trainer.py`.

Exact dataset/corpus:

The prepared Stage-6 artifact derived from all three UVW shards in locked order,
after opened-material contamination screening, document split before chunking,
and verified `COMPLETE.json`/manifest/chunk hashes.

All 11 nominal runs:

Three LR candidates at seed `21230`, five r candidates at seed `21230`, and
three final-main runs at seeds `36930`, `7309`, and `5993`. The shared frozen
encoder/tokenizer/validation data/preparation pool are intended harmless shared
stage state; adapter/objective/optimizer/sampler/histories/checkpoints are
intended run-local.

Selection:

Pure selection functions are strict and frozen. Artifact handoff between stages
is not strict enough and is a blocker.

Checkpoint/resume:

Disk publication is atomic and adapter/optimizer/sampler restore design is
strong. Production resume is blocked by `ValidationPoint` schema mismatch and
continuation cap reconstruction failure.

Provenance:

Stage-6 derives actual repository HEAD. Stage-1 training records an optional
caller-supplied HEAD and is blocked on provenance integrity.

Authoritative CUDA dependency:

CUDA/FP32/determinism source contract is strong but must be reaccepted after
repairs to current launch/resume code. Local `.venv` cannot provide that
evidence.

Human-controlled:

Git operations, official TEST access, real Stage-1 launch, corpus placement,
output paths, and Colab/runtime environment remain operator-controlled.

## 32. Scientific optimizer.step status

No real scientific optimizer step was executed during this review.

Production occurrence:

- `unmark/stage1/trainer.py:677` is the single Stage-1 production
  `optimizer.step()`.

TEST-ONLY occurrences exist in pytest fixtures and historical/probe code. They
are not Stage-1 scientific training.

## 33. Official UIT-VSFC TEST status

Official UIT-VSFC TEST status: SEALED / UNUSED.

This review did not open it, inspect it, screen it, evaluate it, pass it to any
command, or derive information from it.

## 34. Final verdict

BLOCKER count: 5

MAJOR count: 0

Because one or more genuine BLOCKER findings exist, the training gate verdict
is:

FINAL FULL-REPOSITORY PRE-TRAIN REVIEW BLOCKED -- MATERIAL ISSUE(S) REQUIRE HUMAN DECISION BEFORE TRAINING

