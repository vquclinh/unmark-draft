# Audit 046 - Stage-1 r-phase1 Operational Acceleration

**Scope:** add speed-oriented execution controls for `r-phase1` without changing
the Stage-1 scientific protocol.
**Date:** 2026-09-04

## 1. Motivation

The completed LR pilot showed that one 20,000-update candidate can take roughly
20 hours on the current Colab runtime. Running the five `r-phase1` candidates
sequentially would therefore spend about 100 hours before final-main.

The runtime also showed spare device memory. That does not license a batch-size
change: `BATCH_SIZE = 128`, fp32 deterministic execution, the 20,000-update
budget, the validation cadence, and the locked r grid are scientific protocol
values and remain unchanged.

## 2. Worker Count Override

Training-time preparation now resolves the worker count through
`UNMARK_STAGE1_PREPARATION_WORKERS`, defaulting to the existing
`PREPARATION_WORKERS = 8`.

This is operational-only:

- the main process still owns sampler advancement;
- batch membership and order are unchanged;
- checkpoint cadence and payloads are unchanged;
- prepared examples are already required to be byte-identical across worker
  counts;
- the chosen worker count is recorded in artifact `preparation` provenance and
  remains outside resume-blocking scientific identity.

Invalid values fail closed before the worker pool is built.

## 3. Fused r-phase1 Execution

`UNMARK_STAGE1_R_PHASE1_EXECUTION=fused` enables an opt-in fused/interleaved
execution path for `r-phase1`.

The path is accepted only for the locked `r-phase1` schedule:

- exactly five candidates;
- stage `r_phase1`;
- one frozen learning rate from the LR artifact;
- one shared selection seed;
- the locked r grid `{0.25, 0.5, 1, 2, 4}`.

Each candidate still has its own adapter, optimizer, checkpoint directory,
validation history and run JSON. The only shared work is the already-selected
training batch preparation. On each update, the first active candidate advances
the sampler, every other active candidate must produce the identical
`(chunk_id, visit)` batch, the batch is prepared once, and then each candidate
takes its own optimizer step.

If a resume finds active fused candidates at different update/cap/cursor states,
fused execution refuses rather than guessing. The same per-candidate checkpoints
can still be resumed by the original sequential path.

## 4. Colab Handoff

`docs/colab/run_r_phase1_accelerated_cell.py` is a copyable Colab cell that:

- checks no Stage-1 process is already running;
- checks out the latest pushed `origin/main`;
- reissues `lr_pilot.json` under that current HEAD without rerunning LR;
- sets `UNMARK_STAGE1_PREPARATION_WORKERS` to a bounded runtime-derived value;
- sets `UNMARK_STAGE1_R_PHASE1_EXECUTION=fused`;
- launches monitored `r-phase1` using the existing LR handoff artifact.

The cell does not add any scientific override flag.

## 5. Tests

Targeted validation:

```text
pytest -q tests/test_stage1_parallel_preparation.py \
  tests/test_stage1_fused_r_phase1.py \
  tests/test_stage1_runner_contract.py \
  tests/test_stage1_runner_cli_contract.py \
  tests/test_stage1_run_independence.py \
  tests/test_stage1_name_resolution.py \
  tests/test_stage1_artifact_identity.py \
  tests/test_stage1_schedule.py

198 passed, 2 skipped
```

The added tests cover:

- default worker count remains 8;
- environment worker override accepts positive integers and rejects bad values;
- fused mode is environment-only, not a CLI scientific flag;
- fused mode refuses non-r-phase1, split-LR, split-seed or incomplete schedules;
- the fused mini-run prepares one batch per update rather than once per
  candidate per update.

No UIT-VSFC official TEST data was opened or used.
