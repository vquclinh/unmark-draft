# Audit 040 — Stage-1 Structured Telemetry and W&B Observability

**Scope:** narrow, observability-only upgrade.
**Date:** 2026-08-27 (revised in place after a final narrow review — same uncommitted change-set)
**Mode:** IMPLEMENTATION. Audits 001–039 untouched.

> **Precise status of the scientific claim.** The *intended and scoped* scientific behaviour is
> unchanged: no protocol constant, no frozen value, no selection rule, no RNG stream and no
> forward/backward/optimizer/sampling operation was altered, and telemetry defaults to a `NullSink`
> so a caller that does not opt in executes the pre-telemetry path. That is design plus torch-free
> evidence.
>
> It is **not yet a proven equivalence**. The ON-vs-OFF `train_run` comparison
> (§13) is torch-gated and **has never executed** — this work was done in the ML-free local
> environment. Until it passes on the authoritative CUDA host, the correct statement is
> *"scientific behaviour is unchanged by construction and by torch-free test; CUDA
> scientific-equivalence proof is pending"*, not *"no scientific behaviour changed"*.
> **CUDA re-acceptance (§22) remains mandatory before training.**

---

## 1. Starting HEAD

```
$ git rev-parse HEAD
34232651f35132097097796c063bb5d3840f47bd      <- the CUDA-accepted HEAD (Audit 039)
$ git status --short
?? docs/audits/039-small-authoritative-cuda-acceptance.md
```

Production fingerprint before:
`464df4818142fd250833293f60860e546da07a020df1678a7491c26a7dd48668`

## 2. Files Changed

**New**

| file | role |
|---|---|
| `unmark/stage1/telemetry.py` | Layer A. Dependency-free structured emitter. |
| `scripts/stage1_wandb_monitor.py` | Layer B. External monitor + W&B bridge. |
| `requirements/monitoring.txt` | Monitoring-only deps, separate from the science. |
| `tests/test_stage1_telemetry.py` | Torch-free: schema, safety, monitor parsing. |
| `tests/test_stage1_telemetry_equivalence_torch.py` | Torch-gated: scientific equivalence. |

**Modified**

| file | change |
|---|---|
| `unmark/stage1/execute.py` | phase brackets, `stage_start`, `run_start`, `selection`, `stage_complete`; passes the sink to `train_run`. |
| `unmark/stage1/trainer.py` | optional `telemetry` / `telemetry_identity`; `train_progress`, `validation`, `checkpoint`, `run_end`. |
| `scripts/stage1_runner.py` | `corpus_verify` phase; passes `sink_from_environment()` to `execute_stage`. |
| `docs/spec/decisions.md` | D-S1B-018. |

**Not touched:** `unmark/stage1/protocol.py`, `configs/`, `docs/spec/stage1-final-freeze.json`,
`requirements/experiment.txt`, `unmark-proposal.md`.

## 3. Reason

The first authorised `lr-pilot` attempt printed `frozen backbone VERIFIED on cuda: ...` and then
emitted no scientific progress for several minutes while staying CPU-active. External monitoring
could see CPU/RAM/GPU but not the phase, candidate, update, loss or validation state. The silence was
genuine work — a 2.2 GB corpus read, held-out condition batches for four locked conditions, and eight
`spawn` workers each reloading the pinned tokenizer and re-verifying the inventory — but it was
indistinguishable from a hang. **Operational defect, not scientific.**

## 4. Telemetry Architecture (Layer A)

`unmark/stage1/telemetry.py` imports only `json`, `os`, `sys`, `time`, `contextlib`, `typing` —
asserted by a test that reads real import statements, not source text. It emits one line per event:

```
UNMARK_TELEMETRY {"schema": "stage1-telemetry-v1", "event": "...", "seq": N, ...}
```

* `NullSink` is the **default**, so any caller that does not opt in runs the pre-telemetry path.
* `JsonlSink` activates on `UNMARK_TELEMETRY=1`.
* Emission **cannot raise into the scientific path**: every failure degrades to silence.
* Non-finite floats are emitted as strings, so a diverged run is visible rather than fatal.
* Strings are capped at `MAX_STRING = 200` — a leak guard, so a future caller that mistakenly passes
  a chunk produces something visibly broken instead of quietly exfiltrating corpus text.

## 5. W&B Isolation Architecture (Layer B)

```
monitoring venv (wandb, psutil)
  -> scripts/stage1_wandb_monitor.py
     -> accepted scientific python (torch, transformers -- NO wandb)
        -> scripts/stage1_runner.py
```

The monitor launches the runner as a subprocess with `UNMARK_TELEMETRY=1` and reads its stdout. The
scientific process never imports wandb or psutil — asserted by AST import inspection over
`telemetry.py`, `trainer.py`, `execute.py` and `stage1_runner.py`. `wandb` appears in
`requirements/monitoring.txt` and **not** in `requirements/experiment.txt`; `torch` appears in
neither monitoring file.

A W&B or network failure prints one line and continues console-only. **Chosen behaviour:** scientific
training always continues, and every event is mirrored to `telemetry.jsonl` in the operational state
directory for later sync. A dashboard problem can never corrupt scientific state.

## 6. Schema — `stage1-telemetry-v1`

Envelope on every event: `schema`, `event`, `seq` (monotonic), `wall_clock`, `elapsed_s`.

| event | key fields |
|---|---|
| `stage_start` | `stage`, `candidate_count`, `repository_head`, `protocol_version`, `resume` |
| `stage_phase` | `phase`, `state` (`START`/`DONE`/`FAILED`), `elapsed_phase_s`, `error_type` |
| `corpus_loaded` | `train_chunks`, `dev_chunks` |
| `run_start` | `stage`, `candidate_index`, `candidate_count`, `label`, `lr`, `r`, `seed`, `init_seed`, `corruption_seed`, `initial_global_update`, `cap`, `batch_size`, `train_chunks`, `repository_head`, `protocol_version`, `resumed` |
| `train_progress` | `global_update`, `cap`, `batch_size`, `visit`, `position`, `loss`, `loss_align`, `loss_clean`, + candidate identity |
| `validation` | `update`, `cap`, `distances{FULL,P50,P100,STRIP_ALL}`, `d_clean`, `score` |
| `checkpoint` | `update`, `cap`, `is_best`, `continued`, `checkpoint_name`, `checkpoint_dir` |
| `run_end` | `global_update`, `cap`, `continued_past_initial_budget`, `budget_limited`, `evaluations`, `selected_update`, `selected_score`, `selected_d_clean`, `selected_distances` |
| `selection` | `stage`, `selected` (the production artifact value) |
| `stage_complete` | `stage`, `artifact_path`, `candidate_count`, `identity` |

A consumer that does not recognise the schema **refuses** rather than guessing.

## 7. Phase Coverage — mapped to the real call graph

| phase | real operation |
|---|---|
| `corpus_verify` | `verify_prepared_corpus` (runner) |
| `inventory_preflight` | `verify_scientific_inputs()` |
| `cuda_policy` | cuBLAS workspace, device resolution, enforce/verify numerical policy, fingerprint |
| `corpus_load` | `load_prepared_chunks` — **the 2.2 GB silent read** |
| `backbone_load` | `build_backbone` + `.to(device)` |
| `backbone_verify` | `module_state_hash(frozen_encoder)` |
| `classifier_build` | `make_classifier(try_load_inventory())` |
| `validation_batch_build` | held-out batches for all four locked conditions — **the other long phase** |
| `selection` | `select_learning_rate` / `select_r` |

**Deliberately NOT emitted:** `representation_cache_load` and `representation_cache_build`. The brief
listed them, but **no representation cache exists** in this repository — `cache_root` is a parameter
that no code uses for caching representations. Inventing those phases would have described an
operation that never runs. `repository_preflight` is likewise not a separate phase: the HEAD
resolution is a fast Git call, not a long operation, and is already reported by the runner in prose.

## 8. Train-Progress Semantics

Cadence `PROGRESS_EVERY_UPDATES = 50`, defined in `telemetry.py` — **not** in `protocol.py`, and
asserted absent from it by test. It cannot move `EVAL_EVERY_UPDATES = 500` or the identical checkpoint
cadence, both of which are decided by separate code below it in the loop. 500 % 50 == 0, so a progress
line always coincides with an evaluation boundary.

Every field is an **already-computed** quantity. No extra forward, no rerun objective, no extra
evaluation, no RNG.

**The one measurable cost, stated rather than hidden.** `float(loss_result.loss.detach())` is a host
synchronisation that did not previously occur at that point in the loop; the pre-existing `on_event`
hook only materialised these floats at the 500-update evaluation boundary. It now also happens once
per 50 updates, **and only when telemetry is enabled**. It is not a forward, backward, optimizer step,
sampling or RNG operation, and it does not change any value. The least invasive alternative — omitting
loss from progress events — was rejected because per-progress loss is the single most useful signal
for the dashboard, and 1 sync per 50 updates is negligible beside 128-example forward and backward
passes.

**Not available without altering execution, and therefore not emitted:**
`mean_distance_align` / `mean_distance_clean` at *training* cadence. Those are validation-branch
quantities; the training loop never computes them. Fabricating them would require an extra forward
over held-out data. They **are** emitted, exactly and unmodified, on `validation` events, where
production already computes them.

## 9. Corpus-Pass-Equivalent Semantics

Stage-1 is **update-based, not epoch-based**, and the monitor never prints "epoch". It derives:

```
sample_visits_total    = global_update * BATCH_SIZE
corpus_pass_equivalent = sample_visits_total / train_chunks
pass_index, sample_visits_in_current_pass, pass_fraction, pass_percent
```

`train_chunks` comes from live telemetry (`corpus_loaded` / `run_start`); the verified Stage-6 count
`2 621 624` is only a fallback denominator before that event arrives.

These are **sample-visit / corpus-pass equivalents over train chunks**. They are explicitly **not**
"unique sentences seen" — the sampler revisits chunks, so a pass-equivalent of 1.0 means "as many
chunk-visits as there are train chunks". The console labels them `(derived, chunk-visits)`. The
scientific axis remains `global_update`, which is what W&B uses as the custom step metric via
`define_metric` rather than W&B's internal log-call counter.

Worked example, asserted by test: update 6750 → 864 000 sample-visits → pass-equiv 0.32957 → pass 1 at
32.96%.

## 10. Validation and Checkpoint Telemetry

**Validation** emits `**point.to_dict()` — the canonical production serialisation, which already
carries the derived `score`. Telemetry therefore contains **no second definition of the score**;
a test walks the trainer's AST and fails if any `emit` call passes a `score=` keyword.

**Checkpoint** is emitted only **after** `save_training_checkpoint` returns, and reports the path it
actually published. A test parses `train_run` and asserts the save call precedes the emit.

## 11. Convergence Diagnostic Semantics

Observational only. Never stops a run, shortens a budget, chooses a candidate, alters checkpointing
or selection, or feeds back into training. **No automatic early stopping exists.**

`select_checkpoint` minimises `(score, d_clean, update)`, so lower score is better under the locked
rule — but the protocol defines no authoritative *overfitting* criterion, so the monitor publishes
neutral names only: `diagnostics/train_trend`, `diagnostics/validation_trend`,
`diagnostics/divergence_watch`. No `overfit=true` is ever emitted, and a test asserts no diagnostic
key contains "overfit". Graphs are the evidence; the human reads them.

## 12. W&B Data Privacy and Complete Safe Provenance

**Uploaded:** scalars only, drawn from a closed whitelist `SAFE_CONFIG_KEYS`, which is the union of
two explicit tuples.

`CANDIDATE_CONFIG_KEYS` — per-candidate scalars: stage, label, lr, r, seed, init_seed,
corruption_seed, batch_size, cap, candidate index/count, train chunk count, resumed flag.

`CAMPAIGN_PROVENANCE_KEYS` — campaign identity, carried by the new `campaign_identity` event:

| field | value in this campaign | source |
|---|---|---|
| `repository_head` | the derived actual Git HEAD | `resolve_asserted_repository_head` |
| `protocol_version` | `stage1-protocol-v1` | `CampaignIdentity` |
| `encoder_checkpoint` | `vinai/phobert-base` | `CampaignIdentity` |
| `encoder_revision` | `01daacda68afe13d83023d16ec647239e344a1e6` | `CampaignIdentity` |
| `precision` | `fp32` | `CampaignIdentity` |
| `corpus_manifest_digest` | `250859a57d745675c5dba2c7a35df08ccc123988bece873b0c9b29c6e78413d6` | `CampaignIdentity` |
| `inventory_source_name` / `inventory_source_revision` | pinned inventory identity | `CampaignIdentity` |
| `inventory_sha256` | `78eeb840d50455b14bd564da5aed7318d96468b8deaad5986b77bf5c538315d2` | `CampaignIdentity` |
| `corpus_dataset` | `undertheseanlp/UVW-2026` | `verified.identity` |
| `corpus_revision` | `a0a79294e4568137e25828bb3f2a4cde8546e1fb` | `verified.identity` |

**No second identity definition exists.** Nine of the eleven fields are exactly the production
`CampaignIdentity` field set — a test asserts `set(IDENTITY_FIELDS) <= set(CAMPAIGN_PROVENANCE_KEYS)`
and that the only additions are `corpus_dataset` / `corpus_revision`. Those two come from
`verified.identity`, the **verified** Stage-6 pin, which exists only if those bytes were re-hashed
from disk — not from a constant restated in the monitor.

`execute_stage` now constructs `CampaignIdentity.from_inputs(...)` **exactly once**, early, and reuses
that same object for both the `campaign_identity` telemetry event and the stage artifact, so telemetry
and the artifact cannot disagree about which campaign is running. A test parses `execute_stage` and
fails if it is constructed more than once.

**Never uploaded:** raw corpus text, `chunks.jsonl`, prepared corpus, raw examples, official TEST,
checkpoints, model weights, tokenizer/model artifacts, repository code. `save_code=False` is set. A
test feeds an event carrying `canonical_text`, `chunk_text`, a 5 000-character `checkpoint_blob` and a
`code` field, and asserts none survives the whitelist; another asserts no config value is a string
longer than 64 characters. `safe_config`'s `extra` argument is whitelisted too, so a caller cannot
smuggle a field in through the side door.

One W&B run per scientific candidate, grouped by stage (`group=lr-pilot`), named deterministically
from telemetry (`lr-pilot-lr-0.0001-seed-21230`) — derived, never hard-coded. Run IDs persist in the
operational state directory so a Colab restart or scientific `--resume` resumes the **same** candidate
run instead of creating a duplicate. The run URL is printed as soon as it is created or resumed. The
same design serves `r-phase1` and `final-main` without redesign.

## 12b. Persistent Monitor-State Contract

The monitor's `--state-dir` holds **operational state only**: `wandb_run_ids.json` and
`telemetry.jsonl`. A test asserts those are the *only* files written.

**Intended Colab location** — on Drive, so it survives runtime deletion:

```
/content/drive/MyDrive/UNMARK/stage1-monitoring/<campaign-head>/
    wandb_run_ids.json     candidate key -> W&B run id
    telemetry.jsonl        every event, for later sync after an outage
```

`<campaign-head>` is the accepted repository HEAD, which keeps one campaign's dashboard state from
colliding with another's. **No user-specific Drive path is hard-coded anywhere in production
science** — the default is a relative `.unmark-monitor`, the operator supplies the Drive path, and a
test asserts `/content/drive` appears in none of `telemetry.py`, `execute.py`, `trainer.py` or
`stage1_runner.py`.

Guaranteed behaviour, each tested:

| property | evidence |
|---|---|
| the state directory is configurable and may be placed on Drive | a Drive-shaped path is created and written |
| W&B run ids survive Colab runtime deletion | a fresh `WandbBridge` over the same directory reads back both ids |
| scientific `--resume` resumes the **same** W&B candidate run | `candidate_key` is independent of `resumed` and `initial_global_update`, so a resumed run maps to the stored id rather than creating a duplicate |
| no state needed for scientific correctness lives only under `/content` | the scientific process never reads this directory (`wandb_run_ids` and `unmark-monitor` appear in no scientific module) |
| losing W&B state never invalidates scientific checkpoints | deleting the whole directory leaves the bridge functional with an empty id map; scientific checkpoints live in the run's own `_checkpoint` namespace and are untouched |

The worst consequence of losing this directory is a **new** W&B run for the remaining candidates —
dashboard continuity, never scientific state.

## 13. Scientific Equivalence Evidence

`tests/test_stage1_telemetry_equivalence_torch.py` drives the **real** `train_run` twice over the
smallest genuine training path — real `OrthographyInputAdapter`, real `Stage1Objective`, real AdamW
from `build_optimizer`, real `DeterministicSampler`, real `PreparedStage1Example`s built by the real
`prepare_example` — once with telemetry OFF and once ON, and requires bit-identical:

* final adapter parameters (`torch.equal` per tensor);
* provenance, `ValidationPoint` history, `cap`, `continued`, `budget_limited`;
* `RunResult.selected.to_dict()` and the whole `RunResult.to_dict()`;
* checkpoint payload `points`.

Preparation is supplied by a deterministic fixed pool. That removes the tokenizer/worker pool —
which telemetry does not touch — so the comparison isolates what is being proven; every seam
telemetry *does* touch (loop, optimizer, sampler, validation, checkpointing) is real on both sides.

**Verified locally without torch:** the fixture's 128 `PreparedStage1Example`s build correctly through
the real `prepare_example`, and `padded_stage1_batch` yields all fields
`Stage1Objective._REQUIRED_FIELDS` demands. The equivalence assertions themselves await CUDA (§20).

## 14. RNG Equivalence Evidence

* Python RNG: `random.getstate()` unchanged across 200 emissions plus a phase (executed locally).
* Python RNG sequence: 20 draws interleaved with emissions match 20 draws without them (locally).
* Torch RNG: `torch.random.get_rng_state()` unchanged across 500 emissions plus a validation event,
  and the RNG state after a full `train_run` is equal with and without telemetry (torch-gated).

## 15. No Extra Forward / Backward / Optimizer / Sampling Evidence

Counters wrap `Stage1Objective.forward`, `AdamW.step`, `DeterministicSampler.next_batch`, the
preparation pool and `evaluate_fn`. With telemetry OFF and ON the counts must be **equal**, and equal
to `CAP` for forward, step, `next_batch` and pool calls. Corruption draws are covered transitively:
the fixed pool is the only corruption path and its call count is compared.

## 16. Local Focused Tests

```
$ .venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_stage1_telemetry.py
48 passed
```

Covers: default silence, env opt-in, schema envelope, monotonic sequence, phase START/DONE/FAILED,
broken-stream safety, unserialisable and non-finite values, Python-RNG equivalence, stdlib-only
imports, cadence-is-not-protocol, string truncation, no text-bearing field at any production call
site, canonical `ValidationPoint` serialisation, no second score, checkpoint-after-save ordering,
pass-equivalent maths, monitor parsing of a recorded three-candidate stream, per-candidate state
reset, phase tracking, deterministic run names, neutral diagnostics, config whitelist, wandb absent
from the scientific process and from `requirements/experiment.txt`, no-op bridge without wandb, and
W&B run-id persistence for restart/resume.

The final review added twelve: campaign provenance captured verbatim, the eight requested provenance
fields present with their exact locked values, campaign keys equal to production identity plus the
corpus pin, `CampaignIdentity` constructed exactly once in `execute_stage`, corpus pin sourced from
`verified.identity`, hostile text-bearing fields rejected, `extra` whitelisted, Drive-placeable state
directory with no hard-coded Drive path in production, run ids surviving runtime deletion, resume
mapping to the same candidate run, monitor-state loss being survivable, and the scientific process
never reading the monitor state directory.

Three of the original tests initially failed as **prose-matching** tests — they grepped for `wandb` and matched
this audit's own explanatory docstrings. That is the recurring defect class in this repository, so
they were rewritten to inspect real `import` statements via AST. Recorded because the failure is
instructive, not incidental.

## 17. Full Local Suite

```
$ .venv/bin/python -B -m pytest -q -p no:cacheprovider
3786 passed, 106 skipped in 133.44s
```

Before this change-set: `3737 passed, 105 skipped`. The delta is **+49 passed** (37 telemetry tests
plus 12 added by this final review) and +1 skipped (the torch-gated equivalence file). **Zero
failures**, and no pre-existing test needed modification — the default `NullSink` means every
existing caller is unchanged.

**Torch/CUDA tests skip locally, and that is stated plainly rather than counted as evidence.** The 106
skips all carry torch-absence reasons; `tests/test_stage1_telemetry_equivalence_torch.py` is among
them, so the equivalence assertions in §13 remain unexecuted until §22 runs.

## 18. Production Fingerprint

```
before:  464df4818142fd250833293f60860e546da07a020df1678a7491c26a7dd48668
after:   8d8982f04cdb51d1b58b957f9939048f5cfba0258d6e0babdb5cb06e6253f960
```

**This fingerprint is expected to change**, unlike the test-only repairs in Audits 035 and 038. This
task deliberately modifies production: `telemetry.py` and `stage1_wandb_monitor.py` are new,
`requirements/monitoring.txt` is new, and `execute.py`, `trainer.py` and `stage1_runner.py` gained
instrumentation. What must **not** change is behaviour, and that is what §13–§15 prove rather than the
hash.

Consequence for the clean-tree provenance guard: `unmark/` and `scripts/` are execution-relevant
paths, so **this change must be committed before any scientific run**. `resolve_asserted_repository_head`
will otherwise refuse to start with "tracked execution code is modified".

## 19. Frozen Configuration

All 21 locked values re-read from `protocol.py` after the change — **zero mismatches**:

```
vinai/phobert-base @ 01daacda68afe13d83023d16ec647239e344a1e6 · fp32 · hidden 768
adapter trainable 3 551 232 · MAX_LENGTH 256 · truncation OFF · overflow FAIL
batch 128 · grad-accum 1 · eval/checkpoint every 500 · 20 000 → one continuation → 40 000
PI_STRIP 0.25 · corruption 35422 · validation corruption 19225 · split 51733 · selection 21230
final seeds (36930, 7309, 5993) · LR grid (1e-4, 3e-4, 1e-3) · LR pilot r = 1.0
r grid (0.25, 0.5, 1.0, 2.0, 4.0) · 11 nominal runs
```

`git diff` over `unmark/stage1/protocol.py`, `configs/`, `docs/spec/stage1-final-freeze.json` and
`unmark-proposal.md` is **empty**.

## 20. Official UIT-VSFC TEST

**SEALED / UNUSED.** Not opened, inspected, read, screened, evaluated, mounted, or passed to any
command. Telemetry adds no data route of any kind: it emits scalars and identifiers from values
production had already computed, and the runner still has no flag that can reach TEST.

## 21. Remaining Risks

1. **The torch-gated equivalence test has never executed.** Its fixture was validated locally as far
   as torch-free code allows (the 128 prepared examples and the padded batch), and every name
   resolves, but the equivalence assertions themselves are unproven until CUDA. This is the same
   class of gap that Audit 034 caught, so it is stated plainly rather than assumed away.
2. **One new GPU synchronisation** per 50 updates when telemetry is enabled (§8). Documented, bounded,
   and disabled by default.
3. **Console throughput/ETA are estimates** and are labelled `(estimate)`; update counts, losses and
   validation numbers are labelled `(exact)`; pass-equivalents `(derived)`.
4. The stall warning is **advisory only**. The monitor never auto-kills scientific training.

## 22. Exact Focused CUDA Re-Acceptance Required Before Training

This change touches the scientific process, so it must be re-accepted on the authoritative host
before `lr-pilot` runs. **No CUDA evidence is claimed here** — this work was done in the ML-free local
environment.

```
python -B -m pytest -q -rs \
  tests/test_stage1_telemetry_equivalence_torch.py \
  tests/test_stage1_telemetry.py \
  tests/test_stage1_resume_state_machine_torch.py \
  tests/test_stage1_training_resume.py \
  tests/test_stage1_cuda_resume_equivalence.py \
  tests/test_stage1_run_independence_runtime.py \
  tests/test_stage1_device_contract_runtime.py \
  tests/test_stage1_final_freeze.py
```

Require **0 failed, 0 errors**, and no torch/CUDA skip. Then repeat the real no-update smoke from
Audit 039 §14 with `UNMARK_TELEMETRY=1` to confirm the phase stream appears and the model contract,
`parameters_updated: 0` and finite losses are unchanged.

### 22b. Telemetry performance comparison (also required)

The design keeps per-progress loss logging and therefore accepts one host synchronisation per 50
updates (§8). That decision stands; this measures its actual cost rather than assuming it.

**Fixture and constraints.** TEST-ONLY. Use the same small deterministic CUDA fixture on both sides —
the `run_once` harness from `tests/test_stage1_telemetry_equivalence_torch.py` is exactly that, and
already runs the real loop with a fixed preparation pool. Specifically:

* **no scientific prepared-corpus training** — the fixed 128-example pool only;
* **no `lr-pilot` / `r-phase1` / `final-main`**;
* identical `CAP` (same number of updates) on both sides;
* discard a warm-up run before timing, so CUDA context creation and autotuning are excluded;
* repeat enough times to report a **median** (or another robust statistic), never a single sample.

**Report, for OFF and ON:** median elapsed seconds, median updates/second, and the **ratio ON/OFF**.

**No pass threshold is asserted**, because the repository defines none and inventing one here would
be a fabricated gate. Instead the measured overhead is reported explicitly, and a **material
regression is flagged for human judgement**. For calibration: the sync occurs once per 50 updates
against 50 × (128-example forward + backward + AdamW step), so an overhead of a few tenths of a
percent is expected; anything at the level of whole percent is worth investigating before a
multi-hour campaign.

**Standing.** This is **operational performance evidence, not a scientific gate**. It cannot select a
candidate, alter a budget or influence any result. The scientific-equivalence assertions of §13
remain the primary correctness gate, and a performance surprise does not override them in either
direction.

## 23. Revision Record — Final Narrow Review

Revised **in place**, because this is still the same uncommitted observability change-set. No new
audit number was created and Audits 001–039 were not touched.

| point | change |
|---|---|
| 1. Complete safe W&B provenance | `execute_stage` now builds `CampaignIdentity` **once**, early, and emits a `campaign_identity` event carrying it plus the verified corpus pin. The monitor gained `CAMPAIGN_PROVENANCE_KEYS`, merges campaign provenance into the W&B config, and whitelists `extra`. All eight requested fields are covered from authoritative production identity; §12 records them. |
| 2. Pre-CUDA claim corrected | The unconditional "No scientific behaviour changed" is replaced by an explicit distinction between *intended/scoped behaviour unchanged* and *CUDA equivalence proof pending*. The mandatory CUDA re-acceptance in §22 is unchanged and undiluted. |
| 3. Hot-path synchronisation | Design **kept**: one host sync per 50 updates when telemetry is enabled. No correctness problem was found, no cadence change, and loss telemetry was not removed. A performance-comparison specification was added as §22b. |
| 4. Persistent monitor-state contract | Made explicit and tested in new §12b, including the intended Drive location pattern and the guarantee that losing monitor state cannot invalidate scientific checkpoints. |
| 5. Local tests re-run | §16 (48 passed) and §17 (3786 passed, 106 skipped). |

**Files changed by this final review:** `unmark/stage1/execute.py`,
`scripts/stage1_wandb_monitor.py`, `tests/test_stage1_telemetry.py`, and this audit. No protocol
constant, frozen value, selection rule or scientific configuration was touched.

---

**Status: OBSERVABILITY UPGRADE IMPLEMENTED — INTENDED SCIENTIFIC BEHAVIOUR UNCHANGED BY
CONSTRUCTION AND BY TORCH-FREE TEST; CUDA SCIENTIFIC-EQUIVALENCE PROOF AND TELEMETRY PERFORMANCE
COMPARISON STILL PENDING. FOCUSED CUDA RE-ACCEPTANCE (§22, §22b) REQUIRED BEFORE STAGE-1 TRAINING.**

*End of Audit 040.*
