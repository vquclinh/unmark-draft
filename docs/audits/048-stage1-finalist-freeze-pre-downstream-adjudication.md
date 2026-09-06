# Audit 048 - Stage-1 Finalist Freeze and Pre-Downstream Adjudication Amendment

**Scope:** freeze exactly two already-trained Stage-1 adapters as the complete
candidate set for the final downstream adapter, and record that the choice
between them is still OPEN.
**Date:** 2026-09-06
**Type:** post-hoc amendment, made **before any downstream experiment was run**.

This audit does not select the final adapter, does not define the Stage-2
adjudication protocol, and does not reopen any Stage-1 hyperparameter.

---

## 1. Executive verdict

**PASS WITH ISSUES.**

The finalist freeze is implemented, pinned in code, machine-readable and tested.
One piece of external evidence is missing and is represented as a blocker rather
than filled in:

| | |
|---|---|
| Finalist set frozen at exactly two | **PASS** |
| Provisional `canonical-v1` preserved, superseded as FINAL | **PASS** |
| Stage-1 training / LR / `r` / candidate generation recorded CLOSED | **PASS** |
| Adjudication recorded OPEN, final adapter UNSELECTED | **PASS** |
| Official UIT-VSFC TEST sealed; TEST-based selection refused | **PASS** |
| Read-only checkpoint verifier | **PASS** — real-torch run on Colab refused every malformed payload, §10.1 |
| Real-torch test expectations | **REPAIRED** — 2 stale regexes, §10.1 |
| Authoritative A/B checkpoint verification | **NOT PERFORMED** — §10.1 |
| Structural prevention of finalist-set drift | **PASS** |
| Training-core source equivalence of the two HEADs | **PASS** — established, §7.6 |
| Checkpoint provenance verification vs the Stage-1 contract | **PASS** — delegates to `verify_checkpoint`, §9.1 |
| **Finalist B exact checkpoint SHA256** | **BLOCKED** — §6 |
| Full executable / runtime equivalence of the two runs | **NOT ESTABLISHED** — §7.6 |

Because finalist B's digest is unbound, `evidence.freeze_complete` is `false` and
`READY_FOR_STAGE2_PROTOCOL_REVIEW=NO`. The freeze is **not** claimed complete.

---

## 2. Starting repository state

```
branch : main
HEAD   : 7773c77b1df92a6e685dac13c49765ce974f84d8
status : clean (git status --porcelain produced no output)
```

Recent history:

```
7773c77 Use population std for resource-bounded r summary
b6d22e2 Fix r-phase1 telemetry LR field validation
552f2e3 Harden Stage-1 r-phase1 resource-bounded amendment
49f8c68 r-phase1 resource bounded
16af326 delete wandb
```

Inherited Audit 047 state: the `r-phase1` handoff selects `r = 1.0` at frozen
`LR = 0.0001` under `selection_override.kind =
author_r_override_after_resource_bounded_validation_review`, over the observed
window `[4000, 4500, 5000, 5500, 6000, 6500]`, with `global_optimum_claimed =
false` and both `official_test_used` and `downstream_score_used` false
(D-S1B-022). Nothing in Audit 047 is rewritten here.

`docs/spec/stage1-final-freeze.json` — the **pre-training configuration** freeze
generated for HEAD `649ad741b8e7` — is left untouched. It contains no checkpoint
digests; the artifact added by this audit is a separate, differently named file.

---

## 3. Historical scientific state (not rewritten)

**Original v1.5 run plan** — `unmark-proposal.md` §5.1, "exactly 11 runs, and
nothing follows them":

| Stage | Runs | LR | `r` | Seed |
|---|---|---|---|---|
| LR pilot | 3 | swept `{1e-4, 3e-4, 1e-3}` | fixed `1` | `21230` |
| `r` Phase 1 | 5 | frozen pilot winner | swept `{0.25, 0.5, 1, 2, 4}` | `21230` |
| Final main Stage-1 | 3 | selected | selected | `36930, 7309, 5993` |

**Two amendments already on record, both preserved as deviations:**

* **D-S1B-020 / Audit 045.** `LR = 1e-4` is an **author override after
  validation-curve review**. It was **not** the automatic locked-rule winner;
  that rule preferred `3e-4` on a single lowest-score point at update 500.
* **D-S1B-022 / Audit 047.** `r = 1.0` was selected from a **resource-bounded**
  observed `r-phase1` window ending at update 6500 against a planned 20,000 cap.
  It is **not** claimed globally optimal.

**Actual final-main execution status — the deviation this audit must keep
explicit:**

| Seed | Planned | Actual |
|---|---|---|
| 36930 | 20,000 updates | **completed 20,000 updates** |
| 7309 | 20,000 updates | **invocation/start event only; stopped before any training update; no checkpoint produced** |
| 5993 | 20,000 updates | **never started** |

One of three planned final-main seeds ran. This audit does not describe the
final-main stage as complete.

---

## 4. Stage-1 closeout evidence inherited

A read-only closeout reconstructed **full** validation trajectories for two runs.
Each carries 41 points at updates `0, 500, ..., 20000`, and at every point the
evidence contains `validation/score`, `validation/d_clean`, `FULL`, `P50`,
`P100`, `STRIP_ALL`.

Update 0 was **absent from exported telemetry / W&B history** and was recovered
and verified from the checkpoint `points` history for **both** runs. Training
losses and execution diagnostics exist in telemetry but were **not** added to the
checkpoint selection objective.

**Historical locked-rule best checkpoints, independently verified:**

| Run | Best update | `validation/score` |
|---|---|---|
| seed 21230 (LR pilot, `lr=1e-4`, `r=1`) | 14500 | 0.09000698585438581 |
| seed 36930 (final main, `lr=1e-4`, `r=1`) | 3500 | 0.0845640671895974 |

**Post-hoc Stage-1-only stability rule, written BEFORE candidate ranking was
applied:**

* the candidate must have physically existing checkpoint weights;
* local neighbourhood = exactly the five validation points nearest the
  checkpoint update, ordered by `(abs(validation_update - checkpoint_update),
  validation_update)`;
* `robust_score = median(local validation/score) + MAD(local validation/score)`;
* `robust_d_clean = median(local d_clean) + MAD(local d_clean)`;
* deterministic ordering: `robust_score`, then `robust_d_clean`, then the
  checkpoint's own `validation/score`, then earlier update, then SHA256;
* **no downstream information admitted.**

**Ranking produced:**

| Rank | Candidate | point score | `robust_score` | local window |
|---|---|---|---|---|
| 1 | seed 36930 @ 3500 | 0.0845640671895974 | 0.10167897852382013 | 2500, 3000, 3500, 4000, 4500 |
| 2 | seed 21230 @ 14500 | 0.09000698585438581 | 0.106576028028 | 13500, 14000, 14500, 15000, 15500 |

**Whole-run descriptive contrast, which points the other way:**

| | seed 21230 | seed 36930 |
|---|---|---|
| n | 41 | 41 |
| min score | 0.090006985854 | 0.084564067190 |
| median score | **0.104140541733** | 0.118297903550 |
| score MAD | **0.008719086715** | 0.015337812987 |
| late 10k–20k median | **0.101381556196** | 0.118880324589 |
| late 10k–20k MAD | 0.006928638375 | **0.004966503974** |

The tension is genuine: **36930 @ 3500 wins the explicitly written local-stability
rule**, while **21230** is a much later checkpoint from a run with better overall
trajectory behaviour. This audit does not resolve that tension. It freezes both.

---

## 5. Why provisional `canonical-v1` is superseded for FINAL adjudication

A `canonical-v1` package was already built from seed 36930 @ update 3500, with
source/canonical SHA256
`6773fbb59c7381ba8ddaa944302124a124f5b8a5cb0a5dbb1a5063f3db4a2a91` and embedded
Stage-1 implementation HEAD `7773c77b1df92a6e685dac13c49765ce974f84d8`. It was
verified for seed, `global_update`, exact backbone revision, protocol version, 8
adapter tensors, 3,551,232 parameters, all FP32, all finite, structural
compatibility with 21230 @ 14500, and bit-identical canonical copy. A Stage-1
CLOSED artifact was written at that time.

**It is preserved. It is not deleted and its creation history is not rewritten.**

It is superseded as the FINAL downstream selection for one stated reason:

> a **pre-downstream training-maturity concern** about promoting update
> 3500 of a planned 20,000.

Explicitly **not** the reason: no downstream experiment had been run, and no
downstream result, DEV or TEST, had been produced or inspected. The reopening
occurred **before any downstream result existed**, and therefore the A/B finalist
definition was **not informed by downstream outcomes**.

That is a statement about what evidence could and could not have influenced the
candidate set. It is not a claim that the amendment is costless: this remains a
**post-hoc** amendment to a plan that did not precommit to a two-finalist
adjudication, and it is recorded as a deviation rather than folded back into the
original protocol.

What was reopened is **only**:

> which of exactly two already-trained frozen checkpoints becomes the final
> downstream adapter?

---

## 6. The exact two-finalist freeze

### FINALIST A

| Field | Value |
|---|---|
| role | `stage1_stability_rule_winner` |
| source stage | `final_main` |
| run seed | `36930` |
| update | `3500` |
| LR | `0.0001` |
| `r` | `1.0` |
| `validation/score` | `0.0845640671895974` |
| `robust_score` | `0.10167897852382013` |
| **checkpoint SHA256** | `6773fbb59c7381ba8ddaa944302124a124f5b8a5cb0a5dbb1a5063f3db4a2a91` |
| source repository HEAD | `7773c77b1df92a6e685dac13c49765ce974f84d8` |

### FINALIST B

| Field | Value |
|---|---|
| role | `mature_historical_lr_pilot_alternative` |
| source stage | `lr_pilot` |
| run seed | `21230` |
| update | `14500` |
| LR | `0.0001` |
| `r` | `1.0` |
| `validation/score` | `0.09000698585438581` |
| `robust_score` | `0.106576028028` |
| **checkpoint SHA256** | **`PENDING_AUTHORITATIVE_EVIDENCE` — BLOCKED** |
| source repository HEAD | `bca24ade208265a5a46a54fb2d2d9bd77d8f6703` |

### 6.1 The evidence blocker, and what was searched

Finalist B's exact digest is **not obtainable from this machine**. Searched, all
negative:

* `grep -rn "9405bd76"` across the repository — no match;
* `grep -rn "6773fbb59c7381ba"` — no match either, i.e. **neither** finalist
  digest is recorded anywhere in git;
* `find` for `checkpoint-inventory*`, `*closeout*`, `*canonical*`, `*finalist*`,
  `*stability*rank*` under the repository and under the user's evidence trees —
  the only hit belongs to an unrelated project;
* `find` for `stage1-training`, `stage1-checkpoints`, `UNMARK-BACKUP`,
  `UNMARK-REAL` directories — none present locally;
* `find` for `*.pt` over 1 MB — no UNMARK checkpoints present locally.

`docs/spec/stage1-final-freeze.json` is the **pre-training configuration** freeze
and carries no checkpoint digests.

The observed prefix `9405bd76c0493964...` was **not** expanded. A 16-character
prefix is not a digest: it would let a different file pass verification while
looking authoritative. It is therefore recorded as the explicit sentinel
`PENDING_AUTHORITATIVE_EVIDENCE`, and:

* `require_sha256` refuses any truncated, uppercase, non-hex or wrong-length
  value, including that exact prefix (tested);
* `verify_finalist_checkpoint` refuses to verify a finalist whose digest is
  unbound — there is nothing to verify against;
* `evidence.freeze_complete = false` and `evidence.pending_finalist_digests =
  ["B"]`, and the validator refuses any artifact whose declaration disagrees with
  the finalist records;
* `READY_FOR_STAGE2_PROTOCOL_REVIEW=NO`.

**To clear the blocker**, on the machine holding the checkpoints:

```
python -B scripts/stage1_verify_finalist_checkpoint.py \
    --finalist B --checkpoint <path to seed 21230 update-14500 checkpoint> \
    --emit-binding docs/audits/evidence/048-finalist-b-binding.json
```

This runs the **full** verification of §9.1 — the authoritative
`verify_checkpoint` gate over all twelve scientific-identity fields plus the two
derived objective weights, the inventory pins, and the adapter tensor contract —
**before** reporting the digest, then writes a small evidence record for human
review. It deliberately does **not** edit the freeze artifact: a tool that both
discovers and installs its own evidence would be self-certifying.

**The authoritative binding workflow.** Binding the digest is a **four-field**
edit, enumerated in `finalists.BINDING_FIELDS` and mirrored in the artifact at
`evidence.binding_workflow`:

| # | File | Field | New value |
|---|---|---|---|
| 1 | `unmark/stage1/finalists.py` | `FINALIST_B.checkpoint_sha256` | the verified 64-hex digest |
| 2 | `docs/spec/stage1-adapter-finalists.json` | `adjudication.finalists[key=B].checkpoint_sha256` | the same digest |
| 3 | `docs/spec/stage1-adapter-finalists.json` | `evidence.pending_finalist_digests` | `[]` |
| 4 | `docs/spec/stage1-adapter-finalists.json` | `evidence.freeze_complete` | `true` |

All four must move together. `validate_finalists` refuses any artifact whose
finalist digest disagrees with the code constant — in either direction — and
`validate_freeze_payload` refuses a `pending_finalist_digests` or
`freeze_complete` that disagrees with the finalist records. So every partial
binding fails closed:

| Partial edit | Refused by |
|---|---|
| artifact bound, code still `PENDING` | `validate_finalists` — "binding a digest is a 4-field edit" |
| code bound, artifact still `PENDING` | same |
| both bound, `pending_finalist_digests` still `["B"]` | `validate_freeze_payload` — pending list disagrees |
| both bound, `freeze_complete` still `false` | `validate_freeze_payload` — freeze_complete disagrees |
| a truncated digest bound everywhere | `require_sha256` — not 64 lowercase hex |

Each of those five refusals has its own test (§10), and the all-four-together
case has a test asserting it validates. After binding, re-run the focused tests
and update this audit's §1 verdict and the closing
`READY_FOR_STAGE2_PROTOCOL_REVIEW` line.

---

## 7. Provenance comparison — the two finalists came from different HEADs

Repository HEAD is part of Stage-1 campaign scientific identity in this project,
so A and B are **not** asserted to share one campaign identity.

| | A | B |
|---|---|---|
| HEAD | `7773c77b1df92a6e685dac13c49765ce974f84d8` | `bca24ade208265a5a46a54fb2d2d9bd77d8f6703` |
| stage | `final_main` | `lr_pilot` |
| run seed | 36930 | 21230 |
| init seed | 51800 | 3203 |

They are genuinely different runs, not two views of one run.

### 7.1 Scope of the comparison

The first version of this audit compared only files under `unmark/`, which is not
the executable surface. This revision compares the **complete tracked tree**, then
narrows the claim to what that evidence actually supports.

`git diff --name-only bca24ad..7773c77` over all tracked paths returns 21 files.
Excluding `docs/` and `tests/`, which cannot execute during training, the
executable delta is exactly **seven** files: six under `unmark/` and one script.

### 7.2 The entrypoint and the environment are identical

| Path | B vs A |
|---|---|
| `scripts/stage1_runner.py` — **the `lr-pilot` / `final-main` entrypoint** | **IDENTICAL** (`9a52c9f791b3`) |
| `pyproject.toml` (also holds the pytest config; there is no `setup.py`, `setup.cfg` or `pytest.ini`) | **IDENTICAL** |
| `requirements/base.txt`, `requirements/experiment.txt`, `requirements/monitoring.txt` | **IDENTICAL** |
| `configs/linguistics/vietnamese_syllables.yaml` | **IDENTICAL** |
| `configs/corruption/default.yaml` | **IDENTICAL** |
| `configs/data/uvw_2026.json` | **IDENTICAL** |

So the command that ran both finalists, the declared dependency set, the pinned
inventory manifest, the corruption configuration and the corpus configuration are
byte-identical at the two HEADs.

### 7.3 The training core is byte-identical

Every one of these is identical by git blob hash at both HEADs:

`trainer.py`, `objective.py`, `optim.py`, `sampler.py`, `data.py`,
`validation.py`, `initialisation.py`, `protocol.py`, `corpus.py`, `chunking.py`,
`lengths.py`, `selection.py`, `contracts.py`, `manifest.py`, `checkpoint.py`,
`telemetry.py`, `preflight.py`.

And these whole subtrees are identical:

| Subtree | Tree hash at both HEADs |
|---|---|
| `unmark/modeling` | `0c11a3ab0db5` |
| `unmark/alignment` | `2101f0e2fdaf` |
| `unmark/orthography` | `f21192eaa1a3` |
| `unmark/linguistics` | `1b7b3785540d` |
| `unmark/evaluation` | `ceddcd757ea1` |

The adapter architecture, the objective, the optimizer, the sampler, the
corruption/alignment code, the validation code and every locked constant are the
same source at both HEADs.

### 7.4 The seven executable files that DO differ

| File | Change | Reachable by A or B? |
|---|---|---|
| `unmark/stage1/artifact.py` | selection-artifact validation (Audits 045/047) | runs after training, not during |
| `unmark/stage1/r_phase1_amendment.py` | **new**; post-training handoff rebuild | no |
| `unmark/stage1/fused.py` | **new**; fused `r-phase1` execution | **no** — §7.5 |
| `unmark/stage1/execute.py` | worker-count indirection; `r_phase1`-gated fused branch | yes, but see §7.5 |
| `unmark/stage1/preparation.py` | adds `resolve_preparation_workers`, an env override for worker **count** | yes; default unchanged at 8 |
| `unmark/stage1/device.py` | `torch.device("cuda")` → `torch.device(SCIENTIFIC_DEVICE_BACKEND)` where `SCIENTIFIC_DEVICE_BACKEND = "cuda"`; same substitution in `current_fingerprint` | yes; named-constant refactor, identical values |
| `scripts/stage1_wandb_monitor.py` | adds `"execution_mode"` to `CANDIDATE_CONFIG_KEYS` | **out-of-process** — D-S1B-018 runs the monitor in a separate interpreter and venv; the scientific process never imports it |

### 7.5 Why the fused path cannot have touched either finalist

The fused branch in `execute.py` is guarded by `stage == "r_phase1"`; for any
other stage `r_phase1_execution` is forced to `sequential` and the branch is not
entered. Independently, `require_fused_r_phase1_schedule` fails closed with
*"fused execution is only defined for r_phase1"* on any non-`r_phase1` schedule.
Finalist A is `final_main` and finalist B is `lr_pilot`; neither can reach it.
Fused execution is additionally opt-in through
`UNMARK_STAGE1_R_PHASE1_EXECUTION`, which defaults to `sequential`.

The `preparation.py` change alters worker **count** only. The pre-training freeze
already classifies `workers` as `operational_provenance` with the note "NOT
scientific identity; NOT resume-blocking", and records
`prepared_exact_equality: true` — prepared examples are required byte-identical
across worker counts. At the default the value is unchanged at 8 in both cases.

### 7.6 The claim, narrowed

**Established: training-core source equivalence.** The entrypoint, the dependency
and configuration files, and the entire training-numerics source are byte-identical
at the two HEADs; the three executable files that differ are argued above to be
unreachable, operational-only, or a value-preserving refactor.

**NOT established: full executable or runtime equivalence.**

* Three files on the executable path genuinely differ. Their irrelevance to
  numerics is an argument from reading the code and its guards — it is not a
  byte-identity proof, and it is not a measured numerical comparison.
* Nothing here addresses the **runtime**: torch / CUDA / cuDNN versions, driver,
  or GPU model. This project treats `gpu_name` as **resume-blocking** (D-S1B-015)
  precisely because it declines to assume cross-device numerical identity. The
  two runs are known to span a period in which the campaign moved between GPUs.
* Each checkpoint carries its own `execution` fingerprint, and those live in the
  external `.pt` files, which **could not be read here**. Comparing them is the
  concrete next check; the verifier now emits that fingerprint into the binding
  evidence (§9.1) precisely so the comparison can be made without re-deriving it.
* **Structural compatibility is not equivalence.** A and B sharing 8 tensors,
  identical shapes and 3,551,232 parameters is necessary, not sufficient.
* The two runs differ **by design** in stage, `run_seed` and `init_seed` (51800
  vs 3203). Nothing here claims otherwise.

No claim of "full source-path equivalence" is made.

---

## 8. New scientific state

| Item | State |
|---|---|
| Stage-1 training | **CLOSED** |
| Stage-1 LR selection | **CLOSED** (`0.0001`, author override — D-S1B-020) |
| Stage-1 `r` selection | **CLOSED** (`1.0`, resource-bounded — D-S1B-022) |
| Stage-1 candidate generation | **CLOSED** |
| Allowed final adapter candidates | **EXACTLY TWO**: A = 36930@3500, B = 21230@14500 |
| Candidate universe extensible | **NO** |
| May 17000 / 17500 / 18000 be added later because they look good | **NO** |
| May Stage-1 be retrained after downstream results | **NO** |
| Provisional `canonical-v1` | preserved; superseded as FINAL; not deleted, not rewritten |
| Final downstream adapter | **NOT YET SELECTED** |
| Final adjudication | **OPEN** |
| Allowed adjudication evidence | downstream **DEV only**, under a protocol frozen separately **before** it is run |
| Downstream TEST | **SEALED** |
| TEST used to choose A vs B | **FORBIDDEN** |
| Stage-1 reopening based on downstream | **FORBIDDEN** |

This audit deliberately does **not** define the Stage-2 adjudication protocol.
Dataset, head architecture, head LR, seeds, pooling, DEV metric and winner
criterion are all out of scope and will be locked in a separately reviewed task.

---

## 9. Implementation

### 9.1 `unmark/stage1/finalists.py` (new, 824 lines)

The freeze contract, in code.

* `FinalistFreezeViolation(Stage1ContractViolation)` — follows the existing
  fail-closed exception convention.
* `FINALIST_A`, `FINALIST_B`, `FINALISTS` — frozen `FinalistIdentity` dataclasses.
  `robust_score` is `float | None`: absent where it was not computed, never zero.
* `SHA256_PENDING = "PENDING_AUTHORITATIVE_EVIDENCE"` — deliberately not `null`
  (which reads as "no such field") and not a prefix (which reads as "we know").
* `require_sha256(value, what, *, allow_pending=True)` — full 64 lowercase hex,
  or the sentinel. Everything else is refused.
* `validate_finalists(...)` — the anti-drift core: exactly two; keys exactly
  `{A, B}`; roles exactly the two frozen roles; no duplicate
  `(stage, seed, update)`; no shared digest; and **every pinned field of each
  finalist compared against the frozen constant**, so a coherently edited set that
  is internally consistent is still refused. LR and `r` are pinned to the closed
  values, so a checkpoint from another hyperparameter cannot enter.
* `validate_freeze_payload(...)` / `load_freeze(...)` — validate the committed
  artifact: schema version; all four Stage-1 statuses CLOSED; model identity
  against the **imported** constants; the finalist set; adjudication OPEN with
  `final_adapter_selected == false` and `winner is None`; candidate universe
  closed; `downstream_results_seen == false`; TEST sealed and unused;
  `canonical-v1` preserved and not final; and `freeze_complete` /
  `pending_finalist_digests` consistent with the finalist records.
* `verify_finalist_checkpoint(path, finalist, *, require_bound_digest=True,
  inventory=None)` — **read-only**, and it **delegates to the authoritative
  Stage-1 gate** rather than re-implementing a weaker one.

  The first version of this verifier checked eight provenance fields by hand. The
  contract a resume must satisfy — `RunProvenance.require_match`, reached through
  `trainer.verify_checkpoint` — compares **fourteen**. It was therefore weaker
  than the existing contract on `init_seed`, `corruption_seed`,
  `corpus_manifest_digest`, `inventory` and the two derived objective weights.
  It now calls `verify_checkpoint(payload, expected)` directly, so it enforces:

  | Enforced by `verify_checkpoint` / `require_match` | |
  |---|---|
  | the full `REQUIRED_CHECKPOINT_KEYS` set | `schema_version`, `provenance`, `adapter_state`, `optimizer_state`, `global_update`, `sampler_state`, `cap`, `points`, `execution` |
  | `CHECKPOINT_SCHEMA_VERSION` | `stage1-checkpoint-v2` |
  | scientific identity, 12 fields | `run_seed`, `init_seed`, `corruption_seed`, `learning_rate`, `r`, `corpus_manifest_digest`, `backbone_checkpoint`, `backbone_revision`, `protocol_version`, `precision`, `repository_head`, `inventory` |
  | derived objective weights, 2 | `lambda_align`, `lambda_clean` — must be consistent with the `r` the artifact claims |

  `expected_run_provenance` builds that identity **from the plan, never from the
  artifact under test**: `init_seed` from `adapter_init_seed(run_seed)`,
  `corruption_seed` from `protocol.CORRUPTION_SEED`, the backbone / protocol /
  precision from the dataclass defaults, and two new Audit 048 pins —

  | Pin | Value |
  |---|---|
  | `CORPUS_MANIFEST_DIGEST` | `250859a57d745675c5dba2c7a35df08ccc123988bece873b0c9b29c6e78413d6` |
  | `INVENTORY_SHA256` | `78eeb840d50455b14bd564da5aed7318d96468b8deaad5986b77bf5c538315d2` |
  | `INVENTORY_SOURCE_REVISION` | `135a4d9716e49a981624474156d6f247b9b46f6a` |
  | `INVENTORY_SIZE_BYTES` | `116290` |

  The corpus digest and inventory sha256 are cross-checked by test against
  `docs/spec/stage1-final-freeze.json`, so the two committed specs cannot drift.
  `resolve_inventory()` obtains the inventory identity through the repository's
  own `verify_scientific_inputs` and **fails closed** if it is unavailable —
  skipping the inventory half of the comparison would silently drop a D-S1A-008
  identity field. The three load-bearing inventory pins are additionally checked
  field-by-field against the checkpoint's own record, so a failure names the
  offending field instead of printing two whole dicts.

  On top of that it checks `global_update`, the adapter key set against the locked
  8, that every value is a tensor, FP32 dtype, all-finite, and the 3,551,232
  parameter count. No field absent from the real checkpoint schema was invented.

  Returns an evidence record of identifiers, scalars and hashes only — **no
  tensors** — now including `init_seed`, `corruption_seed`, `lambda_*`, the corpus
  digest, the full inventory block, `cap`, and the checkpoint's `execution`
  fingerprint. The fingerprint is **emitted, not enforced**: `require_match` does
  not compare it and D-S1B-015 treats it as resume-blocking rather than
  run-defining, so reporting it is what lets §7.6's runtime gap be closed later
  without re-deriving anything. Torch is imported lazily, after the digest gate.

### 9.2 `docs/spec/stage1-adapter-finalists.json` (new, 275 lines)

The machine-readable freeze, in the `classification` / `value` style established
by `docs/spec/stage1-final-freeze.json` (`scientific_identity`,
`scientific_protocol`, `scientific_input`, `operational_acceptance`,
`operational_provenance`, `safety_gate`). Named distinctly from the pre-training
configuration freeze so the two cannot be confused. Contains identifiers,
scalars, hashes and provenance only.

This revision adds a `corpus` block (membership digest, dataset, revision), an
`inventory` block (sha256, source revision, size), and three `evidence` entries:
`provenance_fields_verified` — which `validate_freeze_payload` requires to equal
`VERIFIED_PROVENANCE_FIELDS` exactly, so the artifact cannot advertise a stronger
or weaker check than the verifier performs; `binding_workflow` — the four-field
checklist of §6.1; and `execution_fingerprint`, recording that the fingerprint is
emitted into binding evidence rather than enforced.

### 9.3 `scripts/stage1_verify_finalist_checkpoint.py` (new, 187 lines)

Read-only operator verifier, following the existing read-only diagnostic pattern.
The checkpoint path is **operator-supplied**; no `/content/drive/...` path is
hard-coded. Exit codes `0` verified, `1` refused/fail-closed, `2` bad invocation.
`--emit-binding` handles the finalist-B case described in §6.1 and now prints the
full four-field checklist from `BINDING_FIELDS`, so the operator is not left to
infer the workflow from the audit. The script never modifies or copies the
checkpoint, reads no downstream data, and cannot participate in choosing between
A and B.

### 9.4 Tests

`tests/test_stage1_finalist_freeze.py` (torch-free, 695 lines) and
`tests/test_stage1_finalist_checkpoint_torch.py` (needs real torch, 339 lines).

The split is deliberate and follows the warning recorded in
`tests/test_stage1_device_contract.py`: a module-level `importorskip` would skip
the structural checks too. That is not hypothetical here — the first version of
this file was a single module and reported `1 skipped`, silently losing all
torch-free assertions. The `_torch` filename suffix matches
`test_stage1_telemetry_equivalence_torch.py`.

Tests added by this revision:

* **Not weaker than the contract.** `test_the_verifier_covers_every_field_
  require_match_compares` parses `require_match` **with the AST** and asserts the
  compared field list equals `VERIFIED_PROVENANCE_FIELDS` in both directions. If
  `require_match` ever gains a field, this fails rather than the finalist gate
  silently falling behind. A companion test asserts by AST that
  `verify_checkpoint` is actually *called*, so the delegation cannot be quietly
  replaced by a local copy. AST rather than grep, because prose in the module
  mentions both names.
* **Derivation direction.** `expected_run_provenance` is asserted to derive
  `init_seed`, `corruption_seed` and the λ weights rather than read them, to carry
  exactly the 14 contract fields, and to give the two finalists different
  `init_seed`s (51800 vs 3203).
* **Spec drift.** The corpus digest and inventory sha256 are asserted equal to
  `docs/spec/stage1-final-freeze.json`, and wrong pins in the artifact are refused.
* **Binding consistency**, six tests: all-four-together validates; artifact-only,
  code-only, un-cleared pending list, unset `freeze_complete`, and a truncated
  digest are each refused.

### 9.5 `docs/spec/decisions.md`

**D-S1B-023** appended (79 insertions, 0 deletions — strictly append-only; no
existing decision record was altered).

---

## 10. Tests

```
.venv/bin/python -m pytest -q \
  tests/test_stage1_finalist_freeze.py \
  tests/test_stage1_finalist_checkpoint_torch.py

90 passed, 1 skipped in 5.74s
```

The single skip is the **whole torch module**:

```
SKIPPED [1] tests/test_stage1_finalist_checkpoint_torch.py:98:
  adapter tensor inspection needs real torch; run where checkpoints live
```

**torch is not installed in this environment** (`ModuleNotFoundError`), and per
standing instruction it was not installed. That file now defines **20 test
functions expanding to 39 cases**: non-mapping payload; missing
`REQUIRED_CHECKPOINT_KEYS` entries; wrong checkpoint schema; wrong
`global_update`; **twelve** wrong-scientific-identity-field cases covering every
field `require_match` compares; inconsistent and missing derived objective
weights; three wrong-inventory-pin cases; missing provenance; missing
`adapter_state`; dropped tensor; extra tensor; wrong parameter count; non-FP32;
non-tensor entry; NaN; Inf; verifier non-mutation; execution-fingerprint
emission; and the bound-digest round trip. **They were not executed here and are
not reported as passing.**

Two mitigations were applied so a skipped file is not simply an unknown:

* **All sixteen `require_match` refusal messages** those tests assert on were
  exercised directly in this environment — `RunProvenance.require_match` is
  torch-free — confirming every field is genuinely refused and every `match=`
  pattern hits. Without this, a wrong pattern would have surfaced only on the
  torch host.
* The adapter fixture's tensor shapes were checked arithmetically: the 8 keys
  equal `ADAPTER_STATE_KEYS`, and the shapes sum to exactly 3,551,232 — matching
  `AdapterConfig(hidden_size=768).parameter_count().total` derived independently
  from the real architecture.

The binding-consistency guard was additionally mutation-checked: with the guard
in place, artifact-bound-only, un-cleared pending list, unset `freeze_complete`
and a truncated digest are each refused, while all four bound together validates.

### 10.1 Authoritative real-torch validation on Colab

The torch half was subsequently executed on the authoritative environment, against
commit `a1bd8573bbb6734348943c9dd713495c041d019a`.

| | |
|---|---|
| Python | 3.13.15 |
| torch | 2.11.0+cu128 |
| CUDA | 12.8 |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition |

```
129 collected
127 passed, 2 failed
```

**Both failures were stale test expectations, not verifier defects.** The
verifier behaved correctly and failed closed in both cases:

| Test | Expected regex | Actual `FinalistFreezeViolation` |
|---|---|---|
| `test_a_missing_provenance_block_is_refused` | `no provenance` | `checkpoint is missing ['provenance']; cannot resume exactly` |
| `test_a_missing_adapter_state_is_refused` | `no adapter_state` | `checkpoint is missing ['adapter_state']; cannot resume exactly` |

`no provenance` and `no adapter_state` were the wording of the **local guards
this verifier used before verification was delegated** to
`trainer.verify_checkpoint` (§9.1). The authoritative gate checks
`REQUIRED_CHECKPOINT_KEYS` **first**, so for an absent key those local guards are
now unreachable and the message comes from the contract. The two regexes were
left behind by the delegation change; the production code is right and was **not**
altered to satisfy them.

**Repair — test expectations only.** Both tests now assert the contract's two
invariants through a shared `_assert_missing_required_key` helper: the exception
is a `FinalistFreezeViolation`, its message says `checkpoint is missing`, and it
names the absent key. They no longer pin an exact sentence, and they do not match
the temporary path that also appears in the message.

Three further items were addressed while inspecting adjacent tests:

* `execution` was missing from the parametrized `REQUIRED_CHECKPOINT_KEYS` sweep
  — a real gap in the same contract. Added, and that test now uses the same
  helper. The torch file goes from 39 to **40 cases**.
* `test_a_wrong_inventory_pin_in_the_checkpoint_is_refused` carried a docstring
  claiming the operator is told *which* inventory field is wrong. That is not
  true after delegation: `require_match` compares `inventory` as a whole block
  and fires first, naming `inventory`. The test already asserted only the
  exception type, so no assertion changed; the docstring was corrected.
* **Two torch-free regression guards were added**, because this drift was
  locally detectable and simply was not checked — `verify_checkpoint` and
  `require_match` need no torch. One walks every `REQUIRED_CHECKPOINT_KEYS` entry
  and asserts the `checkpoint is missing <key>` shape; the other parses the torch
  file with the AST and fails if either retired regex is reintroduced. The second
  was verified to bite: restoring `match="no provenance"` fails it, and the probe
  was reverted byte-exactly.

**What this run did NOT establish.** The gate stopped at the test failures, so
**authoritative verification of the real A and B checkpoints was not performed**.
No downstream data was read and no checkpoint was modified. Finalist B's digest
remains `PENDING_AUTHORITATIVE_EVIDENCE`, `evidence.freeze_complete` remains
`false`, and `READY_FOR_STAGE2_PROTOCOL_REVIEW` remains `NO`.

Local suite after the repair: **92 passed, 1 skipped** (the two new torch-free
guards; the skip is still the whole torch module).

Related existing suites, unaffected:

```
.venv/bin/python -m pytest -q tests/test_stage1_final_freeze.py \
  tests/test_stage1_artifact_identity.py tests/test_stage1_r_phase1_amendment.py \
  tests/test_stage1_torch_contracts.py

90 passed, 7 skipped in 0.32s
```

Whole-repository regression suite:

```
.venv/bin/python -m pytest -q

4165 passed, 108 skipped in 159.97s (0:02:39)
```

Zero failures and zero errors (`grep -cE "^(FAILED|ERROR)"` returned `0`). One of
the 108 skips is this audit's torch module; the remainder are the repository's
pre-existing skips, predominantly torch- and CUDA-gated. No environmental
forkserver/socket permission failures occurred in this environment.

```
git diff --check
```

produced no output.

**No Stage-1 training was run. No optimizer step was executed. No downstream
data, DEV or TEST, was read.**

---

## 11. Data and artifact boundary

**Stays external** (durable scientific evidence, never copied into git): the
`.pt` checkpoints for both finalists and the archived 36930 checkpoints
17000–20000; the ~2.2 GB prepared corpus; HF and tokenizer caches; raw W&B
history dumps; training telemetry archives; the Drive campaign directories; the
`canonical-v1` package itself.

**Represented in git**: protocol/spec/decision metadata, exact cryptographic
identities, source run identity, the verifier and its tests, and this audit.

Assurance: every file added or modified by this audit is small text — the largest
is `unmark/stage1/finalists.py` at 37 KB. `git status --porcelain
--untracked-files=all` shows no `*.pt`, no `chunks.jsonl`, no `wandb` artifact.
`.gitignore:46` already ignores `*.pt`, confirmed with `git check-ignore -v`.

---

## 12. Reproducibility and paper evidence

The repository preserves enough identity metadata to map a future figure or table
back to durable artifacts without treating a W&B screenshot as primary numeric
evidence:

* both finalists are named by `(source_stage, run_seed, update, LR, r)` plus
  source repository HEAD, and A additionally by full checkpoint SHA256;
* `evidence.validation_trajectories` records that both runs have full 41-point
  `0..20000` histories reconstructed by the read-only closeout, and that update 0
  was recovered from checkpoint `points` history rather than from telemetry;
* the locked-rule best checkpoints and the stability-rule ranking with local
  windows and `robust_score` values are recorded in §4 above;
* the whole-run descriptive statistics in §4 are recorded as numbers, so a paper
  table can be regenerated and checked against them;
* `evidence.heavy_artifacts_location` states plainly that checkpoints, corpus,
  telemetry and W&B history are external and not committed.

A future figure should cite the checkpoint SHA256 and run identity; W&B remains a
convenience view over evidence whose authority is the checkpoint payload.

---

## 13. Deviations and limitations

1. **Finalist B's checkpoint SHA256 is unbound.** §6.1. The freeze is
   consequently **incomplete**, and this is enforced mechanically, not merely
   noted.
2. **The torch half of the verifier is not executed locally.** §10. It HAS now
   been executed on the authoritative Colab host (§10.1): 127 passed, 2 failed,
   both failures stale regexes since repaired. It has **not** been re-run there
   since the repair, so the repaired expectations are verified only by the
   torch-free reproduction of the same contract messages in this environment.
3. **Authoritative A/B checkpoint verification has still not been performed.**
   §10.1. The Colab gate stopped at the test failures before reaching it, so no
   real checkpoint has been verified against its frozen identity.
4. **Full executable / runtime equivalence between the two HEADs is not
   established.** §7.6. Training-core source equivalence is established and citable; three executable
   files differ on argued-irrelevant grounds, and the `execution` fingerprints in
   the two checkpoints have not been compared.
5. **Finalist A's checkpoint was not verified here either.** Its digest is
   recorded from prior evidence; no local file was hashed to confirm it, because
   no UNMARK checkpoint exists on this machine.
6. **One of three final-main seeds ran.** §3. Seed 7309 produced no checkpoint and
   seed 5993 never started; the finalist set therefore draws on a single
   final-main run plus a historical LR-pilot run.
7. **The stability rule and the whole-run statistics disagree** about which
   finalist is preferable. §4. That is why both are frozen and neither is
   selected.
8. **`unmark-proposal.md` is not updated** and the PDF is stale, consistent with
   how D-S1B-020 and D-S1B-022 were handled: this is recorded as an explicit
   amendment, not folded back into the original plan.

---

## 14. Final repository state

```
git status --short
 M docs/spec/decisions.md
?? docs/audits/048-stage1-finalist-freeze-pre-downstream-adjudication.md
?? docs/spec/stage1-adapter-finalists.json
?? scripts/stage1_verify_finalist_checkpoint.py
?? tests/test_stage1_finalist_checkpoint_torch.py
?? tests/test_stage1_finalist_freeze.py
?? unmark/stage1/finalists.py
```

Seven entries: **one modified file and six new**. Six of them are the change-set
proper — one modified (`docs/spec/decisions.md`) plus five new implementation and
test files — and the seventh is this audit document itself.

| | File | State |
|---|---|---|
| 1 | `docs/spec/decisions.md` | modified — D-S1B-023 appended |
| 2 | `unmark/stage1/finalists.py` | new |
| 3 | `docs/spec/stage1-adapter-finalists.json` | new |
| 4 | `scripts/stage1_verify_finalist_checkpoint.py` | new |
| 5 | `tests/test_stage1_finalist_freeze.py` | new |
| 6 | `tests/test_stage1_finalist_checkpoint_torch.py` | new |
| — | `docs/audits/048-…md` | new — this audit |

There were **no pre-existing modifications**: the tree was clean at the start, so
every entry above was created by this audit. `docs/spec/decisions.md` is the only
modified file and its diff is `79 insertions(+), 0 deletions(-)` — append-only.

### 14.1 First change set — COMMITTED AND PUSHED

The reviewed change set above is on `main` and on the remote as:

```
a1bd8573bbb6734348943c9dd713495c041d019a
Freeze Stage-1 adapter finalists for downstream adjudication
parent 7773c77b1df92a6e685dac13c49765ce974f84d8
7 files changed, 3216 insertions(+)
```

`origin/main` is at this commit, so it is fetchable by Colab; this is the
revision the real-torch run in §10.1 validated.

An earlier commit object `8acb2ed5d2957cbfabdb5e9b423f331d9fc4af78` was created
for the same change set and then reset away by the author, who re-committed it as
`a1bd857` under their own authorship. `git diff 8acb2ed a1bd857` is empty — the
two commits have **identical trees**; only the commit object and message differ.
`8acb2ed` is not an ancestor of `HEAD` and should not be cited anywhere.

### 14.2 Second change set — real-torch repair, UNCOMMITTED

The authoritative Colab run (§10.1) then reported two stale test expectations.
The repair touches **test files and this audit only** — no production code, no
scientific behaviour, no change to the finalist set:

```
git status --short
 M docs/audits/048-stage1-finalist-freeze-pre-downstream-adjudication.md
 M tests/test_stage1_finalist_checkpoint_torch.py
 M tests/test_stage1_finalist_freeze.py
```

```
 ...-finalist-freeze-pre-downstream-adjudication.md | 101 ++++++++++++++++++---
 tests/test_stage1_finalist_checkpoint_torch.py     |  38 +++++++-
 tests/test_stage1_finalist_freeze.py               |  63 +++++++++++++
 3 files changed, 183 insertions(+), 19 deletions(-)
```

**Nothing from this second change set was committed or pushed**, and nothing was
staged. No destructive git command was used, and no prior scientific artifact was
deleted or rewritten. Finalist B's digest is untouched at
`PENDING_AUTHORITATIVE_EVIDENCE`, `evidence.freeze_complete` remains `false`, and
the finalist universe remains exactly {A, B}.

## 15. Exact next allowed step

**Not** "run Stage 2." In order:

1. **Independently review this audit** and the six files it adds or modifies
   (one modified, five new — enumerated in §14).
2. **Re-run the real-torch suite on the authoritative host** to confirm the two
   repaired expectations pass there (§10.1). The previous run stopped at those
   failures before reaching any real checkpoint.
3. **Clear the finalist-B evidence blocker** by running the verifier against the
   real seed-21230 update-14500 checkpoint (§6.1), reviewing the emitted binding
   record, and updating the freeze artifact so `freeze_complete` becomes `true`.
   While comparing, also compare the two checkpoints' `execution` fingerprints to
   close the §7.6 gap.
4. **Design and freeze the downstream DEV-only adjudication / Stage-2 protocol**
   — dataset, head, seeds, pooling, DEV metric, winner criterion, tie-breaks —
   and have it reviewed **before any downstream result is produced**.

Only after all four may a downstream experiment be run.

---

```
STAGE1_TRAINING=CLOSED
STAGE1_CANDIDATE_GENERATION=CLOSED
FINALIST_COUNT=2
FINALIST_A=36930@3500
FINALIST_B=21230@14500
FINAL_ADAPTER_SELECTED=NO
DOWNSTREAM_RESULTS_SEEN=NO
DOWNSTREAM_TEST=SEALED
STAGE2_STARTED=NO
READY_FOR_STAGE2_PROTOCOL_REVIEW=NO
```

`READY_FOR_STAGE2_PROTOCOL_REVIEW=NO` because finalist B's exact checkpoint
SHA256 has not been verified (§6.1) and `evidence.freeze_complete` is `false`.
