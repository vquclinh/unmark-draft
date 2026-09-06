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

**PASS.**

The finalist freeze is implemented, pinned in code, machine-readable, tested, and
— since the authoritative verification of §10.2 — **complete**. Both finalists
have been verified read-only against their frozen identities, and finalist B's
digest is bound from that evidence.

An earlier revision of this audit read PASS WITH ISSUES while B's digest was
unbound. That blocker is now cleared; the entry is kept in the table below so the
history of the verdict is visible rather than silently rewritten.

| | |
|---|---|
| Finalist set frozen at exactly two | **PASS** |
| Provisional `canonical-v1` preserved, superseded as FINAL | **PASS** |
| Stage-1 training / LR / `r` / candidate generation recorded CLOSED | **PASS** |
| Adjudication recorded OPEN, final adapter UNSELECTED | **PASS** |
| Official UIT-VSFC TEST sealed; TEST-based selection refused | **PASS** |
| Read-only checkpoint verifier | **PASS** — real-torch run on Colab refused every malformed payload, §10.1 |
| Real-torch test expectations | **REPAIRED** — 2 stale regexes, §10.1 |
| Structural prevention of finalist-set drift | **PASS** |
| Training-core source equivalence of the two HEADs | **PASS** — established, §7.6 |
| Checkpoint provenance verification vs the Stage-1 contract | **PASS** — delegates to `verify_checkpoint`, §9.1 |
| **Finalist B exact checkpoint SHA256** | **CLEARED** — bound from authoritative evidence, §10.2 |
| Authoritative A/B checkpoint verification | **PASS / PASS** — §10.2 |
| Freeze completeness | **COMPLETE** — `freeze_complete = true`, no pending digests |
| Runtime-environment comparison (execution fingerprints) | **CLOSED** — exactly equal, §10.2 |
| Full executable equivalence of the two runs | **NOT CLAIMED** — §7.6, §10.2 |

`evidence.freeze_complete` is `true`, `pending_finalist_digests` is `[]`, and
`READY_FOR_STAGE2_PROTOCOL_REVIEW=YES`. **A complete freeze does not select an
adapter**: the final downstream adapter is still UNSELECTED and adjudication is
still OPEN.

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
| **checkpoint SHA256** | `9405bd76c04939641170cb71507ce8eb669eb2987016b86b495a403ceafcb9d2` — verified, §10.2 |
| source repository HEAD | `bca24ade208265a5a46a54fb2d2d9bd77d8f6703` |

### 6.1 The evidence blocker — RESOLVED in §10.2

> **Status:** this section records the blocker as it stood before the
> authoritative verification. It is retained because the *reasoning* — why a
> 16-character prefix was refused — is what the binding workflow rests on. The
> digest is now bound; see §10.2.


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
* **[CLOSED in §10.2]** Nothing *in this section* addresses the **runtime**:
  torch / CUDA / cuDNN versions, driver, or GPU model. The authoritative
  verification later compared the two checkpoints' embedded execution
  fingerprints and found them **exactly equal**, closing this gap at the
  fingerprint level. The reasoning below is retained as written. This project treats `gpu_name` as **resume-blocking** (D-S1B-015)
  precisely because it declines to assume cross-device numerical identity. The
  two runs are known to span a period in which the campaign moved between GPUs.
* Each checkpoint carries its own `execution` fingerprint, and those live in the
  external `.pt` files, which could not be read *here*. That comparison has since
  been performed on the authoritative host and is recorded in §10.2.
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

**D-S1B-023** appended with the first change set (79 insertions, 0 deletions).
**D-S1B-024** appended with the binding change set (63 insertions, 0 deletions),
recording the authoritative verification, B's bound digest, the equal execution
fingerprints, and that the completed freeze selects nothing.

Both edits are **strictly append-only**; no existing decision record was altered.
D-S1B-023 deliberately still reads `PENDING` for finalist B's digest: that was
true when it was written and committed, and the decision log is a history, not a
current-state view. D-S1B-024 supersedes that state rather than rewriting it.

An earlier revision of this change set edited D-S1B-023 in place (18 insertions,
12 deletions). That was wrong — it destroyed the append-only property of an
already-committed record — and was reverted; `docs/spec/decisions.md` now differs
from `054c6d8` by appended lines only.

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
**authoritative verification of the real A and B checkpoints was not performed in
this run**. No downstream data was read and no checkpoint was modified. As of
this run finalist B's digest was still `PENDING_AUTHORITATIVE_EVIDENCE`,
`evidence.freeze_complete` was `false`, and `READY_FOR_STAGE2_PROTOCOL_REVIEW`
was `NO`. **All three were superseded by §10.2**, which records the verification
that followed once the repaired gate passed.

Local suite after the repair: **92 passed, 1 skipped** (the two new torch-free
guards; the skip is still the whole torch module).

### 10.2 Authoritative checkpoint verification — COMPLETE

After the §10.1 test repair the real-torch gate passed, and the authoritative
read-only verification of both finalists was allowed to proceed.

| | |
|---|---|
| Implementation commit | `054c6d8fa4c912f10a2bb4c21e272e5064e8f355` |
| Environment | Python 3.13.15, torch 2.11.0+cu128, CUDA 12.8, NVIDIA RTX PRO 6000 Blackwell Server Edition |

| | FINALIST A | FINALIST B |
|---|---|---|
| identity | `final_main` / seed 36930 / update 3500 | `lr_pilot` / seed 21230 / update 14500 |
| checkpoint SHA256 | `6773fbb59c7381ba8ddaa944302124a124f5b8a5cb0a5dbb1a5063f3db4a2a91` | `9405bd76c04939641170cb71507ce8eb669eb2987016b86b495a403ceafcb9d2` |
| verification | **PASS** | **PASS** |
| adapter tensors | 8 | 8 |
| adapter parameters | 3 551 232 | 3 551 232 |
| dtype | fp32 | fp32 |
| all finite | true | true |

Evidence, retained externally:

```
B binding          .../stage1-finalist-verification/054c6d8fa4c9/20260906T120114Z/048-finalist-b-binding.json
execution compare  .../stage1-finalist-verification/054c6d8fa4c9/20260906T120114Z/048-execution-comparison.json
```
(under `/content/drive/MyDrive/UNMARK/UNMARK-BACKUP/`)

**The §6.1 evidence blocker is CLEARED.** Finalist B's digest is bound from this
verification, not from the 16-character prefix that had been visible throughout.
The bound value does begin with that prefix, which is a consistency check on the
evidence rather than its source. The four-field binding of `BINDING_FIELDS` was
applied together; `evidence.freeze_complete` is now `true` and
`evidence.pending_finalist_digests` is `[]`.

**Execution fingerprints are exactly equal.** `EXECUTION_FINGERPRINTS_EQUAL=true`:

| field | value (both checkpoints) |
|---|---|
| `backend` | cuda |
| `device` | cuda |
| `gpu_name` | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| `compute_capability` | 12.0 |
| `torch_version` | 2.11.0+cu128 |
| `cuda_version` | 12.8 |
| `cudnn_version` | 91900 |
| `deterministic_algorithms` | true |
| `cudnn_deterministic` | true |
| `cudnn_benchmark` | false |
| `cublas_workspace_config` | `:4096:8` |
| `float32_matmul_precision` | highest |
| `cuda_matmul_allow_tf32` | false |
| `cudnn_allow_tf32` | false |

This is exactly the comparison §7.6 identified as missing, and it is the reason
the verifier emits the fingerprint into the evidence record rather than enforcing
it. **The previously identified runtime-equivalence gap is therefore closed at
the checkpoint-embedded execution-fingerprint level.**

**What this does NOT establish.** A and B are still **distinct runs**, not one
run and not one campaign identity: different source stages (`final_main` vs
`lr_pilot`), different run seeds (36930 vs 21230), different init seeds (51800 vs
3203) and different source repository HEADs. Equality of execution fingerprints
closes the runtime-environment comparison and nothing else; §7.6's training-core
source equivalence is unchanged, and no claim of full executable equivalence is
made.

**A complete freeze does not select an adapter.** Both finalists are now fully
verified and bound. The final downstream adapter remains **UNSELECTED**,
adjudication remains **OPEN** and DEV-only under a protocol that does not yet
exist, and the official UIT-VSFC TEST remains **SEALED**.

No downstream data or result was read, no checkpoint was modified, no training
occurred, and Stage 2 has not started.

Local suite after binding: **94 passed, 1 skipped** — the six tests that encoded
B's former pending state were rewritten to exercise the same generic guards
against a synthetic unbound identity, so the `SHA256_PENDING` contract keeps its
coverage while no longer asserting a superseded fact about B.

Related existing suites, unaffected:

```
.venv/bin/python -m pytest -q tests/test_stage1_final_freeze.py \
  tests/test_stage1_artifact_identity.py tests/test_stage1_r_phase1_amendment.py \
  tests/test_stage1_torch_contracts.py tests/test_stage1_runner_contract.py \
  tests/test_stage1_schedule.py

159 passed, 7 skipped in 0.49s
```

Whole-repository regression suite:

```
.venv/bin/python -m pytest -q

4169 passed, 108 skipped in 154.15s (0:02:34)
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
  source repository HEAD **and a full authoritative checkpoint SHA256** — A
  `6773fbb5…a2a91`, B `9405bd76…b9d2` — each verified read-only against its
  frozen identity (§10.2);
* `evidence.validation_trajectories` records that both runs have full 41-point
  `0..20000` histories reconstructed by the read-only closeout, and that update 0
  was recovered from checkpoint `points` history rather than from telemetry;
* the locked-rule best checkpoints and the stability-rule ranking with local
  windows and `robust_score` values are recorded in §4 above;
* the whole-run descriptive statistics in §4 are recorded as numbers, so a paper
  table can be regenerated and checked against them;
* `evidence.heavy_artifacts_location` states plainly that checkpoints, corpus,
  telemetry and W&B history are external and not committed.

A future figure should cite the checkpoint SHA256 and run identity — both are now
authoritative for both finalists; W&B remains a convenience view over evidence
whose authority is the checkpoint payload.

---

## 13. Deviations and limitations

1. **[RESOLVED]** Finalist B's checkpoint SHA256 was unbound; it is now bound
   from the authoritative verification (§10.2) and the freeze is complete.
2. **The torch half of the verifier is not executed locally.** §10. It has been
   executed on the authoritative Colab host twice: the first run (§10.1, under
   `a1bd857`) reported 127 passed / 2 failed, both failures stale regexes since
   repaired; the **repaired gate was then re-run under
   `054c6d8fa4c912f10a2bb4c21e272e5064e8f355` and PASSED**, which is what allowed
   the A/B verification of §10.2 to proceed. No exact pass count for that second
   run is present in the retained evidence, so none is stated here. The remaining
   limitation is only that this repository's own environment cannot execute it.
3. **[RESOLVED]** Authoritative A/B checkpoint verification has been performed:
   both PASS (§10.2).
4. **Full executable equivalence between the two HEADs is not established.**
   §7.6. Training-core source equivalence is established and citable, and the
   runtime-environment comparison is now **closed**: the two checkpoints'
   embedded execution fingerprints were compared on the authoritative host and
   are **exactly equal** (§10.2). What remains unestablished is narrower — three
   executable files still differ, and their irrelevance to numerics is an argument
   from reading the code and its guards rather than a byte-identity proof or a
   measured numerical comparison.
5. **[RESOLVED]** Finalist A's checkpoint has now been verified on the
   authoritative host (§10.2). It still cannot be verified locally, because no
   UNMARK checkpoint exists on this machine.
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

## 14. Repository state

This audit spans three change sets. **Only §14.4 is the current working-tree
state**; §14.1-§14.3 are history and their `git status` output is quoted as it
stood at the time, not now.

### 14.1 Change set 1 — the finalist freeze. COMMITTED.

Seven files: `docs/spec/decisions.md` modified (D-S1B-023 appended, 79
insertions, 0 deletions) plus six new — `unmark/stage1/finalists.py`,
`docs/spec/stage1-adapter-finalists.json`,
`scripts/stage1_verify_finalist_checkpoint.py`,
`tests/test_stage1_finalist_freeze.py`,
`tests/test_stage1_finalist_checkpoint_torch.py`, and this audit. The tree was
clean beforehand, so there were no pre-existing modifications.

```
a1bd8573bbb6734348943c9dd713495c041d019a
Freeze Stage-1 adapter finalists for downstream adjudication
parent 7773c77b1df92a6e685dac13c49765ce974f84d8
7 files changed, 3216 insertions(+)
```

An earlier commit object `8acb2ed5d2957cbfabdb5e9b423f331d9fc4af78` was created
for the same change set and then reset away by the author, who re-committed it as
`a1bd857` under their own authorship. `git diff 8acb2ed a1bd857` is empty — the
two commits have **identical trees**; only the commit object and message differ.
`8acb2ed` is not an ancestor of `HEAD` and should not be cited anywhere.

### 14.2 Change set 2 — the real-torch test repair. COMMITTED.

```
054c6d8fa4c912f10a2bb4c21e272e5064e8f355
fix finalist freeze
3 files changed, 228 insertions(+), 22 deletions(-)
```

This is also the implementation commit under which the repaired gate was re-run
and the authoritative checkpoint verification of §10.2 was performed.

### 14.3 Change set 3 — finalist B binding. UNCOMMITTED.

The authoritative verification supplied B's digest. The four-field binding of
`BINDING_FIELDS` was applied together, the tests that encoded B's former pending
state were rewritten, and D-S1B-024 was appended.

`unmark/stage1/finalists.py` changes only `FINALIST_B.checkpoint_sha256` and its
docstring — 8 insertions, 5 deletions. No verification logic, no adapter
contract, no finalist identity field and no scientific constant was altered. The
`SHA256_PENDING` sentinel and every guard around it are preserved as the generic
contract for any future unbound digest; only assertions about **B's** state were
updated, and the six affected tests were rewritten to exercise the same guards
against a synthetic unbound identity rather than deleted.

### 14.4 Current state — UNCOMMITTED, awaiting author review

```
$ git rev-parse HEAD
054c6d8fa4c912f10a2bb4c21e272e5064e8f355

$ git rev-parse origin/main
054c6d8fa4c912f10a2bb4c21e272e5064e8f355

$ git status --short
 M docs/audits/048-stage1-finalist-freeze-pre-downstream-adjudication.md
 M docs/spec/decisions.md
 M docs/spec/stage1-adapter-finalists.json
 M tests/test_stage1_finalist_freeze.py
 M unmark/stage1/finalists.py
```

```
$ git diff --stat
 ...-finalist-freeze-pre-downstream-adjudication.md | 310 +++++++++++++++------
 docs/spec/decisions.md                             |  63 +++++
 docs/spec/stage1-adapter-finalists.json            |  22 +-
 tests/test_stage1_finalist_freeze.py               | 121 +++++---
 unmark/stage1/finalists.py                         |  13 +-
 5 files changed, 398 insertions(+), 131 deletions(-)
```

`HEAD` and `origin/main` are both at `054c6d8`, confirmed against the live remote
with `git ls-remote`, so the two committed change sets are pushed and fetchable;
this working tree is one uncommitted change set ahead of both.

**Nothing in this change set was committed or pushed**, and nothing is staged. No
destructive git command was used, no git history was mutated, no prior scientific
artifact was deleted or rewritten, no checkpoint was modified, and the finalist
universe remains exactly {A, B}.

## 15. Exact next allowed step

**Not** "run Stage 2." Steps 1-3 of the previous revision are complete: the audit
was reviewed, the real-torch suite was repaired and passed, and the finalist-B
blocker is cleared with both checkpoints verified and the execution-fingerprint
gap closed (§10.2).

What remains, in order:

1. **Independently review this revision** — the four-field binding, the rewritten
   tests, §10.2, and the appended **D-S1B-024** — and commit it. The author
   performs all commits and pushes. Note that `docs/spec/decisions.md` is
   append-only here: D-S1B-023 is unchanged from `054c6d8` and still records the
   pending state that was true when it was written.
2. **Design and freeze the downstream DEV-only adjudication / Stage-2 protocol**
   — dataset, head architecture, head LR, seeds, pooling, DEV metric, winner
   criterion, tie-breaks — and have it independently reviewed **before any
   downstream result is produced**. It is deliberately **not** defined here.
3. Only then may a downstream DEV experiment be run, and only DEV: the official
   UIT-VSFC TEST stays SEALED and may never adjudicate A vs B.

A complete freeze is **not** a selection. Nothing about A or B has been chosen.

---

```
STAGE1_TRAINING=CLOSED
STAGE1_CANDIDATE_GENERATION=CLOSED
FINALIST_COUNT=2
FINALIST_A=36930@3500
FINALIST_B=21230@14500
FINALIST_A_VERIFIED=PASS
FINALIST_B_VERIFIED=PASS
FINALIST_B_DIGEST_BLOCKER=CLEARED
FREEZE_COMPLETE=YES
PENDING_FINALIST_DIGESTS=[]
EXECUTION_FINGERPRINTS_EQUAL=true
FINAL_ADAPTER_SELECTED=NO
ADJUDICATION=OPEN
DOWNSTREAM_RESULTS_SEEN=NO
DOWNSTREAM_TEST=SEALED
STAGE2_STARTED=NO
READY_FOR_STAGE2_PROTOCOL_REVIEW=YES
```

`READY_FOR_STAGE2_PROTOCOL_REVIEW=YES`: both finalists are verified against their
frozen identities, B's digest is bound from authoritative evidence, and
`evidence.freeze_complete` is `true`. This authorises **designing and reviewing
the Stage-2 adjudication protocol** — not running it, and not selecting an
adapter. `FINAL_ADAPTER_SELECTED=NO` and `ADJUDICATION=OPEN` are the operative
facts for what happens next.
