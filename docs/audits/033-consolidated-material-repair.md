# Audit 033 — Consolidated Material Repair

**Scope:** ONE consolidated repair of the five material findings agreed by Audit 031 and Audit 032.
**Date:** 2026-08-26
**Mode:** REPAIR. Production code, tests and one new module were changed. No prior audit was modified.

---

## 1. Exact Starting HEAD

```
$ git rev-parse HEAD
55aa4064780b37626bcae7eef83c504a96fcc51f
```

Lineage verified mechanically rather than from commit-message wording:

```
$ git diff --name-status 479fac5bb5fb7be4518e8ed36162c137700851ed HEAD
M       docs/audits/030-pretrain-repository-wide-audit.md
```

The scientific implementation at HEAD is byte-identical to `479fac5`; every commit since changed
documentation only.

## 2. Exact Starting Working Tree

```
$ git status --short
?? docs/audits/031-final-pretraining-full-repository-review.md
?? docs/audits/032-second-independent-full-repository-pretrain-review.md

$ git diff --cached --name-status      (empty — nothing staged)
$ git diff --check                     (clean, exit 0)
```

## 3. Audit 031 / 032 Material Finding Union

| # | Finding | 031 | 032 | Severity |
|---|---|---|---|---|
| 1 | `objective_cls` unbound in `execute_stage` → `NameError` before `train_run` | `B1` | *missed* | BLOCKER |
| 2 | `ValidationPoint(**p)` cannot read writer-emitted `to_dict()` (`score`) | `B2` | `B1` | BLOCKER |
| 3 | Persisted `cap` never read; 20k→40k continuation unrecoverable | `B3` | `B2` | BLOCKER |
| 4 | Stage handoff validates only `stage` + `protocol_version` | `B4` | `MAJ1` | BLOCKER / MAJOR (disputed) |
| 5 | `--repository-head` optional, caller-supplied, unverified; no dirty-tree guard | `B5` | `MAJ2` | BLOCKER / MAJOR (disputed) |

All five are repaired here. The 031/032 severity disagreement on #4 and #5 is not resolved and did
not need to be: both audits agreed both must be fixed before the campaign.

## 4. Files Changed

**Modified (production):**
- `unmark/stage1/execute.py`
- `unmark/stage1/trainer.py`
- `unmark/stage1/selection.py`
- `unmark/stage1/checkpoint.py`
- `scripts/stage1_runner.py`

**Added (production):**
- `unmark/stage1/artifact.py` — campaign identity + artifact consumer validation (torch-free)

**Modified (tests):**
- `tests/test_stage1_training_resume_state.py`
- `tests/test_stage1_run_independence.py`
- `tests/test_stage1_pretrain_audit.py`

**Added (tests):**
- `tests/test_stage1_name_resolution.py`
- `tests/test_stage1_resume_state_machine.py` (torch-free)
- `tests/test_stage1_resume_state_machine_torch.py` (torch-gated)
- `tests/test_stage1_artifact_identity.py`
- `tests/test_stage1_repository_provenance.py`

```
 scripts/stage1_runner.py                   |  87 ++++++++++++-----
 tests/test_stage1_pretrain_audit.py        |  37 +++++++-
 tests/test_stage1_run_independence.py      |   6 +-
 tests/test_stage1_training_resume_state.py |  67 +++++++++++--
 unmark/stage1/checkpoint.py                | 116 +++++++++++++++++++++-
 unmark/stage1/execute.py                   |  55 +++++++++--
 unmark/stage1/selection.py                 |  98 +++++++++++++++++++
 unmark/stage1/trainer.py                   | 122 +++++++++++++++++++++++-
```

`unmark/stage1/protocol.py`, `configs/`, `docs/spec/`, the proposal and the decisions record were
**not touched**.

## 5. Repair 1 — Objective Construction

`execute.py:299`:

```python
-          objective = objective_cls(unmark_encoder, provenance.weights)
+          objective = Stage1Objective(unmark_encoder, provenance.weights)
```

`Stage1Objective` is already imported inside `execute_stage` (line 192, lazy so torch stays lazy),
so it is a genuine **local** binding. `objective_cls` was a stale name left by the Audit 030 §AD
refactor that split `build_objective` into `build_backbone` + per-run construction: `build_objective`
still returns the class as its third element, and `smoke_check` (line 519) legitimately unpacks it —
that call site is unchanged and correct.

Constraints preserved: construction stays **inside** the nominal-run loop, uses that run's fresh
`UnmarkEncoder` and that run's `provenance.weights`, and no mutable objective state moved outside the
loop. The signature matches `Stage1Objective.__init__(self, unmark_encoder, weights)` exactly, which
is the same call shape `smoke_check` uses. No objective mathematics changed.

## 6. Repair 2 — `ValidationPoint` Writer/Reader Contract

Implemented as a **schema**, in one place, not as a key deletion at the call site.

`selection.py` gains `ValidationPoint.from_dict` — the canonical inverse of `to_dict` and the only
production reader:

- `update`, `distances`, `d_clean` are **state**: required, and validated by the existing constructor
  (so the locked condition grid, the update sign and the no-extra-conditions rule are reused, not
  restated);
- `score` is **derived** and is never restored as independent state;
- a persisted `score` is **validated against the recomputed value** (`math.isclose`, rel 1e-9) rather
  than trusted or silently discarded — the writer always emits it, so a payload disagreeing with its
  own distances describes a point that cannot exist;
- unknown keys are refused.

The **writer** was tightened in the same change, because the escape hatch is what hid the defect:

```python
-        "points": [p.to_dict() if hasattr(p, "to_dict") else dict(p) for p in points],
+        "points": [_canonical_point(p).to_dict() for p in points],
```

`dict(p)` accepted anything dict-like, which let a test persist `[{"update": 0}]` and stay green. A
payload the reader cannot restore is now not writable.

`trainer.py:621` (the reader):

```python
-        result.points = [ValidationPoint(**p) for p in resume.get("points", [])]
+        result.points = [ValidationPoint.from_dict(p) for p in resume.get("points", [])]
```

Repository-wide search confirms **one** production reader; no second incompatible reader exists.
`Candidate.from_dict` was added alongside for repair 4.

## 7. Repair 3 — 20k → 40k Continuation State Machine

`trainer.py` gains two functions and one constant:

- `LEGAL_CAPS = (INITIAL_MAX_UPDATES, EXTENDED_MAX_UPDATES)` — the caps a *persisted* payload may
  claim. The scientific budget rule is untouched; `selection.budget_decision` still owns it.
- `resume_cap(payload)` — the validated cap a resumed run continues under. Refuses a non-integer cap,
  a cap outside the locked pair, a negative `global_update`, `global_update > cap`, and a payload
  claiming the 40k budget at or below 20k updates (a state this machine cannot produce).
- `require_resumable_leg(payload, cap)` — the caller's cap must be the checkpoint's leg or its legal
  successor. **Lowering a cap is always refused.** The only promotion is 20 000 → 40 000 from a
  checkpoint at exactly `INITIAL_MAX_UPDATES`.

`execute.py` now derives the leg instead of defaulting it:

```python
+          carried = load_training_checkpoint(run_checkpoints) if resume else None
+          leg_cap = resume_cap(carried) if carried is not None else INITIAL_MAX_UPDATES
           result = train_run(..., cap=leg_cap, resume=carried)
```

The continuation gate also had to change, and this is the one non-obvious part of the repair:

```python
-          if result.continued:
+          if leg_cap == INITIAL_MAX_UPDATES and result.cap == EXTENDED_MAX_UPDATES:
```

`RunResult.continued` means "this trajectory passed 20k" — the artifact field
`continued_past_initial_budget`. Once a resumed run can *already* be on the 40k leg, that flag is set
at restore time, so the old gate would have re-entered the continuation block after the extended leg
finished. The leg that just ran is the authority instead, which also makes "no 60k/80k extension"
structural: after the extended leg the gate is false, and `require_resumable_leg` refuses any cap
outside the locked pair.

States distinguished: FRESH (0, cap 20k) · INITIAL LEG (0 < gu ≤ 20k, cap 20k) · CONTINUATION LEG
(20k < gu ≤ 40k, cap 40k). `budget_decision`, `resolve_budget` and `select_checkpoint` are unchanged.

`REQUIRED_CHECKPOINT_KEYS` already contains `cap`, so the schema always mandated persisting the value
no reader consumed — evidence this was an oversight rather than a design choice.

## 8. Repair 4 — Stage Artifact Handoff Integrity

New torch-free module `unmark/stage1/artifact.py`:

- `CampaignIdentity` over nine `IDENTITY_FIELDS`: `repository_head`, `protocol_version`,
  `corpus_manifest_digest`, `encoder_checkpoint`, `encoder_revision`, `precision`,
  `inventory_source_name`, `inventory_source_revision`, `inventory_sha256`. Wall-clock, worker count
  and GPU name are deliberately excluded — they are operational, not campaign identity.
- `validate_selection_artifact(...)` — validates stage, protocol, identity, then **recomputes** the
  winner with the production `select_learning_rate` / `select_r` and requires the recorded `selected`
  to equal the recomputation.

Two properties make this sound:

1. **Expected values come from current verified inputs, never from the artifact.** `_campaign_identity`
   in the runner builds them from `resolve_asserted_repository_head()`, the re-hashed
   `verified.chunk_membership_digest`, `args.revision`, and `verify_scientific_inputs().inventory`.
   A self-consistent forgery that rewrites both its identity block and its evidence is still refused.
2. **The selection is rerun, not read.** Because the consumer calls the same functions that produced
   the artifact, the locked-grid rules (exact grid; no missing, duplicate or extra candidate; single
   frozen LR; `r == LR_PILOT_R` in the pilot) are enforced by construction rather than approximated.

`_load_selection` now returns the recomputed winning `Candidate`, not raw JSON, so a caller cannot
regress to `artifact["selected"][...]`. `run_final_main` validates **both** artifacts against the
**same** current identity, so an LR artifact from one campaign and an r artifact from another cannot
be mixed. `execute_stage` writes the `identity` block into every stage artifact; the two legacy
top-level keys are retained.

## 9. Repair 5 — Authoritative Repository Provenance

`checkpoint.py` gains, beside the existing `resolve_repository_head`:

- `EXECUTION_RELEVANT_PATHS = ("unmark", "scripts", "configs", "requirements")`
- `repository_execution_modifications(root)` — tracked modifications only; `??` untracked and `!!`
  ignored entries are filtered out, so `.venv/`, caches, prepared corpora, run outputs and new audit
  files never cause a false failure. `docs/` and `tests/` are outside the pathspec: a modified audit
  note or test does not change the training computation, and a guard that fired on them would be
  switched off within a day.
- `require_clean_execution_tree(root)` — fails closed when tracked execution code differs from HEAD.
- `resolve_asserted_repository_head(asserted, root, require_clean)` — derives the head, and treats a
  supplied `--repository-head` as an **assertion** that must agree.

`stage1_runner.py::_execute` now passes `resolve_asserted_repository_head(args.repository_head)`
instead of `args.repository_head`. `run_smoke` uses the same authority with `require_clean=False`:
smoke is a no-update diagnostic run *while* code is being edited and writes no scientific artifact,
but it still cannot claim a false head.

`execute_stage` carries a structural backstop — `repository_head` must be a full 40-hex sha — placed
**before** the heavy lazy imports, so it is cheap and reachable in the ML-free venv.

The `resolve_repository_head` docstring was corrected in the same change: it claimed there was
"deliberately no CLI flag", which Stage-1 contradicted. It now states the actual contract (no
*override*; an assertion is permitted). This is a documentation edit required for repair 5 to be
internally consistent, not opportunistic cleanup.

Consequence: the Audit 030 §V repository-head resume gate is no longer vacuous. It previously
compared `None` against `None` and passed.

## 10. Cross-Repair Identity / State Composition

Traced end to end:

```
actual Git identity      resolve_asserted_repository_head()  [derived + clean-tree]
  -> verified corpus     verified.chunk_membership_digest    [re-hashed from disk]
  -> stage identity      CampaignIdentity.from_inputs(...)
  -> fresh objective     Stage1Objective(unmark_encoder, provenance.weights)  [per run]
  -> train_run           cap = leg_cap
  -> checkpoint writer   checkpoint_payload(points=_canonical_point(...))
  -> checkpoint reader   ValidationPoint.from_dict + resume_cap + require_resumable_leg
  -> 20k/40k             leg_cap == INITIAL and result.cap == EXTENDED
  -> LR artifact         artifact["identity"]
  -> r artifact          validated against the SAME current identity
  -> final-main          both artifacts, one identity
```

There is **one** concept of repository identity, and it flows from a single source:

| Consumer | Value | Source |
|---|---|---|
| `RunProvenance.repository_head` | derived head | `execute_stage(repository_head=...)` |
| checkpoint `provenance` | same | `provenance.to_dict()` |
| resume matching | same | `require_match` (now never `None`) |
| LR / r / final-main artifacts | same | `identity.repository_head` + legacy top-level key |

`corpus_manifest_digest` likewise has one source (`verified.chunk_membership_digest`) shared by
`RunProvenance` and `CampaignIdentity`. `encoder_revision` is `args.revision`, which `build_backbone`
already refuses unless it equals the locked `ENCODER_REVISION`.

## 11. Real Writer/Reader Checkpoint Evidence

`tests/test_stage1_resume_state_machine.py` (torch-free) and its `_torch` companion.

- Writer-emitted points, through JSON, read back by the production reader — equal to the originals.
- **Mutation check**: `ValidationPoint(**written[0])` still raises `TypeError: ... 'score'`, proving
  the defect was real and that its return would be caught.
- Contradictory persisted score → refused. Missing field / unknown field / short condition grid →
  refused.
- The writer now refuses a payload the reader could not restore.

`tests/test_stage1_training_resume_state.py` was rewritten to carry **real** `ValidationPoint`s
through `checkpoint_payload` → JSON → `ValidationPoint.from_dict` in all five parametrised
interruption cases. Its old body persisted `[{"update": 0}, {"update": 500}]` and asserted they came
back unchanged — a test that exercised the fallback branch and never the reader.

The torch companion drives the **real** `train_run` against a **real** checkpoint written by
`save_training_checkpoint` and read by `load_training_checkpoint`, with the real
`OrthographyInputAdapter`, the real AdamW from `build_optimizer` and a real `DeterministicSampler`
state (which binds a `corpus_digest`, so a fabricated state would be refused).

## 12. 20k / 40k Resume Evidence

Cases A–J, per the repair brief:

| Case | Scenario | Result |
|---|---|---|
| A | initial-leg checkpoint (gu 20 000, cap 20 000) | resumes; history restored; update 0 not re-measured |
| B | boundary checkpoint, best == cap | promotes to `cap=40000`, `continued=True` |
| C | continuation checkpoint (gu 20 500, cap 40 000) | reconstructs `cap=40000` — **the blocker** |
| C′ | same checkpoint offered `cap=20000` | refused (`smaller cap`) — the silent mislabel is impossible |
| D | late continuation (gu 40 000) | `cap=40000`, `continued=True` preserved |
| D′ | best == 40 000 | `budget_limited=True` |
| E | `global_update > cap` | refused through the real reader |
| F | invalid persisted cap (30 000) | refused through the real reader |
| G | malformed / contradictory point | refused through the real reader |
| H | writer-emitted `score` | accepted by the canonical reader |
| I | contradictory persisted `score` | refused |
| J | optimizer step | **none required, and none possible** |

Case J is enforced, not asserted: every case restores at `global_update == cap` so the loop body
cannot execute, `AdamW.step` is monkeypatched to fail the test if called, and the objective's forward
raises.

Also verified: no third continuation is reachable (caps of 60 000 / 80 000 are refused), and adapter,
optimizer, sampler and provenance restore behaviour is unchanged.

## 13. Multi-Run Resume Evidence

`test_three_candidates_resume_independently`: candidate **A** completed the initial leg, **B** crashed
during the 40k continuation, **C** never started.

```
caps == {"A": 20000, "B": 40000, "C": 20000}
carried_c is None            -> C starts fresh
result_a.cap == 20000        -> A is not retrained under B's budget
result_b.cap == 40000, continued=True
result_a.points is not result_b.points
```

Each candidate owns its own checkpoint namespace (`run-<label>/_checkpoint`), so B's continuation cap
cannot reach A or C.

## 14. Artifact Adversarial Evidence

`tests/test_stage1_artifact_identity.py` — 33 tests. Refusals verified for: wrong stage; wrong
protocol; wrong HEAD; wrong corpus digest; wrong backbone checkpoint; wrong backbone revision; wrong
precision; wrong inventory name / revision / sha256; missing identity block (a pre-repair artifact);
truncated identity block; a **self-consistent foreign campaign**; missing candidate; duplicate
candidate; extra candidate; off-grid `r` in the pilot; split LR in the r stage; empty candidate list;
edited selected scalar; edited selected score; edited candidate evidence that changes the winner;
corrupt candidate point; unknown candidate field.

Plus a producer→JSON-on-disk→consumer round trip, and confirmation that `CampaignIdentity.from_inputs`
reads the **real** `InventoryIdentity` dataclass attributes.

## 15. Actual Git HEAD / Provenance Evidence

`tests/test_stage1_repository_provenance.py` — 32 tests, **no Git write anywhere**. Every case is a
read-only query against the real repository or a canned `git` result injected with `monkeypatch`; no
temporary repository is created, nothing is staged, and the working tree is never modified.

Verified: derived head equals `git rev-parse HEAD`; omitted assertion yields the actual head and
never `None`; correct assertion accepted (case-insensitively); false assertion refused; abbreviated
assertion refused; non-repository root fails closed; missing `git` binary fails closed; a branch name
is refused as an identity; untracked and ignored files do **not** make the tree dirty; six porcelain
forms of tracked modification (` M`, `M `, `A `, ` D`, `R `, `UU`) fail closed; the status query
covers exactly `EXECUTION_RELEVANT_PATHS` and excludes `docs`/`tests`; a git failure during the clean
check fails closed; smoke may run dirty but cannot claim a false head; the runner's real `_execute`
passes the **derived** head, not the flag, and refuses a false assertion; and the head gate is no
longer vacuous.

## 16. Frozen-Science Diff Review

Every locked value re-read from `protocol.py` after the repair and unchanged:

```
vinai/phobert-base @ 01daacda68afe13d83023d16ec647239e344a1e6 · fp32 · MAX_LENGTH 256
BATCH_SIZE 128 · EVAL_EVERY_UPDATES 500 · 20000 -> 40000 · PI_STRIP 0.25
CORRUPTION_SEED 35422 · VALIDATION_CORRUPTION_SEED 19225 · SPLIT_SEED 51733
SELECTION_SEED 21230 · TRAIN_SEEDS (36930, 7309, 5993)
LR_PILOT_GRID (1e-4, 3e-4, 1e-3) · LR_PILOT_R 1.0 · R_PHASE1_GRID (0.25, 0.5, 1.0, 2.0, 4.0)
TOTAL_NOMINAL_RUNS 11 · ADAPTER_TRAINABLE_PARAMETERS 3551232 · HIDDEN_SIZE 768
GRADIENT_ACCUMULATION_STEPS 1 · VALIDATION_CONDITIONS ('FULL','P50','P100','STRIP_ALL')
STAGE1_PROTOCOL_VERSION stage1-protocol-v1
```

`selection.py` has **zero removed lines** — the change is purely additive, so `select_checkpoint`,
`budget_decision`, `select_learning_rate`, `select_r` and `descriptive_summary` are untouched. Every
removed line across the four other production files was inspected individually and each is a direct
target of one of the five repairs.

Unchanged: PhoBERT model and revision, hidden size, fp32, frozen-encoder behaviour, adapter
architecture and parameter count, objective mathematics, adapter-init derivation, all seeds,
PI_STRIP, corruption behaviour, sampler order, MAX_LENGTH, overflow and truncation policy, batch
size, gradient accumulation, validation and checkpoint cadence, optimizer definition, weight-decay
grouping, both grids, checkpoint/LR/r selection mathematics, the scientific 20k/40k budget rule, and
official TEST sealing.

## 17. Proposal / Decisions / Freeze Impact

- **Proposal:** unchanged.
- **Decisions:** unchanged. No new decision was created. Replacing an unbound variable, defining the
  inverse of an existing serializer, reading a value the schema already required, validating an
  artifact against current inputs and deriving a HEAD that Stage-6 already derives are all bug fixes
  under existing fail-closed provenance and resume policy, not new scientific choices.
- **Freeze (`docs/spec/stage1-final-freeze.json`):** unchanged. It records
  `generated_for_head = 649ad741...` as deliberately historical implementation metadata and was not
  rewritten to match current documentation HEAD.

The one contract clarification — that `--repository-head` is an assertion rather than a supplied
value — is documented in `resolve_asserted_repository_head`, the `resolve_repository_head` docstring
and the CLI help, which is where an operator will look.

## 18. Focused Test Commands and Results

```
$ .venv/bin/python -B -m pytest -q -p no:cacheprovider -rs \
    tests/test_stage1_name_resolution.py \
    tests/test_stage1_resume_state_machine.py \
    tests/test_stage1_resume_state_machine_torch.py \
    tests/test_stage1_artifact_identity.py \
    tests/test_stage1_repository_provenance.py \
    tests/test_stage1_training_resume_state.py \
    tests/test_stage1_training_resume.py \
    tests/test_stage1_run_independence.py \
    tests/test_stage1_pretrain_audit.py \
    tests/test_stage1_checkpoint.py \
    tests/test_stage1_runner_cli_contract.py \
    tests/test_stage1_final_freeze.py \
    tests/test_stage1_schedule.py \
    tests/test_stage1_corpus_verification.py \
    tests/test_stage1_provenance_contract.py \
    tests/test_stage1_inventory_preflight.py \
    tests/test_stage1_parallel_preparation.py \
    tests/test_stage1_device_contract.py

406 passed, 4 skipped in 22.32s
```

The four skips, all torch-gated and all with stated reasons:

```
test_stage1_resume_state_machine_torch.py:44  the real train_run half needs torch
test_stage1_training_resume.py:40             the tensor half needs torch
test_stage1_name_resolution.py:80             unmark.stage1.objective needs torch
test_stage1_final_freeze.py:270               H0 recomputation needs torch
```

## 19. Full Suite

```
$ .venv/bin/python -B -m pytest -q -p no:cacheprovider
3737 passed, 105 skipped in 130.66s
```

Before this repair the same command reported `3622 passed, 103 skipped`. All 105 skips carry
torch-absence reasons; the ML-free local venv is deliberate policy (`requirements/dev.txt`), and no
skip masks a non-torch code path.

**Eight tests failed on first run after the repair, and every one of them was a test that encoded a
defect.** They are listed here because a reviewer should see them:

| Test | Why it failed | Resolution |
|---|---|---|
| `test_stage1_run_independence.py::test_a_fresh_adapter_is_constructed_inside_the_loop` | required `"objective_cls"` among the loop's calls — the suite **mandated** the `NameError` | now requires `Stage1Objective` |
| `test_stage1_training_resume_state.py::test_points_are_persisted_now` | asserted bare dicts round-trip unchanged | rewritten through the real writer and reader |
| `test_resume_equals_uninterrupted[4,8,12,16,20]` | miniature loop persisted fake point-dicts | now carries real `ValidationPoint`s through the real reader |
| `test_stage1_pretrain_audit.py::test_the_continuation_preserves_optimizer_and_sampler_state` | split source text on the literal `"if result.continued:"` | rewritten as an AST check on the two `train_run` calls |

## 20. Scientific `optimizer.step` Status

**No real scientific optimizer step was executed.**

- `lr-pilot`, `r-phase1` and `final-main` were not invoked.
- The single production step remains `unmark/stage1/trainer.py:677`.
- The new torch-gated resume tests restore at `global_update == cap`, monkeypatch `AdamW.step` to
  fail if called, and use an objective whose forward raises — so a no-update restore test cannot
  train even by accident.
- In the ML-free local venv those tests skip entirely.

## 21. Official UIT-VSFC TEST Status

**SEALED / UNUSED.** Not opened, inspected, read, screened, evaluated, mounted, or passed to any
command; no information derived from it. `manifest.py:221` still fails closed unless
`official_test_used is false`, and no Stage-1 production path constructs a TEST loader. The repair
touched none of that surface, and `test_official_test_has_no_cli_route` still passes.

## 22. Remaining BLOCKER / MAJOR Findings After Repair

**BLOCKER: 0. MAJOR: 0** — for the five material findings, all of which are closed.

Pre-existing MINOR items from Audit 031, deliberately **not** repaired here under the prompt's scope
discipline, and unchanged by this work:

| ID | Item | Note |
|---|---|---|
| `031-M1` | Freeze file records `649ad741...`; HEAD differs | intentionally historical metadata |
| `031-M2` | README stale (claims nothing implemented) | documentation only |
| `031-M3` | `budget_limited` written but absent from `REQUIRED_CHECKPOINT_KEYS`; written pre-`resolve_budget` | not read on resume; `resume_cap` needs only `cap` and `global_update`, both already required |
| `031-M4` | `load_prepared_chunks` routes any non-`train` partition to dev | production path verifies the corpus first |
| `032-I4` | torch deliberately unpinned (Colab-supplied) | accepted policy |

One observation noted during self-review and **not** acted on, because it is pre-existing and
trivial: `manifest = verified.manifest` is unused in `run_r_phase1` and `run_final_main`. It was
already unused before this repair.

Two limitations a reviewer should weigh:

1. **The torch-gated half of the resume evidence did not execute locally.** `test_stage1_resume_state_machine_torch.py`
   is statically checked (name resolution, no unresolved or unused imports) but has never run. It
   must pass in the authorised CUDA environment before this repair is considered verified.
2. **No real Stage-1 run was performed.** Repairs 1–3 are exercised through real production functions
   with a tiny model and a no-update restore; they are not proof that a full 20k→40k campaign
   completes.

## 23. Final Status

All five material findings are closed, tested through real production seams, and composed
consistently. Frozen science is unchanged, the proposal/decisions/freeze are untouched, official TEST
remains sealed, no scientific optimizer step ran, and nothing is staged.

**CONSOLIDATED MATERIAL REPAIR COMPLETE — READY FOR ONE FINAL FULL-REPOSITORY REVIEW**

---

*End of Audit 033.*
