# Audit 034 — Final Post-Repair Full-Repository Review

**Scope:** independent review of the COMPLETE working tree after the consolidated repair (Audit 033).
**Date:** 2026-08-26
**Mode:** REVIEW ONLY. No production file, test, script, config, spec or prior audit was modified.

---

## 1. Exact HEAD

```
$ git rev-parse HEAD
55aa4064780b37626bcae7eef83c504a96fcc51f

$ git log --oneline -3
55aa406 (HEAD -> main, origin/main) final training-launch readiness review
855a1d3 fix bug before training stage-1
479fac5 stage-1 configuration stage
```

## 2. Exact Starting Working Tree

```
 M scripts/stage1_runner.py           ?? docs/audits/031-...md
 M tests/test_stage1_pretrain_audit.py        ?? docs/audits/032-...md
 M tests/test_stage1_run_independence.py      ?? docs/audits/033-...md
 M tests/test_stage1_training_resume_state.py ?? tests/test_stage1_artifact_identity.py
 M unmark/stage1/checkpoint.py                ?? tests/test_stage1_name_resolution.py
 M unmark/stage1/execute.py                   ?? tests/test_stage1_repository_provenance.py
 M unmark/stage1/selection.py                 ?? tests/test_stage1_resume_state_machine.py
 M unmark/stage1/trainer.py                   ?? tests/test_stage1_resume_state_machine_torch.py
                                              ?? unmark/stage1/artifact.py
```

`git diff --cached --name-status` empty (nothing staged). `git diff --check` exit 0.
`docs/audits/034-...md` did **not** exist before this review — no collision.

## 3. Repair Diff Reviewed

542 insertions / 46 deletions across 8 tracked files, plus `unmark/stage1/artifact.py` and five new
test files. Every **removed** line in the four production files was inspected individually; each is a
direct target of one of the five findings. `unmark/stage1/selection.py` has zero removed lines
(purely additive). `protocol.py`, `configs/`, `docs/spec/`, the proposal and the decisions record are
untouched.

## 4. Environment

```
pwd                /mnt/vquclinh/PROJECT-CMAKE/UNMARK-DRAFT/unmark-draft
sys.executable     .../unmark-draft/.venv/bin/python
sys.prefix         .../unmark-draft/.venv
pip                26.2.1 (python 3.14)
torch/transformers ABSENT (deliberate ML-free local venv)
```

Nothing installed, upgraded or modified.

## 5. Full Pipeline Reconstruction

```
UVW shards (pinned: name/bytes/sha256, locked order)
  -> concatenate -> schema + duplicate-id check
  -> contamination screen (official TEST never opened)
  -> document split (SPLIT_SEED 51733, BEFORE chunking)
  -> deterministic pre-chunking (MAX_LENGTH 256, truncation off, on_overflow FAIL)
  -> COMPLETE.json + chunk_membership_digest      [Stage-6, identity = resolve_repository_head()]
  -> verify_prepared_corpus  -> VerifiedCorpus (re-hashed from disk, not declared)
  -> resolve_asserted_repository_head()           [derived HEAD + clean tree]
  -> CampaignIdentity.from_inputs(head, digest, revision, inventory)
  -> [r-phase1/final-main only] validate_selection_artifact(prior artifact)
  -> execute_stage: verify_scientific_inputs -> CUDA/numerical policy -> fingerprint
       -> build_backbone (frozen PhoBERT, loaded ONCE, stage-scope)
       -> per nominal run: fresh_adapter(init_seed) -> UnmarkEncoder -> Stage1Objective
            -> train_run: build_optimizer -> DeterministicSampler -> preparation pool
                 -> objective -> zero_grad -> backward -> gradient_report -> step
                 -> evaluate @500 -> ValidationPoint -> checkpoint_payload -> save
            -> resolve_budget -> budget_decision
       -> [if leg_cap==20k and result.cap==40k] ONE continuation leg
       -> require_frozen_backbone_unchanged
  -> Candidate -> select_learning_rate / select_r
  -> stage artifact (with identity block)
```

Cross-file contracts identified and checked: Stage-6 → Stage-1 (`VerifiedCorpus`), writer ↔ reader
(`ValidationPoint`), caller ↔ leg (`require_resumable_leg`), producer ↔ consumer (`CampaignIdentity`),
runner ↔ `execute_stage` (derived head), `UnmarkEncoder` ↔ encoder (position profile + padding index).

## 6. Five-Way Consistency Matrix (proposal / decisions / freeze / code / tests)

| Item | Locked value | Code | Drift |
|---|---|---|---|
| Model @ revision | `vinai/phobert-base @ 01daacda…` | same | none |
| Precision | fp32 | `PRECISION = fp32` | none |
| Hidden size | 768 | 768 | none |
| Frozen encoder | never trainable, eval | enforced 3× | none |
| Adapter params | 3 551 232 | same | none |
| Objective math | unchanged | untouched | none |
| Seeds | 35422 / 19225 / 51733 / 21230 / (36930, 7309, 5993) | same | none |
| MAX_LENGTH / overflow / truncation | 256 / FAIL / off | same | none |
| Batch / accumulation | 128 / 1 | same | none |
| Cadence | eval 500 = checkpoint 500 | `CHECKPOINT_EVERY_UPDATES is EVAL_EVERY_UPDATES` | none |
| Optimizer | AdamW, wd 0.01/0.0 | untouched | none |
| LR grid / r grid | (1e-4, 3e-4, 1e-3) / (0.25…4.0) | same | none |
| Selection rules | worst-case, d_clean, earliest | zero removed lines | none |
| 20k/40k rule | one continuation, then BUDGET_LIMITED | `budget_decision` untouched | none |
| PI_STRIP | 0.25 | same | none |
| TEST sealing | SEALED | `manifest.py:221` intact | none |

**No scientific drift.**

## 7. B1 — Objective Construction: VERIFIED CLOSED

`execute.py:299` constructs `Stage1Objective(unmark_encoder, provenance.weights)`. `Stage1Objective`
is imported at `execute.py:192` inside the function, so it is a genuine **local** binding —
independently confirmed by bytecode: it appears in `execute_stage.co_varnames`, and `objective_cls`
appears in neither `co_varnames` nor `co_names`.

The signature matches `Stage1Objective.__init__(self, unmark_encoder, weights)` exactly. Construction
remains inside the nominal-run loop, on that run's fresh `UnmarkEncoder`, with that run's
`provenance.weights`. No cross-run mutable state introduced.

`objective_cls` still appears at `execute.py:519` and `scripts/stage1_pretrain_measurements.py:407`.
Both are **legitimate tuple unpackings** of `build_objective(...)`'s third return value, correctly
bound locally. Not defects.

Beyond string matching, a repository-wide `LOAD_GLOBAL`/`LOAD_NAME` resolution sweep over every
`unmark/stage1` module found **zero** unresolved global names (one module, `objective.py`, skips
locally because it imports torch at module scope). This catches the defect *class*, not the instance.

## 8. B2 — ValidationPoint Schema: VERIFIED CLOSED (production), evidence gap — see §20 / 034-MAJ1

Writer language: `to_dict()` → `{update, distances, d_clean, score}`, now routed through
`_canonical_point()` so the old `dict(p)` fallback is gone.
Reader language: `from_dict()` → requires `{update, distances, d_clean}`, tolerates `score`, refuses
unknown keys, delegates grid/sign validation to the constructor.

Writer-emittable set == reader-accepted set. Repository-wide search confirms exactly **one**
production reader (`trainer.py:737`) plus `Candidate.from_dict` → `ValidationPoint.from_dict`
(`selection.py:273`). No second incompatible reader.

**The `rel_tol=1e-9` question, answered independently.** `score` is `max(distances[c] …)` — the
*identity* of one persisted distance, not an arithmetic result — and both persistence paths
(`json.dumps`/`loads`, `torch.save`/`load`) round-trip Python floats exactly. 200 000 randomised
points round-tripped through JSON produced **zero** exact-equality failures. Exact equality is
therefore justified and would be strictly stronger.

It is nevertheless **not** a material defect, because the persisted `score` is *discarded*:
`from_dict` never stores it and `.score` recomputes from `distances`. A corrupt-but-accepted value
(demonstrated at +5e-10 relative) reaches no computation and cannot influence any selection. The
tolerance only slightly widens a corruption tripwire. Classified **INFORMATIONAL** (034-i1).

## 9. B3 — Resume State Machine: VERIFIED CLOSED (production), evidence gap — see §20 / 034-MAJ1

**Writer-emittable states, re-derived from `train_run` rather than from Audit 033 prose.** The write
condition is `global_update % 500 == 0 or global_update == cap`, evaluated *after* `global_update += 1`:

- initial leg (`cap=20000`): writes at 500 … 20 000 — 40 writes;
- continuation leg (`cap=40000`), entered at `global_update == 20000`: writes at **20 500** … 40 000 —
  40 writes.

So **a checkpoint with `cap=40000` and `global_update <= 20000` can never be emitted.** `resume_cap`
refuses exactly that state. The reader's accepted set is therefore identical to the writer's
emittable set — it accepts nothing the writer cannot produce. This is the key soundness property and
it holds.

Boundary sweep executed directly against the production functions:

| gu | cap | `resume_cap` |
|---|---|---|
| 0, 500, 19 500, 20 000 | 20 000 | ACCEPT 20 000 |
| 20 500, 30 000, 39 500, 40 000 | 40 000 | ACCEPT 40 000 |
| 0 / 19 999 / 20 000 | 40 000 | REFUSE "continuation leg begins only after…" |
| 25 000 | 20 000 | REFUSE "cannot have progressed past its own budget" |
| 40 001 | 40 000 | REFUSE |
| −1 | 20 000 | REFUSE negative |
| any | 0 / 30 000 / 60 000 | REFUSE "not one of the locked budgets" |

| persisted → offered | `require_resumable_leg` |
|---|---|
| (20 000, 20 000) → 20 000 | ALLOW (crash resume) |
| (20 000, 20 000) → 40 000 | ALLOW (**the only** promotion) |
| (500, 20 000) → 40 000 | REFUSE "not a legal successor" |
| (0, 20 000) → 40 000 | REFUSE |
| (20 500, 40 000) → 40 000 | ALLOW |
| (20 500, 40 000) → 20 000 | REFUSE "smaller cap" |
| (40 000, 40 000) → 60 000 | REFUSE "not one of the locked budgets" |

**The four required negatives hold.** A resumed 40k leg cannot run under cap 20k (refused); cannot
silently relabel itself 20k (refused, not merely detected); cannot re-enter a second continuation
(the gate is now `leg_cap == INITIAL_MAX_UPDATES and result.cap == EXTENDED_MAX_UPDATES`, and after
the extended leg `leg_cap == 40000` makes it false); and cannot lose
`continued_past_initial_budget` (`result.continued` is set at restore from `cap > INITIAL_MAX_UPDATES`
and `resolve_budget`'s else-branch never clears it).

The gate change from `if result.continued:` is correct and necessary: `continued` means "this
trajectory passed 20k" — which a *continuation resume* also sets — so the old gate would have
re-entered the block after a resumed 40k leg completed. Reviewed independently and confirmed sound.

Ordering note: `execute_stage` calls `resume_cap(carried)` before `train_run`, i.e. before provenance
verification. A foreign checkpoint's `cap` is therefore read before it is known to be ours, but
`train_run` runs `verify_checkpoint` *before* `require_resumable_leg`, so a foreign payload is
rejected on provenance regardless. No exposure.

## 10. Checkpoint Round-Trip Verification

**Torch-free (executes locally, 27 tests):** real `ValidationPoint` → `checkpoint_payload` → JSON →
`ValidationPoint.from_dict`, including a mutation check proving `ValidationPoint(**written[0])` still
raises `TypeError`. `test_stage1_training_resume_state.py` now carries real points through the real
reader in all five parametrised interruption cases. This is genuine writer↔reader evidence and it is
not mocked.

**Torch-gated (`test_stage1_resume_state_machine_torch.py`, 12 tests):** intended to traverse
`save_training_checkpoint` → `load_training_checkpoint` → real `train_run` restore. **It cannot
execute — see 034-MAJ1.** Statically it is otherwise well-formed (all names resolve, no unused
imports, sampler state is built from a real `DeterministicSampler` so the `corpus_digest` binding is
satisfied), and its no-update discipline is sound in design (`AdamW.step` poisoned, forward raises,
every case restores at `global_update == cap`).

**What remains unproven until a corrected torch run:** that `train_run`'s *actual* restore block —
`adapter.load_state_dict(strict=True)`, `optimizer.load_state_dict`,
`require_optimizer_parameter_identity`, `require_optimizer_state_device`,
`DeterministicSampler.from_state`, `ValidationPoint.from_dict` in situ, and `require_resumable_leg`
in situ — executes end to end on a real checkpoint. Every component is covered in isolation; their
composition inside `train_run` is not.

## 11. Multi-Run Resume Verification

The A (completed) / B (crashed mid-40k) / C (not started) scenario is asserted only in the
torch-gated file, so it shares 034-MAJ1 and is currently **unproven by execution**.

By inspection the design is correct: checkpoint namespaces are `output_dir / f"run-{label}" /
"_checkpoint"` — one per run, so no cross-candidate reachability; `carried` is `None` for C, giving
`leg_cap = INITIAL_MAX_UPDATES`; each run builds a fresh adapter, fresh `UnmarkEncoder`, fresh
`Stage1Objective`, fresh optimizer and fresh `RunResult`, so no mutable model/objective/optimizer/
sampler/history state crosses candidates; and `candidates.append(...)` runs exactly once per planned
run, so selection receives exactly one result per candidate.

## 12. B4 — Artifact Handoff: VERIFIED CLOSED

`CampaignIdentity` binds nine fields; `require_match` compares the artifact's block against an
identity built **only** from current inputs (`resolve_asserted_repository_head()`,
`verified.chunk_membership_digest`, `args.revision`, `verify_scientific_inputs().inventory`). No
expected value is read from the document under validation — **no circular validation**. A
self-consistent forgery (identity block *and* evidence rewritten coherently) is still refused, and
this is tested.

`validate_selection_artifact` **recomputes** the winner with the production `select_learning_rate` /
`select_r`, so the locked-grid rules are enforced by construction. All 33 adversarial cases pass:
wrong stage/protocol/HEAD/corpus/backbone/revision/precision/inventory, missing identity block,
truncated identity block, missing/duplicate/extra candidate, off-grid pilot `r`, split LR in the r
stage, edited selected scalar, edited selected score, edited evidence that changes the winner,
corrupt candidate point, unknown candidate field.

`_load_selection` returns a recomputed `Candidate`, not raw JSON, so no caller can regress to
`artifact["selected"][...]`. `run_final_main` validates **both** artifacts against the **same**
current identity and additionally cross-checks that the r artifact's LR equals the LR artifact's
winner — so a consistently-rewritten r artifact is caught there.

**What remains undetectable (asked explicitly).** Editing a *losing* candidate's evidence in a way
that does not change the recomputed winner is accepted — demonstrated: worsening the 1e-3 candidate
from 0.7 to 0.95 validates cleanly. This is the correct boundary between **scientific
identity/selection consistency** (enforced) and **per-datum cryptographic authenticity** (not
enforced, and not required by the current protocol — artifacts are produced and consumed on the
operator's own machine, and no threat model in the proposal or decisions posits a hostile artifact
author). Classified **INFORMATIONAL** (034-i2), not a blocker.

## 13. B5 — Git / Repository Provenance: VERIFIED CLOSED

Actual HEAD is authoritative: `_execute` and `run_smoke` pass
`resolve_asserted_repository_head(...)`, never `args.repository_head`. Omission yields the real head,
never `None`. A false or abbreviated assertion is refused. A branch name is refused as an identity.
Git failure and a missing `git` binary fail closed. `execute_stage` additionally refuses any
`repository_head` that is not a full 40-hex sha, and that guard sits **before** the heavy lazy
imports, so it is reachable (and tested) in the ML-free venv. The Audit 030 §V head gate is no longer
vacuous.

Clean-tree rule: tracked modifications under `unmark/`, `scripts/`, `configs/`, `requirements/` fail
closed across six porcelain forms (` M`, `M `, `A `, ` D`, `R `, `UU`); `??` untracked and `!!`
ignored entries are filtered, so `.venv/`, caches, prepared corpora, run outputs and new audit files
cause no false refusal.

**Path-scope adequacy — checked independently, not assumed.** Tracked files outside the four paths
are: `tests/` (50), `docs/` (43), `results/` (7), `README.md`, `unmark-proposal.{md,pdf}`, a results
zip, `.gitignore`, `pyproject.toml`. None can alter Stage-1 scientific computation:

- `pyproject.toml` contains **only** `[tool.pytest.ini_options]` (`pythonpath`, `testpaths`). The
  repository is not packaged or installed and `scripts/stage1_runner.py` sets its own `sys.path`, so
  this file affects pytest and nothing else. Boundary case → 034-i3.
- `.gitignore` cannot conceal modifications to *tracked* files, so it cannot defeat the guard.
- `tests/`, `docs/`, `results/`, README and proposal do not participate in training.

Critically, the **pinned syllable inventory** — which sets every corruption denominator (D-S1A-008) —
is pinned by `configs/linguistics/vietnamese_syllables.yaml` (tracked, **inside** the guard, carrying
`source_revision` and `sha256`), while the data file itself lives in the untracked
`.resources-cache/`. A pin edit is caught by the clean-tree guard; a cache edit is caught by the
sha256 check in `verify_scientific_inputs`; and `inventory_sha256` is part of `CampaignIdentity`.
Covered from three directions.

**No material provenance gap.**

## 14. Cross-Repair Identity Composition

One concrete identity traced through every stage:

| Consumer | Field | Source |
|---|---|---|
| `RunProvenance.repository_head` | head | `execute_stage(repository_head=…)` ← derived |
| checkpoint `provenance` | head | `provenance.to_dict()` |
| resume matching | head | `require_match` (never `None`) |
| artifact `identity.repository_head` | head | same derived value |
| `RunProvenance.corpus_manifest_digest` | digest | `verified.chunk_membership_digest` |
| `CampaignIdentity.corpus_manifest_digest` | digest | same object |
| backbone / revision / precision | constants + `args.revision` | `build_backbone` refuses a non-locked revision |
| inventory | `verify_scientific_inputs().inventory` | same deterministic pinned source |

The identity used to **validate** an incoming artifact and the identity **written** into the outgoing
artifact are built from the same four sources, so they cannot diverge within a command. No field is
sourced differently at different stages. No circular validation.

## 15. Frozen-Science Review

No scientific choice changed. See §6. `selection.py` purely additive; every production deletion
accounted for; `protocol.py`, `configs/`, `docs/spec/`, proposal and decisions untouched.

## 16. Randomness / Optimizer / Device Review

Intact and unmodified: frozen encoder stays in eval (`UnmarkEncoder.train()` override);
`require_frozen_backbone_unchanged` compares the full `state_dict` and asserts zero trainable
parameters and no gradients; `build_optimizer` refuses any parameter that does not require grad;
`require_optimizer_parameter_identity` (object identity) and `require_optimizer_state_device` run on
fresh construction and again after restore; adapter-only v2 checkpoints with `strict=True`; a fresh
optimizer per nominal run; `DeterministicSampler` deterministic and cursor-exact;
`fresh_adapter` uses `fork_rng(devices=[])` + `torch.default_generator.manual_seed` (not
`torch.manual_seed`); corruption seeded by `CORRUPTION_SEED`; preparation workers are pure functions
of `(sample_id, visit, text)` under `spawn` with order-preserving results; CUDA/numerical policy
(`require_deterministic_cublas_workspace`, `resolve_scientific_device`, `enforce/verify_numerical_policy`)
unchanged. The only `strict=False` occurrences in the changed files are **prose** describing the
retired v1 defect.

## 17. Data / Stage-6 Review

Untouched: UVW pinning, shard order, contamination screen, document split, chunking, manifest,
COMPLETE verification, membership digest. The `checkpoint.py` diff is provenance-only (the new head
assertion and clean-tree helpers); it contains no manifest, shard, split, chunk or digest logic.

## 18. TEST Sealing

**SEALED / UNUSED.** Not opened, inspected, read, screened, evaluated, mounted, or passed to any
command during this review. `manifest.py:221` still fails closed unless `official_test_used is
false`; no Stage-1 production path constructs a TEST loader; no sealing-relevant file was changed by
the repair; `test_official_test_has_no_cli_route` passes.

## 19. Static Regression Hunt

Swept the whole repository for: stale/undefined names (bytecode sweep — none), caller/callee
signature mismatches (every caller of the six changed/new functions enumerated and checked — all
consistent; `execute_stage` has exactly **one** caller), duplicate reader/writer logic (one reader),
mutable defaults (none), unsafe fallbacks (the `dict(p)` fallback was the defect and is removed),
broad exceptions (three pre-existing, all `# noqa: BLE001` and explicitly non-load-bearing —
environment probes and a measurement path), `strict=False` (docstrings only), off-by-one and boundary
errors (§9 sweep), stale cap logic (removed), unknown-key tolerance (now refused in both
`ValidationPoint.from_dict` and `Candidate.from_dict`), comparison semantics (§8), `None` provenance
(impossible), optional identity fields (all nine required by `require_match`), dead branches, double
continuation (§9), stale source-string tests (the one that existed was replaced — see §20),
duplicate scientific constants (none introduced), inconsistent artifact schemas (producer and
consumer verified against each other through JSON on disk), path bugs (checkpoint namespaces
verified run-local).

**One regression found: 034-MAJ1 (§25).** It is in a test, not in production code.

## 20. Test-Quality Assessment

"Could the original bug return while this test stays green?"

| Finding | Primary evidence | Kind | Verdict |
|---|---|---|---|
| B1 | bytecode `LOAD_GLOBAL` resolution over every `unmark/stage1` module + mutation check | **runtime, semantic** | **No** — catches the whole class |
| B2 | real `ValidationPoint` → `checkpoint_payload` → JSON → `from_dict`, + mutation check that `ValidationPoint(**p)` still raises | **real writer↔reader** | **No**, for the schema itself |
| B3 | `resume_cap` / `require_resumable_leg` driven directly across every boundary | **runtime, helper-level** | **No**, for the state machine itself |
| B4 | 33 adversarial cases through the real consumer + producer→disk→consumer round trip | **real consumer** | **No** |
| B5 | real `_execute` with only `execute_stage` stubbed; real repo read-only queries | **real runner-consumer** | **No** |

Three genuine improvements over the pre-repair suite are confirmed: the AST test that *required*
`objective_cls` is gone; the source-string split on `"if result.continued:"` is replaced by an AST
check; and the bare-dict checkpoint test is replaced by a real round trip.

**The false-confidence gap that remains** is B2/B3's *composition* inside `train_run`. The only tests
that were to exercise the real `train_run` restore live in the torch-gated file, and that file cannot
execute (034-MAJ1). Both schema and state machine are individually well tested; their integration
into the actual restore block is currently untested in any environment.

## 21. Tests Actually Run

```
$ .venv/bin/python -B -m pytest -q -p no:cacheprovider -rs <18 focused files>
406 passed, 4 skipped in 23.10s

$ .venv/bin/python -B -m pytest -q -p no:cacheprovider
3737 passed, 105 skipped in 132.54s
```

The four focused skips, exactly as reported:

```
test_stage1_resume_state_machine_torch.py:44  the real train_run half needs torch
test_stage1_training_resume.py:40             the tensor half needs torch
test_stage1_name_resolution.py:80             unmark.stage1.objective needs torch
test_stage1_final_freeze.py:270               H0 recomputation needs torch
```

All 105 full-suite skips carry torch-absence reasons. **Torch skips are not counted as passes**, and
that distinction is exactly what 034-MAJ1 turns on.

## 22. Torch / CUDA Evidence Limitations

Everything torch-dependent is unexecuted locally. Specifically unproven here:

- the real `train_run` restore block end to end (and, because of 034-MAJ1, unproven *anywhere*);
- real-model construction, real adapter parameter count at d=768, real optimizer state device;
- CUDA determinism, fp32 policy, cuBLAS workspace;
- H0 recomputation;
- the 8-worker `spawn` preparation pool under CUDA.

## 23. Historical CUDA Evidence Validity Matrix

| Evidence | Classification | Reason |
|---|---|---|
| Deterministic CUDA / fp32 policy | **STILL VALID** | `device.py` untouched |
| H0 | **STILL VALID** | objective and adapter math untouched |
| Frozen encoder | **STILL VALID** | freeze/verify paths untouched |
| Real model/data no-update probe (smoke) | **PARTIALLY VALID** | unchanged except that `run_smoke` now derives the head; re-run cheaply to confirm the new provenance call |
| Checkpoint / resume | **MUST RE-RUN** | reader, writer and cap reconstruction all changed |
| 20k → 40k continuation | **MUST RE-RUN** | leg derivation and the continuation gate changed |
| Artifact handoff | **MUST RE-RUN** | artifact schema gained `identity`; consumer rewritten |
| Repository-head provenance | **MUST RE-RUN** | new derivation + clean-tree gate on the real CLI |
| Parallel preparation | **STILL VALID** | `preparation.py` untouched |
| Performance benchmark | **STILL VALID** | no change on the hot path; `_canonical_point` is an `isinstance` check on ≤80 points per run, executed at a 500-update cadence |

## 24. Decisions / Change-Log Assessment

Audit 033's "no new decision" judgement is **substantially correct** and reviewed independently. None
of the five items makes the scientific protocol ambiguous or unreproducible, so none blocks training.

| Item | Assessment |
|---|---|
| `--repository-head` is assertion-only | Operator-facing contract change. Aligns Stage-1 with the principle Stage-6 already documents. Worth a change-log line eventually. **MINOR** |
| Clean-tree scope = tracked execution-relevant paths | New **operational** policy with a reviewable path list. Worth recording so the list is revisited deliberately. **MINOR** |
| `CampaignIdentity` field set | Defines what "same campaign" means across stages — the most decision-like of the five. Worth recording. **MINOR** |
| Canonical `ValidationPoint` schema | Internal serialization contract; documented in the code. **INFORMATIONAL** |
| Resume-leg validation | Implements the already-locked budget rule; adds no scientific choice. **INFORMATIONAL** |

Grouped as **034-m1**. Clerical, non-blocking.

## 25. All Findings

### MAJOR

**034-MAJ1 — `tests/test_stage1_resume_state_machine_torch.py` cannot execute; B2/B3 have no real-`train_run` evidence.**

`UnmarkEncoder.__init__` unconditionally calls `resolve_position_profile(encoder)` and
`detect_padding_index(encoder)` — deliberately, to "fail at construction, not deep inside a training
run". The file's `_FrozenStub` exposes only `config.hidden_size`, so:

- `detect_checkpoint` → `None`; `detect_model_family` → `None`; `model_class` → `"_FrozenStub"`;
- the sole registered profile requires `("vinai/phobert-base", "roberta", "RobertaModel")` → no match
  → `UnsupportedPositionSemantics`;
- and independently, `detect_padding_index` finds no `embeddings`, no `word_embeddings.padding_idx`
  and no `config.pad_token_id` → `UnsupportedPositionSemantics`.

Both were verified by replicating the exact `getattr` chains. Therefore `build_objective()` raises,
and **all 12 tests in the file will ERROR** in the CUDA environment — they will not pass, and they
will not be silently skipped.

Consequences:

1. The authoritative CUDA acceptance run will report 12 errors and fail its gate.
2. **B2 and B3 — two BLOCKER repairs — have no regression test that exercises the real `train_run`
   restore path in any environment.** The torch-free tests cover the schema and the state machine
   individually, but not their composition inside `train_run`, which is precisely what the repair
   brief required ("Do not mock away that reader").
3. Audit 033 §11, §12, §13 and §22 assert real-`train_run` and multi-run resume evidence that cannot
   currently exist. Its static check ("name resolution, no unresolved or unused imports") was
   accurate but insufficient: static name binding cannot detect a violated constructor precondition.

Not a defect in production code — the Stage-1 implementation is unaffected. This is an executability
and evidence defect that blocks the imminent authoritative acceptance.

The repository already contains the correct idiom at `tests/test_stage1.py:745-767`:
`config = SimpleNamespace(model_type="roberta", pad_token_id=1, _name_or_path="vinai/phobert-base")`
together with `RobertaLike.__name__ = "RobertaModel"`. **No repair performed in this review.**

### MINOR

**034-m1** — Persistent contract clarifications not yet in the decisions/change log: assertion-only
`--repository-head`, the clean-tree path scope, and the `CampaignIdentity` field set. Non-blocking
(see §24).

**034-m2 (carried, unchanged)** — Pre-existing Audit 031 minors: `031-M1` freeze file records a
historical HEAD; `031-M2` stale README; `031-M3` `budget_limited` written but not in
`REQUIRED_CHECKPOINT_KEYS` (harmless — `resume_cap` needs only `cap` and `global_update`, both
already required); `031-M4` `load_prepared_chunks` routes any non-`train` partition to dev.

**034-m3** — `manifest = verified.manifest` is assigned and unused in `run_r_phase1` and
`run_final_main`. Pre-existing; not introduced by the repair.

### INFORMATIONAL

**034-i1** — `math.isclose(rel_tol=1e-9)` on the derived `score`; exact equality is justified
(proven) and strictly stronger, but the persisted value is discarded on read, so there is no
integrity consequence. See §8.

**034-i2** — Artifact validation enforces selection consistency, not per-datum authenticity: editing
a *losing* candidate's evidence without changing the winner is accepted. Correct for the current
protocol, which posits no hostile artifact author. See §12.

**034-i3** — `pyproject.toml` is tracked but outside the clean-tree path list. Currently pytest-only
configuration, so no material gap; revisit the path list if it ever gains runtime configuration.

**034-i4** — torch deliberately unpinned (Colab-supplied); accepted policy, carried from `032-I4`.

## 26. BLOCKER count: **0**
## 27. MAJOR count: **1** (034-MAJ1)
## 28. MINOR count: **3** (034-m1, 034-m2 carried, 034-m3)
## 29. INFORMATIONAL count: **4** (034-i1 … 034-i4)

## 30. Scientific `optimizer.step` Status

**None executed.** `lr-pilot`, `r-phase1` and `final-main` were not invoked. No test exercising a
real optimizer step ran (all such tests are torch-gated and skipped locally). The single production
step remains `unmark/stage1/trainer.py:677`.

## 31. Official UIT-VSFC TEST Status

**SEALED / UNUSED.** Not opened, inspected, read, screened, evaluated, mounted, or passed to any
command. No information derived from it.

## 32. Final Verdict

The five material findings are genuinely closed **in production code**, and the repair introduced no
scientific, logical, execution, resume, provenance, artifact-handoff, cross-module or TEST-sealing
defect. The five-way consistency review shows no drift, and the identity composition is coherent
end to end.

However, the regression evidence for two of those BLOCKERs does not exist in any executable form, and
the file meant to provide it will error rather than pass in the authoritative environment. Committing
and proceeding to CUDA acceptance now would spend that cycle discovering a broken test.

**FINAL POST-REPAIR FULL-REPOSITORY REVIEW BLOCKED — DO NOT COMMIT OR TRAIN**

---

*End of Audit 034.*
