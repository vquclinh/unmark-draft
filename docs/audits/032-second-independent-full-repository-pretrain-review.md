# Audit 032 — Second Independent Full-Repository Pre-Train Review

**Reviewer role:** SECOND INDEPENDENT FULL-REPOSITORY PRE-TRAIN REVIEWER
**Date:** 2026-08-25
**Repository:** UNMARK — Tone-Factored Input Adaptation for Diacritic-Robust Vietnamese Language Understanding
**Audited HEAD:** `55aa4064780b37626bcae7eef83c504a96fcc51f`

---

## 1. Purpose and Independence Protocol

This audit is a **second, independent** pre-train review of the entire repository, commissioned to
cross-check Audit 031 (`docs/audits/031-final-pretraining-full-repository-review.md`) before Stage-1
training is launched.

The review was run in two strictly ordered phases:

**PHASE 1 — Independent review.** A complete repository-wide forensic review performed *without
reading Audit 031*. Audit 031 was not opened, not grepped, not `cat`-ed, not summarised, and its
finding IDs were not consulted. Its contents were not inferred from Git commit messages. The purpose
was to prevent anchoring: a second reviewer who has read the first reviewer's conclusions is a
proof-reader, not an independent reviewer.

**PHASE 2 — Reconciliation.** Only after the Phase-1 finding list was written to disk and sealed with
an explicit end-of-phase marker (§29) was Audit 031 read, and the two finding sets reconciled.

Independent finding IDs use the `032-` prefix (`032-B*` blocker, `032-MAJ*` major, `032-M*` minor,
`032-I*` informational) and were assigned during Phase 1, before any contact with Audit 031.

**Verifiable independence evidence.** §§1–29 of this file were written to disk in a single write,
before the first read of Audit 031. The Phase-2 sections were appended afterwards. This ordering is
the evidence for the independence claim; see §31 for an honest assessment of its limits.

### 1.1 Constraints observed

- No repository file was repaired, modified, or created except this audit file.
- Audits 030, 031 and all earlier audits were left byte-identical.
- No `git add`, `commit`, `push`, `tag`, `stash`, `reset`, `checkout`, `restore`, or `clean` was run.
  Nothing was staged.
- Every Python and test command used `.venv/bin/python`. No package was installed, upgraded or
  removed; `.venv` and `.env` were not modified.
- No real Stage-1 training was run. `lr-pilot`, `r-phase1` and `final-main` were not invoked. No
  scientific `optimizer.step` was executed (§35).
- Official UIT-VSFC TEST was not opened, inspected, read, screened, evaluated, mounted, or passed to
  any command. It remains SEALED / UNUSED (§36).

---

## 2. HEAD Under Review

```
55aa4064780b37626bcae7eef83c504a96fcc51f
```

Recent history:

| Commit | Subject |
|---|---|
| `55aa406` | final training-launch readiness review |
| `855a1d3` | fix bug before training stage-1 |
| `479fac5` | stage-1 configuration stage |
| `649ad74` | fix the persistent cuda |
| `a84cf7e` | local tests |

**Finding of note.** Despite the subject line "fix bug before training stage-1", a diff of production
code between `479fac5` and HEAD shows **no change to any file under `unmark/` or `scripts/`**. The
commits after `479fac5` are documentation and audit material. Any reader who assumes from the commit
message that a code defect was fixed before training would be mistaken. This is recorded as `032-I3`.

---

## 3. Git Working-Tree State

```
$ git status --short
?? docs/audits/031-final-pretraining-full-repository-review.md
```

- Exactly one entry: Audit 031, **untracked**.
- Nothing staged (`git diff --cached` empty).
- No tracked file modified.
- `git diff --check` clean — no whitespace or conflict-marker damage.

The tree is therefore clean with respect to all tracked production code. Note that Audit 031 — the
document this review cross-checks — is itself **not committed** at the audited HEAD.

---

## 4. Virtual-Environment Verification

```
.venv/bin/python -c "import sys; print(sys.prefix)"
→ /mnt/vquclinh/PROJECT-CMAKE/UNMARK-DRAFT/unmark-draft/.venv
Python 3.14 · pip 26.2.1
```

Installed scientific stack:

| Package | Local `.venv` |
|---|---|
| torch | **ABSENT** |
| transformers | **ABSENT** |
| tokenizers | **ABSENT** |
| numpy | **ABSENT** |

This is **deliberate and documented**, not a defect. `requirements/dev.txt` states it "deliberately
excludes torch, transformers…", and the skip messages read `torch is not installed (ML-free local
.venv); runs on Colab`. The consequence for evidence-weighting is recorded as `032-I1` and discussed
in §26: a green local suite is **not** evidence that torch/CUDA paths are correct.

---

## 5. Architecture Overview

Stage-1 trains a small **orthography input adapter** (3 551 232 parameters) in front of a **frozen**
PhoBERT-base encoder (`vinai/phobert-base @ 01daacda68afe13d83023d16ec647239e344a1e6`), fp32.

Layering:

| Layer | Module | Role |
|---|---|---|
| CLI | `scripts/stage1_runner.py` | operator entry; sub-commands `prepare-corpus`, `lr-pilot`, `r-phase1`, `final-main` |
| Orchestration | `unmark/stage1/execute.py` | preflight, device, backbone, preparation pool, provenance, `train_run` |
| Training | `unmark/stage1/trainer.py` | update loop, evaluation, checkpointing, resume |
| Selection | `unmark/stage1/selection.py` | `ValidationPoint`, `select_checkpoint`, `budget_decision` |
| Persistence | `unmark/stage1/checkpoint.py` | shard/checkpoint IO, `resolve_repository_head` |
| Model | `unmark/modeling/adapter.py` | `OrthographyInputAdapter`, `UnmarkEncoder` |
| Protocol | `unmark/stage1/protocol.py` | locked scientific constants |

---

## 6. Five-Way Consistency Matrix

Locked constants checked across protocol code, the freeze spec, the decisions record, the audit
chain, and the tests.

| Constant | Value | Consistent? |
|---|---|---|
| Backbone | `vinai/phobert-base @ 01daacda…` | ✅ |
| Precision | fp32 | ✅ |
| MAX_LENGTH / ON_OVERFLOW / truncation | 256 / FAIL / off | ✅ |
| Batch size | 128 | ✅ |
| Eval + checkpoint interval | every 500 updates | ✅ |
| Budget | 20 000 → 40 000 | ✅ |
| PI_STRIP | 0.25 | ✅ |
| Corruption seed | 35422 | ✅ |
| Validation seed | 19225 | ✅ |
| Split seed | 51733 | ✅ |
| SELECTION_SEED | 21230 | ✅ |
| TRAIN_SEEDS | 36930, 7309, 5993 | ✅ |
| Adapter parameters | 3 551 232 | ✅ |
| LR grid | 1e-4, 3e-4, 1e-3 | ✅ |
| r grid | 0.25, 0.5, 1.0, 2.0, 4.0 | ✅ |
| Nominal runs | 11 | ✅ |
| AdamW | betas (0.9, 0.999), eps 1e-08, amsgrad False | ✅ |
| Weight decay | 0.01 / 0.0 | ✅ |
| VALIDATION_CONDITIONS | `("FULL","P50","P100","STRIP_ALL")` | ✅ |

No numeric drift was found between the five sources. The constants are genuinely locked.

---

## 7. CLI-to-Optimizer Trace

Verified by reading `execute.py` and confirming call order against the AST:

```
verify_scientific_inputs(201)
  → require_deterministic_cublas_workspace(210)
  → resolve_scientific_device(211)
  → enforce/verify_numerical_policy(212/213)
  → current_fingerprint(214)
  → load_prepared_chunks(224)
  → build_backbone(229) → .to(230) → module_state_hash(231)
  → prepare_condition_batch(242)
  → worker_config(261) → PreparationPool(274) → loop(275)
  → RunProvenance(280) → adapter_init_seed(282)
  → fresh_adapter(295) → .to(297) → UnmarkEncoder(298) → objective_cls(299)
  → train_run(325)
  → require_frozen_backbone_unchanged(374)
```

The gate ordering is correct: every scientific-input and device gate fires **before** the backbone is
built, and the frozen-backbone hash is re-verified **after** training.

Training-loop order inside `train_run` (AST-verified, not inferred from prose):

```
next_batch(636) → prepare(644) → batch_to_device(662) → collate(663)
  → objective(665/666) → zero_grad(667/668) → backward(668/669)
  → gradient_report(669/670) → step(677) → global_update += 1(678)
  → evaluate_fn(681) → objective.train(True)(682) → save_training_checkpoint(696)
```

This is a correct and complete update cycle. `zero_grad` occurring after the forward pass rather than
before it is unusual in style but **not a defect**: it is called before `backward`, so no gradient
accumulation across updates occurs.

---

## 8. Static Bug Review

The two blockers in §28 were found here. Additionally checked and found **sound**:

- `ShardRecord.from_dict` (`checkpoint.py:402`) → `cls(**raw)` with `to_dict() = dict(self.__dict__)`
  — round-trips safely; **not** an instance of the §9 defect class.
- `select_checkpoint` correctly refuses a missing update-0 point and duplicate updates.
- `UnmarkEncoder.train()` correctly re-asserts `self.encoder.eval()` after `super().train(mode)`.

---

## 9. Checkpoint / Resume Review — `032-B1` (BLOCKER)

`trainer.py` writes validation points through `to_dict()` and reads them back through the
**constructor**. Those two are not inverse operations.

Writer — `trainer.py:324`:
```python
"points": [p.to_dict() if hasattr(p, "to_dict") else dict(p) for p in points],
```

Reader — `trainer.py:621`:
```python
result.points = [ValidationPoint(**p) for p in resume.get("points", [])]
```

`ValidationPoint.score` is a derived `@property`:
```python
@property
def score(self) -> float:
    return max(self.distances[c] for c in VALIDATION_CONDITIONS)
```
and `to_dict()` serialises it. The constructor does not accept it.

Executed probe (`.venv/bin/python`):
```
ctor fields  : ['update', 'distances', 'd_clean']
to_dict keys : ['d_clean', 'distances', 'score', 'update']
ValidationPoint(**to_dict()) -> TypeError: ValidationPoint.__init__() got an
                                unexpected keyword argument 'score'
```

**Impact.** Every `--resume` that carries at least one validation point raises `TypeError` at
`trainer.py:621`. A 20 000-update leg produces 40 points, so the mandatory 20 000 → 40 000
continuation **cannot start**. This also breaks crash recovery for any run that has passed update 500.

This is the same defect class Audit 030 §V established for `RunProvenance`: **`to_dict()` is artifact
serialisation, not a constructor round-trip.** The lesson was recorded but this second site was not
remediated.

**Classification:** BLOCKER — `scientific_protocol` (the 40k continuation is protocol-mandatory).

---

## 10. 20k / 40k State Machine — `032-B2` (BLOCKER, currently masked by B1)

The budget cap is persisted but never restored.

Writer — `trainer.py:322`: `"cap": cap,`
Reader — **none.** `grep` finds no read of `resume["cap"]` anywhere.

Instead `execute.py:334` passes the cap unconditionally, on resume as well as on a fresh run:
```python
cap=INITIAL_MAX_UPDATES,
```
and `trainer.py:606/622/630` consume it:
```python
result = RunResult(provenance=provenance, cap=cap)
result.continued = cap > INITIAL_MAX_UPDATES
while global_update < cap:
```

Executed probe:
```
INITIAL_MAX_UPDATES = 20000 | EXTENDED_MAX_UPDATES = 40000
budget_decision(28000, 20000) -> SelectionViolation: selected update 28000 exceeds the cap 20000
budget_decision(28000, 40000) -> BudgetDecision(cap=40000, continue_run=False, …)
budget_decision(19500, 20000) -> BudgetDecision(cap=20000, continue_run=False, …)
```

**Impact.** Two distinct failure modes on the 40 000-update leg:

1. If the best checkpoint lies **beyond** 20 000, `budget_decision` raises `SelectionViolation:
   selected update 28000 exceeds the cap 20000` at `trainer.py:538` — the extended run dies at
   selection time, after paying the full compute cost.
2. If the best checkpoint lies **at or below** 20 000, the run **silently records `cap=20000` and
   `continued=False`** — a 40 000-update run is written to the artifact as if it were a complete
   20 000-update run. This is the more dangerous mode: it is a **silent provenance falsification**,
   not a crash.

Additionally, `while global_update < cap` with a restored `global_update` of 20 000 and a cap of
20 000 means the loop **exits immediately** — a resumed continuation would perform zero updates.

**Currently masked by `032-B1`:** the `TypeError` at line 621 fires before the cap is used, so B2 is
unobservable until B1 is fixed. Fixing B1 alone would expose B2. **These must be fixed together.**

**Classification:** BLOCKER — `scientific_protocol` + `operational_provenance`.

---

## 11. Nominal-Run Independence

D-S1B-017 requires each nominal run to be independent. Verified sound:

- `adapter_init_seed(run_seed) = derive_seeds(f"UNMARK-STAGE1-v1|adapter-init|{run_seed}", 1)[0]`
- Mapping confirmed: 21230→3203, 36930→51800, 7309→45833, 5993→15758.
- Four distinct hash groups with multiplicities `[8, 1, 1, 1]`.
- Adapter-only checkpoint v2; no cross-candidate state carried.

The cross-candidate leakage defect found in Audit 030 §AD is remediated at this HEAD. No new leakage
path was found.

---

## 12. Frozen Encoder

- `build_backbone` → `module_state_hash` captured at `execute.py:231` **before** training.
- `require_frozen_backbone_unchanged` re-checked at `execute.py:374` **after** training.
- Hash covers the **full `state_dict`**, not only parameters (Audit 030 §AD.5 correction) — so buffers
  and running statistics are in scope.
- `UnmarkEncoder.train()` re-asserts `self.encoder.eval()`.

**Sound.** No finding.

---

## 13. Optimizer

AdamW, betas (0.9, 0.999), eps 1e-08, amsgrad False, weight decay 0.01 / 0.0, adapter parameters
only. Verified the optimizer is constructed over adapter parameters exclusively — the frozen backbone
is not in any parameter group.

**Sound.** No finding.

---

## 14. Randomness

The only executable RNG on the scientific path is in `initialisation.py`:

```python
with torch.random.fork_rng(devices=[]):
    torch.default_generator.manual_seed(int(init_seed))
    adapter = OrthographyInputAdapter(AdapterConfig(hidden_size=hidden_size))
```

This is correct, and correct for a subtle reason recorded in Audit 030 §AE: `torch.manual_seed()`
would seed **all** devices, while `fork_rng(devices=[])` only snapshots and restores **CPU** state —
so `torch.default_generator.manual_seed` is the matching call. Using `torch.manual_seed` here would
leak a CUDA-RNG mutation past the context manager.

Preparation uses `spawn`, never `fork`, because the parent holds a CUDA context before pool creation.

**Sound.** No finding.

---

## 15. CUDA Contract

`require_deterministic_cublas_workspace` and `enforce/verify_numerical_policy` fire at
`execute.py:210/212/213`, before backbone construction (D-S1B-015). Device resolution is centralised
in `resolve_scientific_device`. The harness device bug from Audit 030 §AI is fixed at this HEAD.

**Not locally verifiable** — torch is absent from `.venv` (§4). This axis rests on the Colab gate
evidence recorded in Audit 030 §AE.11, not on anything this review executed.

---

## 16. Data / Stage-6

Corpus manifest digest is bound into provenance and re-verified. `manifest.py:221` fails closed:

```python
if manifest.get("official_test_used") is not False:
    raise ManifestViolation("manifest does not record official_test_used=false")
```

**Sound.** No finding.

---

## 17. Parallel Preparation

`spawn` start method, 8 workers, deterministic per-item seeding. The Audit 030 §AG repair (7.70×
preparation speed-up; recurring pre-step cost 5.975526 → 1.965580 s/update) is present at this HEAD.
Determinism is preserved because each work item derives its own seed rather than drawing from a
shared stream.

**Sound.** No finding.

---

## 18. Artifact Handoff — `032-MAJ1` (MAJOR)

`scripts/stage1_runner.py::_load_selection` is the sole gate between stages:

```python
def _load_selection(path: Path, expected_stage: str) -> dict:
    if not path.is_file(): raise Stage1ContractViolation(...)
    artifact = json.loads(path.read_text(encoding="utf-8"))
    if artifact.get("stage") != expected_stage: raise ...
    if artifact.get("protocol_version") != STAGE1_PROTOCOL_VERSION: raise ...
    return artifact
```

It validates **only** `stage` and `protocol_version`. The artifact also carries
`repository_head`, `corpus_manifest_digest`, `candidates`, `preparation`, `raw_text_persisted`,
`official_test_used` and `downstream_score_used` — **none of which are checked**.

**Impact.** An `lr-pilot` artifact produced from a *different commit* or a *different corpus* is
accepted by `r-phase1` / `final-main` without complaint, so long as the stage string and protocol
version match. The scientific chain of custody between stages is therefore unenforced at exactly the
point where it is supposed to be enforced.

**Partial mitigation:** `run_final_main` does cross-check LR consistency between the two artifacts, so
the most obvious inconsistency is caught. That mitigation does not cover commit or corpus identity.

**Not a blocker** because a careful operator running the stages in sequence on one machine will
produce consistent artifacts; this is a missing guard, not an active corruption.

**Classification:** MAJOR — `operational_provenance`.

---

## 19. Git / Provenance — `032-MAJ2` (MAJOR)

Two repository-identity mechanisms coexist and contradict each other.

**Stage-6** derives identity from Git. `checkpoint.py::resolve_repository_head` runs
`git rev-parse HEAD` and fails closed, and its docstring is emphatic about *why*:

> "Checkpoint identity must record the commit that produced the shards, so this is derived from the
> repository rather than accepted from the caller. There is deliberately **no CLI flag and no
> environment override**: a caller-claimed HEAD would let a checkpoint written by commit A resume
> under commit B while asserting it did not."

**Stage-1 training does exactly the thing that docstring forbids.** `stage1_runner.py:513/531` pass
`repository_head=args.repository_head` — an operator-supplied `--repository-head` flag (declared at
513/531/579/633) that **defaults to `None` and is never verified against Git**.

Executed probe:
```
provenance.repository_head when --repository-head omitted: None
require_match(None vs None): PASSES  -> the §V repository-head resume gate is a no-op
```

**Impact.**
1. If the flag is omitted, provenance records **no repository identity at all**, and the
   resume-blocking head check that Audit 030 §V introduced degrades to `None == None` — it passes
   vacuously. The gate exists but does not gate.
2. If the flag is supplied incorrectly, the artifact asserts a commit identity that is simply false,
   with nothing able to detect it.
3. `grep` for `dirty|is_clean|status --porcelain|require_clean` across `unmark/stage1/*.py` and
   `scripts/stage1_runner.py` returns **no matches** — there is no dirty-tree detection anywhere on
   the Stage-1 path, so training from a modified working tree records a clean commit SHA.

**Not a blocker** because it does not corrupt the science of a single correctly-run experiment; it
degrades the *provenance record* of that experiment. But it is a real cross-file contract violation
(§24), and the mitigation is one line: default `--repository-head` to `resolve_repository_head()`.

**Classification:** MAJOR — `operational_provenance`.

---

## 20. CLI Operator Safety

Destructive filesystem operations on the Stage-1 path:

| Site | Operation | Assessment |
|---|---|---|
| `checkpoint.py:201` | `candidate.unlink()` | scoped to stale checkpoint candidates — safe |
| `checkpoint.py:700` | `shutil.rmtree(self._staging, ignore_errors=True)` | scoped to a staging dir owned by the writer — safe |

No unscoped deletion, no user-path-controlled `rmtree`, no recursive delete of an output root.

**Sound.** No finding.

---

## 21. Output / Recovery Behaviour

`stage1_runner.py` guards output collisions:

```
128: if output.exists() and not (checkpoint_root / "state.json").is_file() …
496: if output.exists() and not resuming:
501: if resuming and not output.exists():
```

The logic is symmetric and correct: refuse to overwrite an existing output unless resuming, and
refuse to resume when no output exists. This is a genuinely well-built guard.

**Caveat:** it protects the *artifact*, and is therefore intact; but it cannot help with `032-B1`,
which fails later, inside `train_run`, after these checks pass.

**Sound.** No finding.

---

## 22. Official TEST Sealing

- `manifest.py:221` raises `ManifestViolation` unless `official_test_used is false`.
- `official_test_used: False` / `downstream_score_used: False` are written at
  `preg1_split.py:427/428`, `manifest.py:68`, `preg1_head_diagnostic.py:221`.
- No Stage-1 code path constructs a UIT-VSFC TEST loader. The only `vsfc` matches under `unmark/` are
  in `unmark/evaluation/preg1_*` (a later stage) and are docstrings/label definitions.
- This review did not open, read, mount, or pass TEST to any command.

**Status: SEALED / UNUSED.** See §36.

---

## 23. Test-Suite Quality — `032-M1` (MINOR)

The suite is green and the resume path is broken. The reason is a single test:

`tests/test_stage1_training_resume_state.py`
```python
def test_points_are_persisted_now():
    payload = checkpoint_payload(..., points=[{"update": 0}, {"update": 500}])
    assert payload["points"] == [{"update": 0}, {"update": 500}]
```

Passing **plain dicts** takes the `dict(p)` branch of the writer, so `score` never appears — and the
reader at `trainer.py:621` is **never invoked at all**. The test asserts that a list of dicts survives
being copied into a list of dicts.

This file is torch-free: it reports **27 passed, 0 skipped** locally. So `032-B1` was reachable by the
local suite and was missed by test design, not by environment.

This is the **recurring project defect class** — a test that exercises the seam it names without
crossing it (the prose-matching tests of Audit 030 §§X, Y, AE are the same shape). Recording it as
MINOR rather than MAJOR because it is a test gap, not a production defect; the production defect it
failed to catch is already counted as `032-B1`.

**Classification:** MINOR — test coverage.

---

## 24. Cross-File Conflicts

| # | Conflict | Ref |
|---|---|---|
| 1 | `checkpoint.py::resolve_repository_head` docstring states there is "deliberately no CLI flag and no environment override" for repository head; `stage1_runner.py` provides exactly such a flag for Stage-1 training | `032-MAJ2` |
| 2 | `trainer.py:322` persists `cap`; no reader exists; `execute.py:334` supplies a hardcoded cap instead | `032-B2` |
| 3 | `trainer.py:324` serialises via `to_dict()`; `trainer.py:621` deserialises via the constructor | `032-B1` |
| 4 | Commit `855a1d3` is titled "fix bug before training stage-1" but changes no production code | `032-I3` |

Conflict 1 is the most notable: the repository **states the correct principle in its own source** and
then violates it one module away.

---

## 25. Requirements

`requirements/` contains `base.txt`, `dev.txt`, `experiment.txt`. There is no root `requirements.txt`;
`pyproject.toml` declares no `dependencies` block. The split is intentional:

- `dev.txt` — "Deliberately excludes torch, transformers…"
- `experiment.txt` — pins `transformers==4.57.6`; torch is **deliberately not pinned** because "PyTorch
  already provided by the Colab runtime is reused."

**Assessment: acceptable, with one observation.** Reusing Colab's torch means the torch version is
**not** part of the frozen configuration — it is whatever Colab ships on the day. For a study whose
numerical policy is this tightly specified (§15), the torch version is a scientific input that is not
pinned. I am recording this as informational (`032-I4`) rather than as a finding, because the project
has explicitly chosen this trade-off and the transformers pin — which controls the serialisation
format of the pinned backbone revision — *is* locked.

---

## 26. Historical Evidence Validity

The audit chain (Audit 030 §§T–AK) is internally consistent and its factual claims spot-checked
correctly where I re-derived them independently (the init-seed mapping, the four hash groups with
multiplicities `[8,1,1,1]`, the loop order, the full-`state_dict` hash).

**One weighting caveat, stated plainly.** Because `.venv` is ML-free (§4), *this* review executed no
torch code. Every claim in this audit about CUDA, tensor movement, dtype, or the frozen-backbone hash
at runtime is a **static** claim, and inherits its runtime confidence from the Colab gate evidence in
Audit 030 §AE.11 — not from anything I ran. A reader should not read this audit's green test line as
CUDA validation. That distinction was already drawn correctly by the project in Audit 030 §AB, which
declined to call a device-silent runner smoke a CUDA smoke.

---

## 27. Tests Run

All via `.venv/bin/python -m pytest -q -p no:cacheprovider`.

| Scope | Result |
|---|---|
| Full suite | **3622 passed, 103 skipped** in 134.71 s |
| Targeted Stage-1 suites (11 files) | **265 passed, 1 skipped** in 22.80 s |
| `test_stage1_training_resume_state.py` | 27 passed, **0 skipped** |
| `test_stage1_device_contract.py` | 10 passed, 0 skipped |

All 103 skips carry torch-absence reasons (`torch is not installed (ML-free local .venv); runs on
Colab`). No skip masks a non-torch code path.

**The central observation of this review:** the suite is **fully green at a HEAD where every
`--resume` raises `TypeError`.** Green is not, here, evidence of readiness.

---

## 28. Independent Findings (Phase 1)

| ID | Severity | Finding | Site |
|---|---|---|---|
| `032-B1` | **BLOCKER** | `ValidationPoint(**p)` raises `TypeError: unexpected keyword argument 'score'` because `to_dict()` emits the derived `score`. Breaks every `--resume` and the protocol-mandatory 20k→40k continuation. | `trainer.py:324` ↔ `trainer.py:621` |
| `032-B2` | **BLOCKER** (masked by B1) | Persisted `cap` is never read; `cap=INITIAL_MAX_UPDATES` is passed unconditionally. The 40k leg either raises `SelectionViolation` or **silently records a 40k run as a complete 20k run**. | `trainer.py:322`, `execute.py:334` |
| `032-MAJ1` | MAJOR | `_load_selection` validates only `stage` + `protocol_version`; `repository_head` and `corpus_manifest_digest` are unchecked across the stage handoff. | `scripts/stage1_runner.py` |
| `032-MAJ2` | MAJOR | `--repository-head` is operator-supplied, defaults to `None`, is never verified against Git, and no dirty-tree check exists — contradicting `resolve_repository_head`'s own stated contract and reducing §V's head gate to a no-op. | `stage1_runner.py:513/531` vs `checkpoint.py:353` |
| `032-M1` | MINOR | `test_points_are_persisted_now` passes plain dicts, exercising only the `dict(p)` branch and never invoking the reader — the suite stays green while resume is broken. | `tests/test_stage1_training_resume_state.py` |
| `032-I1` | INFO | Local `.venv` is deliberately ML-free; 103 torch-gated skips; local green ≠ CUDA-validated. | `requirements/dev.txt` |
| `032-I2` | INFO | TEST sealing is enforced fail-closed; no Stage-1 path reaches UIT-VSFC TEST. | `manifest.py:221` |
| `032-I3` | INFO | Commit `855a1d3` "fix bug before training stage-1" changes no production code. | git history |
| `032-I4` | INFO | torch version is deliberately unpinned (Colab-provided); transformers is pinned at 4.57.6. | `requirements/experiment.txt` |

**Blocker count: 2. Major count: 2.** Both blockers sit on the resume/continuation path. Neither
affects a fresh run that completes 20 000 updates without interruption — which is precisely why the
suite and the prior smoke tests did not surface them.

---

## 29. END OF PHASE 1 — INDEPENDENT REVIEW SEALED

**Everything above this line was written before Audit 031 was read.**

Audit 031 had not been opened, grepped, `cat`-ed, or summarised at the time §§1–28 were committed to
disk. The finding IDs `032-B1`, `032-B2`, `032-MAJ1`, `032-MAJ2`, `032-M1`, `032-I1`–`032-I4` were
assigned independently.

Phase 2 begins below.

---

# PHASE 2 — RECONCILIATION WITH AUDIT 031

## 30. Audit 031 — First Read

Audit 031 (`docs/audits/031-final-pretraining-full-repository-review.md`, 1311 lines, untracked) was
read for the first time after §29 was sealed to disk.

Its verdict:

> FINAL FULL-REPOSITORY PRE-TRAIN REVIEW BLOCKED -- MATERIAL ISSUE(S) REQUIRE HUMAN DECISION BEFORE TRAINING

Its counts: **BLOCKER 5, MAJOR 0, MINOR 5, INFORMATIONAL 4.**

| ID | Severity | Subject |
|---|---|---|
| `031-B1` | BLOCKER | `execute_stage()` cannot construct the objective — `objective_cls` unbound |
| `031-B2` | BLOCKER | Validation-point writer/reader schema disagreement |
| `031-B3` | BLOCKER | 20k→40k continuation cap not reconstructible after process death |
| `031-B4` | BLOCKER | Downstream stage handoff artifacts under-validated |
| `031-B5` | BLOCKER | Stage-1 repository HEAD provenance optional and caller-supplied |
| `031-M1`–`M5` | MINOR | freeze metadata stale; README stale; `budget_limited` schema gap; `load_prepared_chunks` partition routing; tests mask failures |
| `031-I1`–`I4` | INFO | ML-free `.venv`; no optimizer step; Stage-1/6 parallelism roles; TEST sealed |

Both audits were performed at the same HEAD, `55aa4064780b37626bcae7eef83c504a96fcc51f`.

---

## 31. Independence Assessment — Stated Honestly

**What is genuinely independent.** §§1–29 were written to disk before Audit 031 was opened. Two
separate reviews, working from source, independently identified the same three core defect sites
(`032-B1`≡`031-B2`, `032-B2`≡`031-B3`, `032-MAJ1`≡`031-B4`, `032-MAJ2`≡`031-B5`). Convergent discovery
by two passes that did not communicate is meaningful evidence that these defects are real and not
artifacts of one reviewer's framing.

**Three qualifications, which materially limit the independence claim:**

1. **I was not a cold reviewer.** I carried extensive prior context from Audit 030 §§V–AK, including
   §V's conclusion that *`to_dict()` is artifact serialisation, not a constructor round-trip*. That
   lesson pointed almost directly at `032-B1`. My agreement with `031-B2` is therefore **less
   independent than it appears** — I was primed to look at exactly that seam. A reviewer without that
   history might not have found it. I flag this rather than claim credit for convergence.

2. **I missed the most severe finding.** `031-B1` — `objective_cls` unbound in `execute_stage()` —
   blocks *every* Stage-1 run, not merely resume. I did not find it. My §7 trace **recorded the call
   `objective_cls(299)` and transcribed it without checking whether the name was bound.** I verified
   call *order* by AST and never verified name *resolution*. That is a real methodological gap, and it
   means my Phase-1 picture — "the blockers are on the resume path; a fresh 20k run is fine" — was
   **wrong**. A fresh run cannot start either.

3. **I under-classified two findings.** I rated the handoff and provenance gaps MAJOR; Audit 031 rates
   them BLOCKER. See §34.

**Conclusion:** this review is a useful independent cross-check that *confirms* four of Audit 031's
five blockers from source, and adds detail to two of them. It is **not** a review that would have
been sufficient on its own — Audit 031 is the stronger of the two documents, and on the single most
important finding it is right and I was silent.

---

## 32. Reconciliation Matrix

| 032 ID | 031 ID | Status | Note |
|---|---|---|---|
| — | `031-B1` | **NOT FOUND BY 032** | Missed by me; independently re-verified as genuine (§33) |
| `032-B1` | `031-B2` | **AGREE** | Same sites, same `TypeError`, same impact. Independently reproduced by both |
| `032-B2` | `031-B3` | **AGREE** | Same root cause; each adds a distinct failure mode (§34) |
| `032-MAJ1` | `031-B4` | **PARTIAL** | Agree on substance and site; **disagree on severity** (MAJOR vs BLOCKER) |
| `032-MAJ2` | `031-B5` | **PARTIAL** | Agree on substance and site; **disagree on severity**; 032 adds the docstring self-contradiction and the absent dirty-tree check |
| `032-M1` | `031-M5` | **AGREE** | Same test, same diagnosis |
| `032-I1` | `031-I1` | **AGREE** | ML-free `.venv` is deliberate policy |
| `032-I2` | `031-I4` | **AGREE** | TEST sealed / unused |
| `032-I3` | `031` §3 + `031-M1` | **AGREE** | 031 went further: proved byte-identity of scientific code across `479fac5`→`855a1d3`→`55aa406` by diff |
| `032-I4` | `031` §27 | **PARTIAL** | Same facts on unpinned torch; I note it as an unpinned scientific input, 031 accepts it as settled policy |
| — | `031-M1` | **NOT FOUND BY 032** | Freeze file records `649ad741…` vs HEAD `55aa4064…` — I observed the fact, did not classify it |
| — | `031-M2` | **NOT FOUND BY 032** | README stale (claims nothing implemented) — I did not review README |
| — | `031-M3` | **NOT FOUND BY 032** | `budget_limited` written but absent from `REQUIRED_CHECKPOINT_KEYS` |
| — | `031-M4` | **NOT FOUND BY 032** | `load_prepared_chunks` routes any non-`train` partition to dev |
| — | `031-I2`, `031-I3` | **AGREE** | Consistent with §§35, 17 |
| *(none)* | — | **NOT FOUND BY 031** | **No finding of mine is absent from Audit 031.** 031 is a strict superset |

**Net:** 031 found everything I found, plus one blocker and four minors I missed. I contributed
sharper evidence on two findings, not new findings.

---

## 33. Independent Verification of `031-B1` (the finding I missed)

I did not accept this on Audit 031's authority. Re-derived from source:

```
execute.py:186   from unmark.stage1.objective import Stage1Objective   # bound (local)
execute.py:299   objective = objective_cls(unmark_encoder, provenance.weights)   # NOT bound
execute.py:480   tokenizer, unmark_encoder, objective_cls = build_objective(revision)  # different function
```

AST scope analysis of `execute_stage()` (lines 152–420):
```
objective_cls BOUND inside execute_stage : False
objective_cls LOADED inside execute_stage: True
objective_cls bound at MODULE level      : False
Stage1Objective bound at MODULE level    : False
```

Bytecode confirmation:
```
execute_stage locals contain objective_cls   : False
execute_stage names   contain objective_cls  : True   -> compiled as LOAD_GLOBAL
execute_stage locals contain Stage1Objective : True   -> imported at 186, unused at 299
```

Runtime confirmation:
```
hasattr(unmark.stage1.execute, "objective_cls") = False
```

**Confirmed.** `LOAD_GLOBAL objective_cls` at line 299 resolves against a module namespace that has no
such attribute, raising `NameError: name 'objective_cls' is not defined`. This fires inside
`execute_stage()` **before `train_run()` is reached**, after preflight, device resolution, data
loading, backbone construction and adapter initialisation have all succeeded.

This is an incomplete refactor: `build_objective()` returns the class as a third tuple element and
line 480 binds it that way, but `execute_stage()` imports `Stage1Objective` directly at 186 and then
calls the stale name. **The first `lr-pilot` invocation cannot reach the optimizer.**

I also verified Audit 031's minors `M1`, `M3`, `M4` from source; all three are accurate. `M3` gains
force from a detail neither audit stated: **`cap` *is* in `REQUIRED_CHECKPOINT_KEYS`** —
```
REQUIRED_CHECKPOINT_KEYS: ['adapter_state','cap','execution','global_update',
                           'optimizer_state','points','provenance','sampler_state','schema_version']
```
so the schema *mandates* persisting the cap that no reader ever consumes. That is direct evidence the
omission in `032-B2`/`031-B3` is an oversight, not a design choice.

---

## 34. Divergences

### 34.1 Severity: `032-MAJ1`/`MAJ2` (MAJOR) vs `031-B4`/`B5` (BLOCKER)

I classified the handoff-validation and HEAD-provenance gaps as MAJOR. Audit 031 classified them as
BLOCKER. Reviewing 031's reasoning, **I move toward its position but do not fully adopt it.**

- Neither defect corrupts the numerical result of a single correctly-run experiment. In that narrow
  sense they are not launch blockers, which is why I rated them MAJOR, and I was conscious of the
  standing instruction not to over-classify hardening opportunities.
- But Audit 031's argument is about **campaign integrity**, and it is a good argument: an artifact
  that misidentifies its own commit, or a downstream stage that accepts a foreign artifact, produces
  results that *look* protocol-compliant and are not. For a multi-stage campaign whose entire claim
  rests on a frozen protocol, an undetectable provenance error is worse than a crash.
- The practical difference is nil: **both audits agree both defects must be fixed before the
  campaign.** I therefore record the disagreement honestly rather than resolving it, and defer the
  severity label to the human. It does not change the verdict either way.

### 34.2 Detail added by 032

- **`032-B2` silent mode.** Audit 031 frames B3 as post-crash recovery failure. I additionally showed
  the *silent* mode: if the selected checkpoint is ≤ 20 000, a 40 000-update leg records `cap=20000`
  and `continued=False` and is written to the artifact **as a complete 20 000-update run, with no
  error**. That is provenance falsification rather than a crash, and is the more dangerous half.
- **`032-MAJ2` self-contradiction.** Audit 031 notes Stage-6 derives HEAD while Stage-1 accepts it. I
  add that `resolve_repository_head`'s own docstring states there is *"deliberately **no CLI flag and
  no environment override**"* precisely because *"a caller-claimed HEAD would let a checkpoint written
  by commit A resume under commit B while asserting it did not."* The repository states the correct
  principle in its own source and violates it one module away.
- **No dirty-tree guard.** Grep-verified: `dirty|is_clean|status --porcelain|require_clean` has **no
  matches** in `unmark/stage1/*.py` or `scripts/stage1_runner.py`.
- **§V head gate is vacuous.** Executed probe: with `--repository-head` omitted, provenance records
  `None`, and `require_match(None, None)` **passes** — the resume-blocking head check introduced in
  Audit 030 §V does not gate.

---

## 35. UNION Table — Credible BLOCKER / MAJOR Findings

Every row below was verified from executable source by this review, independently of which audit
found it first.

| # | Finding | 031 | 032 | Union severity | Site | Blocks |
|---|---|---|---|---|---|---|
| 1 | `objective_cls` unbound → `NameError` before `train_run()` | `B1` | *missed* | **BLOCKER** | `execute.py:299` | **every run**, incl. first `lr-pilot` |
| 2 | `ValidationPoint(**p)` rejects writer-emitted `score` → `TypeError` | `B2` | `B1` | **BLOCKER** | `trainer.py:324` ↔ `621` | every `--resume`; 20k→40k continuation |
| 3 | Persisted `cap` never read; hardcoded `INITIAL_MAX_UPDATES` | `B3` | `B2` | **BLOCKER** | `trainer.py:322`, `execute.py:334` | 40k leg: `SelectionViolation` **or silent 20k mislabel** |
| 4 | Stage handoff validates only `stage` + `protocol_version` | `B4` | `MAJ1` | **BLOCKER / MAJOR — disputed (§34.1)** | `stage1_runner.py:475-487` | campaign integrity across stages |
| 5 | `--repository-head` optional, caller-supplied, unverified; no dirty-tree check | `B5` | `MAJ2` | **BLOCKER / MAJOR — disputed (§34.1)** | `stage1_runner.py:513/531` vs `checkpoint.py:353` | provenance of every artifact |

**Union: 3 undisputed BLOCKERS + 2 findings disputed between BLOCKER and MAJOR.**

Findings 1–3 are strictly sequential in the failure order an operator would encounter them:
#1 stops the first command; fixing it exposes #2 on the first resume; fixing that exposes #3 on the
continuation leg. **They must be fixed and re-verified together**, or each fix will simply reveal the
next failure after another multi-hour run.

---

## 36. Minor and Informational Union

| ID(s) | Subject | Status |
|---|---|---|
| `031-M1` | Freeze file records `649ad741…`; HEAD is `55aa4064…` | Confirmed by 032; documentation-stale only |
| `031-M2` | README claims nothing implemented | Accepted from 031; not independently reviewed by 032 |
| `031-M3` | `budget_limited` written but not in `REQUIRED_CHECKPOINT_KEYS` | Confirmed by 032; strengthened (§33) |
| `031-M4` | `load_prepared_chunks` routes non-`train` → dev | Confirmed by 032; helper-contract only, production path verifies first |
| `031-M5` / `032-M1` | Tests mask the launch and resume failures | **Both audits agree.** 032 adds: the resume test is torch-free and runs locally (27 passed, 0 skipped), so this was missed by test *design*, not environment |
| `031-I1` / `032-I1` | ML-free local `.venv` | Deliberate; both agree |
| `031-I4` / `032-I2` | TEST sealed | Both agree |
| `032-I3` | `855a1d3` "fix bug before training stage-1" changes no production code | Both agree; 031 proved byte-identity by diff |
| `032-I4` | torch deliberately unpinned (Colab-supplied) | Noted; policy accepted |

---

## 37. Scientific `optimizer.step` Status

**No real scientific `optimizer.step` was executed by this review.**

- The single Stage-1 production optimizer step is `unmark/stage1/trainer.py:677`.
- `lr-pilot`, `r-phase1` and `final-main` were not invoked.
- Steps reachable under pytest are TEST-ONLY fixtures on synthetic tensors, and in this ML-free
  environment the torch-dependent ones were skipped entirely (103 skips).
- Independent of intent, `031-B1` means `execute_stage()` **cannot currently reach** `train_run()` at
  all — no scientific optimizer step is reachable at this HEAD even if one were attempted.

This matches Audit 031 §32.

---

## 38. Official UIT-VSFC TEST Status

**SEALED / UNUSED.**

This review did not open, inspect, read, screen, evaluate, mount, or pass official TEST to any
command, and derived no information from it. Enforcement verified fail-closed at `manifest.py:221`
(`raise ManifestViolation` unless `official_test_used is false`). No Stage-1 production path
constructs a TEST loader.

This matches Audit 031 §33.

---

## 39. Final Verdict

Three blockers are undisputed between the two audits and were verified from executable source by
both. One of them — `objective_cls` unbound at `execute.py:299` — prevents the **first** Stage-1
command from reaching the optimizer at all. Two further findings are agreed in substance and disputed
only in severity, and both audits agree they must be fixed before the campaign.

Audit 031's verdict is confirmed, and this review's Phase-1 verdict is corrected by it: my
independent pass concluded the blockers were confined to the resume path, which was wrong.

**SECOND INDEPENDENT FULL-REPOSITORY REVIEW BLOCKED — MATERIAL FINDINGS REQUIRE JOINT HUMAN REVIEW BEFORE TRAINING**

No repair was performed. No file was modified except this audit. Nothing was staged.

---

*End of Audit 032.*
