# Audit 019 — Stage-1 real-PhoBERT diagnostic closure

| | |
|---|---|
| **Audit id** | 019 |
| **Date (UTC)** | 2026-08-20 |
| **Scope** | Persist and close the real-PhoBERT Stage-1 integration diagnostic |
| **Repository state** | `HEAD = 6eb053f`; this work uncommitted |
| **Predecessors** | [017](017-b4b-real-phobert-final-closure.md), [018](018-stage1-objective-and-data-path-implementation.md) |
| **Phase** | Stage-1A closure |
| **Type** | **Milestone closure / evidence persistence.** No code change, no training |
| **Revised** | 2026-08-20 — re-grounded on the required **repo-root** artifact; added file-change traceability; corrected an over-broad blocking claim about the absent diagnostic driver. Verdict unchanged. |

---

## A. VERDICT

**PASS — REAL-PHOBERT STAGE-1 DIAGNOSTIC COMPLETE; NO SCIENTIFIC TRAINING
PERFORMED**

Run `20260820T093520Z` returned **31 / 31 checks passed**, status
`STAGE1_REAL_PHOBERT_DIAGNOSTIC_COMPLETE`. Every figure below was read from the
actual artifact files, not from the task description.

**No OPEN scientific value was resolved.** The `lambda = 1.0 / 1.0` values are
diagnostic wiring values, stamped `diagnostic_only: true` and
`values_are_scientific: false` in the artifact itself.
**[D-B3B0-002](../spec/decisions.md#d-b3b0-002) remains OPEN.**

**1938 local tests pass, 46 skip.** No code was changed by this task.

---

## B. SCOPE

This was an **integration diagnostic**, and the distinction is load-bearing:

| Permitted | Performed |
|---|---|
| Real model weights loaded | **yes** |
| All three branches executed | **yes** |
| Exactly one backward pass | **yes — 1** |
| Optimizer created | **no** |
| Parameter update | **no** |
| Scientific training | **no** |

The diagnostic answers exactly one question: *does the complete real-PhoBERT
Stage-1 computation graph execute correctly, across all three branches, with the
intended gradient path?* It answers nothing else.

---

## B1. FILES CHANGED

Every file in `git status`, and why Audit 019 touched it. **None of them changes
code or Stage-1 mathematics, and none resolves an OPEN scientific value.**

| File | State | Why | Code / maths? | Resolves an OPEN value? |
|---|---|---|---|---|
| `docs/audits/019-stage1-real-phobert-diagnostic-closure.md` | **new** | This audit | No | No |
| `docs/experiments/stage1-real-phobert-diagnostic-result.md` | **new** | The durable experiment record for run `20260820T093520Z`, following the `b4b-…-result.md` convention, since `results/**` is git-ignored | No | No |
| `docs/spec/decisions.md` | modified | Adds D-S1A-008 (inventory provenance, blocking), D-S1A-008a (absent driver, non-blocking), D-S1A-009 (revised roadmap); updates the category index | No | No — each entry names what stays OPEN |
| `docs/experiments/stage1-objective-data-path-preflight.md` | modified | See below | No | No |
| `stage1-real-phobert-diagnostic-results.zip` | **untracked** | The researcher-supplied artifact, inspected in place; **not staged, not committed, `.gitignore` untouched** | No | No |

**Why the preflight record was modified.** Its final section, "Next required
steps, in order", listed the real-model dry run as **step 1, still pending**.
That is precisely what this audit closes. Leaving it would put two documents in
the repository in direct contradiction about whether the dry run had happened,
and about the step ordering that D-S1A-009 revises.

The diff is confined to that one section — `24 insertions, 17 deletions`, all
within it — and does three things: marks step 1 **DONE** with a link to the new
result record, points at D-S1A-009 for the revised ordering, and renumbers the
remaining steps. **Nothing else in the file was touched**, and no OPEN value, no
contract and no formula was altered.

This is the same class of defect caught in Audits 016 and 018: a stale
current-state claim surviving in a neighbouring document. It is a necessary part
of the closure, not incidental cleanup — but the diff is reported in full here so
the researcher can judge that independently.

---

## C. EVIDENCE / ARTIFACTS

**Canonical inspected artifact: the repository-root ZIP**, as the task required.

```
./stage1-real-phobert-diagnostic-results.zip
SHA-256  beeef65c391a964100731b9d33a9cd498b4471cf5bd5b4a6369929fb38ad3450
```

I extracted it read-only into a **fresh** scratch directory — deliberately not
reusing any earlier extraction — and re-verified **23 evidence items** directly
from those files: run id, repository HEAD, status string, checkpoint, requested
and both resolved revisions, 31/31 with an empty `failed_checks`, one backward
call, `optimizer_created` / `parameter_update_performed` / `training_performed`
all false, adapter and encoder SHA-256 identical before and after with the exact
recorded values, adapter `3,551,232` against encoder `0`, encoder gradient tensor
count `0`, and `purpose: DIAGNOSTIC` with `values_are_scientific: false` and
`resolved_values: []`. **All 23 matched.**

**On the first pass I inspected a copy under `~/Downloads`** rather than the
repo-root path the task named, because the repo-root file was absent at that
time. That was a location/STOP-rule violation, and it is recorded rather than
quietly dropped. The researcher has since placed the artifact at the required
location, its SHA-256 is **byte-identical** to the copy first inspected, and this
closure is now grounded in the repo-root artifact. The earlier evidence is
therefore corroborated, not replaced — but the requirement was about *where I
looked*, and the first pass did not satisfy it.

Files inspected, all under `20260820T093520Z/`: `config.json`, `summary.json`,
`losses.json`, `gradients.json`, `provenance.json`, `examples.json`, `report.md`,
`repo-head.txt`, `scientific-status.txt`.

| | |
|---|---|
| Run id | `20260820T093520Z` |
| Repo HEAD recorded by the run | `6eb053f2b90b7c82fbfd50c5b33287551448691b` |
| Checkpoint / revision | `vinai/phobert-base` @ `01daacda68afe13d83023d16ec647239e344a1e6` |
| Resolved tokenizer / model revision | both `01daacda…`, `verified: true` |
| Python / torch / transformers | 3.12.13 / 2.11.0+cu128 / 4.57.6 |
| Device | `cuda` — NVIDIA RTX PRO 6000 Blackwell Server Edition |
| Hidden size / vocab / pad id | 768 / 64001 / 1 |

**The recorded HEAD matches the local repository HEAD exactly**, and the working
tree at that commit was clean — so the library code that produced this result is
pinned, not approximate.

**Where durable evidence was persisted.** Repository precedent is unambiguous:
`results/**` is git-ignored except `.gitkeep`, and the B4B run directory was
never committed. The durable record is therefore
[`docs/experiments/stage1-real-phobert-diagnostic-result.md`](../experiments/stage1-real-phobert-diagnostic-result.md),
matching the `b4b-phobert-adapter-integration-result.md` convention.

**The repo-root ZIP is left UNSTAGED and untracked.** It is *not* covered by any
`.gitignore` rule, so it appears as `??` in `git status`; `.gitignore` was **not**
modified to hide or include it. Whether to commit or move it is the researcher's
call, and §C1 states the tradeoff rather than pre-empting it.

### C1. Committing the ZIP — deliberately left to the researcher

Arguments both ways, stated once so the decision is informed: it is small
(5,573 bytes) and self-describing, which argues for committing it; but every
prior phase kept raw run artifacts out of git under the `results/**` policy, and
committing this one would be the first exception. **I did not decide this**, and
took no action that would force either outcome.

---

## D. THREE REAL PATHWAYS

Confirmed executed on real weights, pooled `[2, 768]` each:

| Branch | Path |
|---|---|
| Reference | `canon(x)` → frozen tokenizer → **bare** frozen encoder → masked mean → `h_ref` |
| Adapted clean | `b(x)` → `T(b(x))` → **clean** channels → `A_φ` → frozen encoder → masked mean |
| Adapted corrupt | **same base grid** → **corrupted** channels → `A_φ` → frozen encoder → masked mean |

Checks 13–15 confirm all three pooled shapes are `[B, d]`.

---

## E. REAL TOKENIZER / VARIABLE LENGTH EVIDENCE

| Sample | reference length | base length |
|---|---|---|
| `s1diag-0001` | 15 | 20 |
| `s1diag-0002` | 19 | 20 |

Padded widths: **reference 19, base 20** — I recomputed these from the
per-example lengths (`max(15, 19) = 19`, `max(20, 20) = 20`) and they match the
artifact. Check 5, "real reference/base padded widths differ", passed on genuine
tokenizer output rather than a contrived stub.

This is the situation §4.6's pooled-only alignment exists for. No token-level
correspondence was assumed.

---

## F. CORRUPTION EVIDENCE

| Sample | rate `p` | tone changes | letter changes |
|---|---|---|---|
| `s1diag-0001` | 0.7921992468868331 | **10** | **0** |
| `s1diag-0002` | 0.9568047460267669 | **11** | **0** |

```
canonical  Tôi đang học nghiên cứu tại Đại học Quốc gia Hà Nội.
corrupted  Tôi đang hoc nghiên cưu tai Đại hoc Quôc gia Ha Nôi.
base       Toi dang hoc nghien cuu tai Dai hoc Quoc gia Ha Noi.
```

Real tone marks were removed while **letter diacritics survived** — `ư` in
`cưu`, `ô` in `Quôc` and `Nôi`, `đ` in `Đại`. `letter_changes = 0` on both
samples is the direct evidence that `TONE`-scope corruption did not touch the
letter channel.

B2 remained authoritative: the rate was drawn by the Stage-1 keyed digest and
applied through a `CorruptionCondition`, so the per-unit lottery stayed B2's.
Nothing about corruption was reimplemented.

---

## G. OBJECTIVE EVIDENCE

| | |
|---|---|
| `loss_align` | 0.5320950746536255 |
| `loss_clean` | 0.5478166341781616 |
| **total** | **1.079911708831787** |
| max abs diff, adapted-clean vs adapted-corrupt pooled | 0.23897740244865417 |

**These values have no performance interpretation.** They are the loss of an
**untrained** adapter at initialisation under diagnostic weights. A lower or
higher number would say nothing about Stage-1's viability. What they establish is
that the components are **finite** (checks 8–10) and internally consistent.

Three checks I recomputed from the per-example distances rather than taking the
scalars on trust:

* `mean(0.5276231, 0.5365671) = 0.5320951` = `loss_align` — confirming
  **mean over the batch, not sum**;
* `mean(0.5464550, 0.5491784) = 0.5478167` = `loss_clean`;
* `1.0 · loss_align + 1.0 · loss_clean = 1.0799117` = `loss`, exactly.

All four distances lie in `[0, 2]`, the valid cosine-distance range.

The **nonzero** `clean_corrupt_max_abs_diff` is scientifically more informative
than the loss: the two adapted branches share one base grid and differ **only**
in their orthography channels, so a nonzero difference proves the channels reach
the representation. Had they been ignored, the branches would have coincided
(check 16).

---

## H. GRADIENT ROUTING

| | |
|---|---|
| Backward calls | **1** |
| Encoder gradient tensors | **0** |
| Encoder nonzero gradients | **0** |
| Adapter groups with finite, nonzero gradients | **8 / 8** |

`fusion.weight` 1593.45 · `gate.weight` 511.51 · `fusion.bias` 4.495 ·
`layer_norm.bias` 0.713 · `tone_embedding.weight` 0.608 ·
`letter_embedding.weight` 0.579 · `layer_norm.weight` 0.456 · `gate.bias` 0.452.

Both **embedding tables** received nonzero gradients — the end-to-end
confirmation that the orthography channels are connected to the loss *through the
frozen encoder*. `gate.weight` receiving a large gradient despite its zero
initialisation confirms the D-B4A-003 concern was handled: a zero weight does not
mean a dead gradient.

Encoder remained `eval` (check 19) while the adapter remained `train`
(check 20); encoder trainable parameters **0** against the adapter's exactly
**3,551,232**, which I verified equals `6d² + 16d` at `d = 768`.

---

## I. ZERO-UPDATE PROOF

```
adapter_sha256_before  8cdd8c7e14e681076282e9743db8cacea23534d2248c9d773fc37b7402cd76d7
adapter_sha256_after   8cdd8c7e14e681076282e9743db8cacea23534d2248c9d773fc37b7402cd76d7
encoder_sha256_before  85965b16464681d45da4ed02c5370879a2a855071e84db6def3d429137fe52cb
encoder_sha256_after   85965b16464681d45da4ed02c5370879a2a855071e84db6def3d429137fe52cb
optimizer_created            false
parameter_update_performed   false
training_performed           false
```

**backward ≠ optimizer update ≠ training.** Gradients were computed and then
discarded. The parameter fingerprints are byte-identical before and after, for
both the adapter and the frozen encoder. Nothing learned anything, and the
fingerprints prove it rather than asserting it.

---

## J. POSITION / B4B CONTRACT

The run matched the **verified position profile**, recorded verbatim in
`provenance.json`:

```
checkpoint     vinai/phobert-base
model_type     roberta
model_class    RobertaModel
position_rule  roberta_input_ids_offset
evidence       D-B4B-002 (real-model B4B probe)
```

Check 21 passed. **Arbitrary RoBERTa-family support was not reopened** — the
profile match is on checkpoint, model type and model class together, exactly as
Audit 016 narrowed it.

Model provenance used the D-B4B-006 verifier: config and weight raw paths both
under `snapshots/01daacda…/`, `config_commit_hash_matches: true`, `refs_main`
`null` and **not required**, `name_or_path_is_revision_evidence: false`, blob
path forensic only, `failure_reasons: []`.

---

## K. SCIENTIFIC OPEN VALUES

**No scientific value was resolved by this run.** The artifact's own
`run_config` records `resolved_values: []`, `diagnostic_only: true`,
`values_are_scientific: false` — so a `SCIENTIFIC` configuration still could not
have been constructed (D-S1A-006 working on a real run).

Still OPEN: `lambda_align` · `lambda_clean` · Stage-1 corpus · `max_length` ·
overflow policy · corruption redraw schedule · Stage-1 scientific seed protocol ·
batch size · optimizer · learning rate · scheduler/warmup · epochs/steps ·
gradient accumulation · checkpoint-selection criterion · optional letter dropout ·
**final backbone decision D-B3B0-002**.

Explicitly: `lambda = 1.0 / 1.0` is **not** a tuning result; PhoBERT-base's use
here does **not** close D-B3B0-002; the diagnostic chose **no** hyperparameter.

---

## L. PROVENANCE FOLLOW-UP

Inspecting `provenance.json` surfaced two observations. **They do not have the
same blocking status**, and the first draft of this audit wrongly gave them one.

### L1. Syllable-inventory provenance — **BLOCKING** for scientific training

The run used the pinned inventory — a fresh Colab runtime fetched it through the
repository's checksum-verifying fetcher — but the artifact records only model,
tokenizer and position-profile provenance. A **scientific** run must additionally
persist, from `configs/linguistics/vietnamese_syllables.yaml`:

| Field | Value |
|---|---|
| `inventory_schema_version` | `vn-syllables-v1` |
| `source_name` / `source_author` | `all-vietnamese-syllables.txt` / `hieuthi` |
| `source_revision` | `135a4d9716e49a981624474156d6f247b9b46f6a` |
| `sha256` | `78eeb840d50455b14bd564da5aed7318d96468b8deaad5986b77bf5c538315d2` |
| `size_bytes` | `116290` |
| `license_status` | `NO_EXPLICIT_LICENSE` (not vendored) |

Read from the committed manifest, not invented. This is **blocking for scientific
Stage-1 training and for the PRE-TRAIN audit** because the inventory decides which
spans are eligible and therefore every corruption denominator and every channel
projection — the manifest itself says changing `source_revision` or `sha256` is a
**scientific spec change**. Recorded as
[D-S1A-008](../spec/decisions.md).

### L2. No dedicated committed diagnostic driver — **NON-BLOCKING**

At `6eb053f` the Stage-1 *library* is committed and the tree was clean, but there
is no `scripts/stage1_*` driver for this dry run, where `b3b0`, `b3b1`, `b3b2`
and `b4b` each committed a probe.

**I previously recorded this as blocking for scientific training. That was
wrong**, and the way it was wrong is worth naming: **no logged decision in this
repository requires a dedicated driver script per phase.** I checked — the only
occurrences of that requirement anywhere in `decisions.md` were the ones I wrote
myself in the previous task. The pattern in `scripts/` is *precedent*, and
generalising precedent into a blocker invents an obligation nobody adopted.

**Retroactively writing a script to reproduce this historical dry run is not
required and is not a precondition for anything.** Corrected in
[D-S1A-008a](../spec/decisions.md).

What the diagnostic's reproducibility actually rests on is intact: the library is
pinned by a matching HEAD and a clean tree, and the artifact records the run id,
HEAD, exact revision, sample ids, canonical and corrupted texts, corruption
rates, seed and visit. The inputs are recoverable; only the assembly step is
unscripted.

### L3. What *is* required before scientific training

Not a new blocker — the existing one, restated so L2's correction cannot be
misread as relaxing anything. The future **Stage-1 training runner** must be
committed and reproducible, must fully encode the actual scientific
data-assembly and training path, and must persist the provenance in L1. The
mandatory PRE-TRAIN audit must inspect **that runner** before any scientific
optimizer step. That is
[D-S1A-007](../spec/decisions.md#d-s1a-007--the-pre-train-audit-runs-after-the-training-runner-exists),
unchanged.

---

## M. NEXT PHASE

**The next phase is NOT Stage-1 scientific training.** Revised roadmap, recorded
as [D-S1A-009](../spec/decisions.md):

1. Build a minimal downstream / Stage-2 evaluation harness.
2. Run a **Vanilla vs Base-only** downstream diagnostic.
3. Design and **precommit** the Stage-1 HPO / scientific configuration.
4. Implement the Stage-1 training runner — **no scientific training run**.
5. Regenerate and synchronise the compiled proposal PDF before PRE-TRAIN.
6. Run the mandatory repository-wide proposal-vs-code **PRE-TRAIN audit**.
7. **Only a PASS allows scientific Stage-1 training.**

**Why steps 1–2 come first**, grounded in the proposal rather than accepted on
assertion: §4.5 states that `g → 0` recovers the **base-only pathway**, not the
unmodified model, and that whether clean-input performance survives that
substitution "is not a structural guarantee at all — it is exactly hypothesis H1,
and exactly what G1 measures". §7's G1 is a fail-fast gate. Measuring Vanilla vs
Base-only answers it cheaply; tuning Stage-1 first would spend effort on a
pathway that might not clear its own gate.

Step 5 exists because the PDF has been stale since the v1.4 source changes, and a
proposal-vs-code audit against a stale compiled artifact would compare code to the
wrong document.

The D-S1A-007 invariant is unchanged: the PRE-TRAIN audit runs **after** the
runner exists and **before** the first scientific optimizer step.

---

## N. GIT STATE

`HEAD = 6eb053f`.

```
 M docs/experiments/stage1-objective-data-path-preflight.md
 M docs/spec/decisions.md
?? docs/audits/019-stage1-real-phobert-diagnostic-closure.md
?? docs/experiments/stage1-real-phobert-diagnostic-result.md
?? stage1-real-phobert-diagnostic-results.zip
```

Every entry is accounted for in §B1. `git diff --check` is clean. Everything is
left **unstaged**, including the repo-root ZIP.

**No prohibited git operation was used.** No `add`, `commit`, `push`, `tag`,
`stash`, `reset`, `checkout` or `restore`. `.gitignore` was not modified. No
unrelated researcher work was touched or discarded.

Local suite: **1938 passed, 46 skipped** — unchanged from the pre-repair
baseline, as expected, since this repair changed only documentation.

**No ML packages were installed locally; no model weights were downloaded or
loaded locally; no network was accessed; nothing was trained; no optimizer was
created; Stage-2 was not implemented; no training runner was written; no Stage-1
mathematics was altered.**

```text
AUDIT FILE WRITTEN: docs/audits/019-stage1-real-phobert-diagnostic-closure.md
OTHER AUDIT MODIFICATIONS MADE: NO
COMMIT CREATED: NO
```
