# Audit 026 — pre-G1 paired measurement runner

| | |
|---|---|
| **Audit id** | 026 |
| **Created (UTC)** | 2026-08-22 |
| **Baseline HEAD** | `d10aaae7c38c6be9f0d382dc9d144f80afe2fc12` |
| **Scope** | Wire the committed `measure` subcommand. **Run nothing.** |
| **Predecessor** | [025](025-preg1-executable-diagnostic-runner.md) — tuning runner |
| **Type** | Wiring + tests. No protocol change, no model, no data, no training, no score |
| **Review 1** | **2026-08-22** — checkpoint-snapshot independence reviewed. The implementation was **already correct**; **no production change**. Executable proof added (§C.1). |

---

## A. VERDICT

**IMPLEMENTATION PASS — MEASUREMENT RUNNER WIRED; NOT RUN**

**2 353 local tests pass, 91 skip** (2 337 / 89 before). Review 1 added one
local structural test and two torch-gated behavioural tests.

No scientific constant changed. **D-B3B0-002 OPEN**, final Stage-2 pooling
**OPEN**, official TEST **SEALED** and structurally unreachable, PDF **STALE**,
Stage-1 untouched. **No new scientific decision.**

---

## B. TUNING COMPLETED — THE FROZEN LR

Vanilla LR tuning executed at HEAD `d10aaae7c38c6be9f0d382dc9d144f80afe2fc12`
and selected:

| LR | mean selected-checkpoint macro-F1 |
|---|---|
| 1e-4 | 0.6087673586082464 |
| 3e-4 | 0.6708010500564968 |
| 1e-3 | 0.7157811478359951 |
| 3e-3 | 0.7481156357379474 |
| **1e-2** | **0.7545393301481959** |

**FROZEN LR = 0.01.** It wins on the **first** criterion of the locked
aggregation rule — highest mean macro-F1 — by a unique margin, so **no
tie-break was consulted**: accuracy, sample SD and smaller-LR never came into
play. Official validation was not used during tuning; official TEST was not
used. **Tuning was not rerun.**

---

## C. WHAT WAS WIRED

`measure` now executes 5 seeds × 2 pathways = **10 head-training runs**, all out
of committed APIs:

| Step | API |
|---|---|
| verify the frozen LR against the tuning artifact | `load_tuning_artifact` (new, thin) |
| approved TRAIN pool | `load_derived_pool` |
| Audit-023 membership | `load_membership` + `require_partitions` |
| official validation, on its own locked identity | `load_derived_pool` with explicit SHA/rows/labels |
| Vanilla and Base-only text | `pathway_text` — `canon(x)` and `b(canon(x))`, no segmenter |
| representations, one set per (role, pathway) | `extract_representations` + `RepresentationCache` |
| training and checkpoint selection | `train_head`, then `run.selected` |
| scoring the selected checkpoint | `build_head` + `score_measurement` |
| the report | `PairedSeedResult` + `PairedDiagnostic` |

**No second trainer, selector, metric or protocol.** A test asserts the
measurement calls `train_head` at exactly one site, takes its checkpoint from
`run.selected`, and performs no local maximum scan.

### Two production changes — both additive, both flagged

**1. `train_head` gained an optional `on_checkpoint(epoch, score, head)` hook.**
This was unavoidable. The protocol requires scoring **the selected checkpoint**
on official validation, the selection is only known after all 30 epochs, and
`train_head` returned no head and gave `on_epoch` no access to one. Without the
hook the only alternative was to reimplement the training loop — explicitly
forbidden, and far more dangerous. The hook changes no equation, no value and no
default behaviour; a caller that ignores it gets exactly the previous semantics.
The runner snapshots each epoch's `state_dict`, then restores the epoch the
**committed selector** chose. Snapshot independence is reviewed in §C.1.

**2. `DERIVED_VALIDATION_CSV_SHA256` was recorded in `preg1_protocol`.** Not a
new decision: the value was established by the D-PREG1-011 reproduction recipe
and re-verified by C24-5B-R1 (Audit 024 §S.8). It is recorded so the runner can
gate on the file's identity rather than trust a filename. Its row count and
label counts are **not** restated — they come from the existing
`PUBLISHED_SPLIT_SIZES["validation"]` and `PUBLISHED_LABEL_COUNTS["validation"]`.

### C.1 Checkpoint snapshots are independent — reviewed, no change needed

`Module.state_dict()` returns tensors that **share storage with the live
parameters**. A bare `saved[epoch] = head.state_dict()` would leave all 30
entries tracking the optimizer, so every "checkpoint" would end up holding the
final-epoch weights and the selector's choice would be silently ignored — a
defect that changes the reported numbers while looking entirely normal.

**The implementation was already safe.** `scripts/preg1_head_diagnostic.py:546`:

```python
_store[epoch] = {k: v.detach().clone() for k, v in head.state_dict().items()}
```

`.clone()` allocates new storage and copies; `.detach()` alone would still
alias. **No production change was made.**

**What was missing was proof**, so three tests were added:

| Test | Runs | Proves |
|---|---|---|
| `test_the_runner_clones_every_checkpoint_tensor` | **locally** | the capture clones; a bare `state_dict()` assignment is rejected |
| `test_checkpoint_snapshots_survive_later_training` | torch-gated | an early checkpoint does not drift, and restoring it does not yield final-epoch weights |
| `test_every_epoch_snapshot_is_a_distinct_object_with_its_own_storage` | torch-gated | no two checkpoints share a `data_ptr` |

The behavioural test is deliberately **self-validating**: it captures a bare
`state_dict()` alongside the cloned one in the same run and first asserts that
the bare entry *did* alias. If that ever stops holding, the test fails and the
assumption gets reviewed rather than silently assumed. It then asserts the
cloned epoch-1 snapshot differs from epoch 30, and that restoring it reproduces
epoch 1 exactly and not the final weights.

**Mutation-verified:** reverting line 546 to `head.state_dict()` fails the local
structural test.

**Honest scope:** the two behavioural tests are torch-gated and have **not
executed** — they are written, not run. The local suite proves the code shape,
not the tensor semantics.

### Boundaries

- **Official validation is measurement-only.** `train_head` is called with
  protocol-train and protocol-dev *only*; `score_measurement` is the sole
  consumer of the validation representations. A test parses the actual call
  nodes and asserts `OFFICIAL_VALIDATION` never appears in the trainer's
  arguments, and that `run_measure` calls no LR selector at all.
- **The LR cannot be substituted.** `--frozen-lr` must equal the artifact's
  `selected_learning_rate`; the artifact must record `selected_on = VANILLA` and
  `official_validation_used = false`. Every other grid rate is refused.
- **Base-only cannot retroactively affect the LR** — by the time it is encoded,
  the rate is a value in a verified file.
- **Pairing.** The pathway loop is nested inside the seed loop and both arms are
  trained with the same `seed`, so `build_head`'s reseed gives bit-identical
  initial parameters (Audit 024 Q4, TORCH-PASS). Asserted on the AST nesting,
  not on text order.
- **Provenance fails closed.** Pathway is part of `RepresentationKey`, so a
  Vanilla cache can never load as Base-only; `extract_or_load` calls
  `cache.load(key)` with the key it requires.
- **Official TEST** — no enum member, no attribute access, no flag.
- **No significance machinery.** The report comes from the committed
  `PairedDiagnostic`: raw scores, per-seed deltas, means and **sample** SDs, and
  no p-value, threshold or pass/fail field.

Artifacts are deterministic, hold ids/digests/floats only, land under gitignored
`results/…`, and an existing output directory is refused.

---

## D. TESTS

**15 new** (40 in `tests/test_preg1_runner.py`), all ML-free: exactly 5 seeds
and one `train_head` site; frozen LR must equal the artifact's selection; every
other grid rate refused; selection must have been on VANILLA; a tuning run that
touched official validation refused; missing and malformed artifacts refused;
official validation gated on its own SHA/rows/labels; official validation never
reaches the trainer or a selector; the same seed drives both arms with the
pathway loop nested inside; pathway-separated keys refuse cross-loading;
provenance mismatch fails closed; the report carries no significance field; no
raw text persisted; existing output refused; wrong revision refused.

**Two mutations confirmed the critical guards are live:** dropping the
frozen-LR equality check → 2 failures; selecting checkpoints on official
validation → 1 failure. Green again after restoring.

**Two of my own tests were defective and repaired**, both the same trap this
project keeps hitting — matching prose or first textual occurrence rather than
structure. One grepped for `BASE_ONLY` and hit a docstring explaining that a
Vanilla cache *cannot* be loaded as Base-only; the other compared string indices
and found the extraction loop instead of the training loop. Both are now AST.

---

## E. NOT RUN

No Colab execution. No model loaded, no corpus read, no representation
extracted, no head trained, **no downstream score**. Official validation was not
read; official TEST untouched.

---

## F. NEXT ACTION

**Only after researcher review and commit**: run the **10 paired runs** on Colab —

```
python scripts/preg1_head_diagnostic.py measure \
    --derived-train <derived_train.csv> --split-dir <preg1-split-v1-…> \
    --official-validation <derived_dev.csv> \
    --text-column text --label-column label --id-column id \
    --cache-root <cache> --output-dir <results/preg1-measure/…> \
    --frozen-lr 0.01 --tuning-artifact <results/preg1-tune/…/tuning.json> \
    --repository-head <sha>
```

5 Vanilla + 5 Base-only, frozen encoder, LR 0.01. **This produces the first
downstream number in the project.** It measures a burden; it does not test
UNMARK and must not be reported as if it did.

---

## G. SELF-AUDIT

| # | Check | Result |
|---|---|---|
| 1 | Audit 026 created; concise; no historical rewrite | **yes** |
| 2 | Frozen LR 0.01 recorded with its selection reason | **yes** — unique highest mean macro-F1, no tie-break |
| 3 | Tuning not rerun | **yes** |
| 4 | Exactly 5 seeds × 2 pathways = 10 runs | **yes** |
| 5 | Same seed pairs the two arms; init identity preserved | **yes** — nesting asserted on the AST |
| 6 | Frozen LR verified against the tuning artifact | **yes** — substitution refused |
| 7 | LR must have been selected on VANILLA | **yes** |
| 8 | Base-only cannot retroactively affect the LR | **yes** |
| 9 | Official validation measurement-only; never selection | **yes** — asserted on the trainer's call node |
| 10 | Official TEST unreachable | **yes** — no member, attribute or flag |
| 11 | Existing trainer / scorer / selector / report reused | **yes** — no second implementation |
| 12 | Provenance mismatch fails closed | **yes** — pathway is a key field |
| 13 | Only the newly required representations are extracted | **yes** — one set per (role, pathway), cached |
| 14 | No raw text persisted | **yes** |
| 15 | No significance test, p-value or threshold | **yes** — committed `PairedDiagnostic` |
| 16 | Two production changes, both additive and flagged | **yes** — `on_checkpoint` hook; recorded validation SHA |
| 17 | No scientific constant changed | **yes** — re-verified from the modules |
| 18 | No new scientific decision | **yes** |
| 19 | Guards mutation-verified | **yes** — 2 injected violations, both caught |
| 20 | No model, data, training or score executed | **yes** |
| 21 | D-B3B0-002 OPEN; Stage-2 pooling OPEN; PDF STALE; Stage-1 untouched | **yes** |
| 22 | Tests | **2 353 passed, 91 skipped**; runner file 41 passed, 2 torch-gated |
| 24 | **Review 1:** checkpoint snapshots independent of later training | **yes** — already cloned at line 546; **no production change**; proof added and mutation-verified |
| 23 | `git diff --check` clean; unstaged; no prohibited git operation | **yes** |

### Limitations

1. **The measurement has never executed.** Every test here is structural; the
   encoder pass, the 10 runs and the artifact write have not run once. The two
   snapshot-behaviour tests added in Review 1 are torch-gated and likewise
   unexecuted.
2. **`on_checkpoint` snapshots 30 state dicts per run.** Trivial for a
   `Linear(768, 3)`, and untested against real memory.
3. **The paired-init guarantee is inherited, not re-proved here.** It rests on
   `build_head` reseeding, verified on GPU in C24-1-R2 — this runner only
   ensures both arms receive the same seed.
