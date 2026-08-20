# Audit 018 — Stage-1 objective and data-path implementation

| | |
|---|---|
| **Audit id** | 018 |
| **Date (UTC)** | 2026-08-20 |
| **Scope** | Stage-1 alignment objective, three forward pathways, data preparation and collation |
| **Repository state** | `HEAD = 4729ae7`; this work uncommitted |
| **Predecessor** | [017](017-b4b-real-phobert-final-closure.md) — B4B COMPLETE |
| **Phase** | Stage-1A |
| **Type** | **Implementation.** No training, no optimizer, no weights loaded locally |
| **Revised** | 2026-08-20 — review found an implicit experiment-facing `max_length`/overflow default and a wrong PRE-TRAIN ordering. Both repaired **before** commit. Verdict unchanged; no scientific value resolved. |

---

## A. VERDICT

**PASS — STAGE-1 OBJECTIVE/DATA PATH IMPLEMENTED; REAL-MODEL DRY-RUN AND
TRAINING RUNNER STILL REQUIRED**

The three branches, the loss, the deterministic data path and the collator exist
and are tested. **1926 local tests pass, 46 skip** because torch is absent by
design.

Stage-1 is **not complete**, this is **not training-ready**, and the **PRE-TRAIN
audit has not happened**. No optimizer, scheduler, training loop or checkpointing
was written, and there is no script that could start learning.

**Two issues found in review of this audit and repaired before commit.** Neither
resolves a scientific value:

| Issue | Was | Now |
|---|---|---|
| **Implicit length policy** | `truncation: TruncationPolicy \| None = None` — omitting it silently selected *both* "unbounded" and "FAIL" | `TruncationPolicy` requires both fields and has **no no-argument form**; `prepare_example` requires `truncation` **and** `visit`. `TruncationPolicy.unbounded()` states the unbounded choice explicitly (D-S1A-003) |
| **PRE-TRAIN before the runner** | audit listed *before* the training runner exists | audit runs **after** the runner, so it can inspect the complete training path (D-S1A-007) |

The first was a real contradiction in this audit: it claimed "`max_length` cannot
be defaulted" while the API defaulted it.

---

## B. SCOPE

Implemented: the objective, the three pathways, deterministic preparation of the
clean / adapted-clean / adapted-corrupted inputs, Stage-1 batch structures and
collation, structured loss reporting, and tests.

Not implemented, deliberately: optimizer, scheduler, epoch loop, gradient
accumulation, checkpointing, corpus loading, experiment logging.

B1/B2/B3 untouched. One pre-existing test was rescoped (§V1).

---

## C. FILES CHANGED

| File | Change |
|---|---|
| `unmark/stage1/contracts.py` | **new** — torch-free enums, `ObjectiveWeights`, `TruncationPolicy` + `OverflowBehaviour`, `CorruptionRatePolicy`, `Stage1Purpose` + `Stage1RunConfig`, OPEN/LOCKED registers |
| `unmark/stage1/data.py` | **new** — preparation and collation; torch imported lazily |
| `unmark/stage1/objective.py` | **new** — cosine distance, `Stage1LossResult`, `Stage1Objective` |
| `unmark/stage1/__init__.py` | **new** — torch-free re-exports only |
| `tests/test_stage1.py` | **new** — 64 local + 14 torch-gated |
| `tests/test_adapter_contract.py` | one guard rescoped (§V1) |
| `docs/spec/decisions.md` | D-S1A-001 … D-S1A-007 |
| `docs/experiments/stage1-objective-data-path-preflight.md` | **new** — implementation record |

---

## D. PROPOSAL STAGE-1 CONTRACT

Read from §4.6 and the §5.1 lock, in the proposal's own notation:

```
L_align = D( h′(x̃ₚ), h(x) )
L_clean = D( h′(x),   h(x) )
L       = λ_a · L_align + λ_c · L_clean
```

| Item | Proposal status |
|---|---|
| `D` = cosine distance | **locked** (§4.6) |
| pooled representations only, per-token deferred | **locked** (§4.6) |
| `p ~ U(0,1)` per example, continuous | **locked** (§4.6, §5.1) |
| encoder fully frozen | **locked** (§5.1) |
| pooling = masked mean over non-special content | **locked** (D-B4A-006) |
| `λ_a`, `λ_c` | **NOT locked** — "tuned on a development split" |
| Stage-1 corpus | **NOT locked** — §5 open items, §13 item 3 |
| Stage-1 `max_length` | **not specified** — §5.3's is about Stage-2 task datasets |

**No conflict was found between the proposal and the task brief.** Every
mathematical statement checked out against §4.6.

---

## E. THREE FORWARD PATHWAYS

```
PATH R  x ─canon─► T(canon(x)) ──────────────────► frozen E_θ ─► masked mean ─► h_ref
PATH C  x ─canon─► b(x) ─► T(b(x)) ─► clean ch. ─► A_φ ─► frozen E_θ ─► masked mean ─► h′(x)
PATH K  x ─canon─► C(·,p) ─► b(x̃) ─► T(b(x̃)) ─► corrupt ch. ─► A_φ ─► frozen E_θ ─► h′(x̃ₚ)
```

`Stage1Branch` names them explicitly rather than positionally, because the class
of bug being guarded against is wiring one branch's tensors into another.

---

## F. CLEAN REFERENCE PATH

The frozen tokenizer on the clean text, the bare frozen encoder, masked pooling.
**No `b(x)`, no adapter, no tone channel, no letter channel, no corrupted text.**
Runs under `torch.no_grad()` — it is a target, `θ` is frozen, and no gradient may
flow into it. The encoder stays in eval.

**One clarification was required (D-S1A-001).** §4.6 says `h(x)` for "clean
original text `x`" without saying whether that is the raw string or its canonical
form. The reference tokenizes **`canon(x)`**, because §5.3 already locks
corruption to operate on `canon(x)` "so that placement variants and NFC/NFD forms
are the same example and receive the same noise". Using the raw string would give
two inputs the corruption engine treats as **one example** two **different**
alignment targets — making the target depend on incoming spelling variation,
which is the separate `VARIANT` axis (§6.3).

---

## G. ADAPTED CLEAN PATH

`b(x)` → `T(b(x))` → clean channels → `A_φ` → frozen encoder → masked pooling.
The base input is `b(x)`, never the marked string. Channel semantics are
unchanged from B4B: `UNMARKED` is a learned row distinct from `NA`; `NONE` is a
real letter contributor; special tokens and padding are `NA` in both channels.

---

## H. ADAPTED CORRUPTED PATH

Starts from the **same** `canon(x)`. The corrupted observation is produced by the
existing B2 engine — nothing about corruption is reimplemented. `p` is expressed
as a `CorruptionCondition` carrying that probability, so the per-unit lottery
remains B2's keyed digest.

The reference target stays `h(x)` from the original clean text. **No restored
text is fed anywhere, and no restoration term exists.**

---

## I. BASE INVARIANCE

`b(C(x)) = b(x)` is **verified per example, not assumed** (D-S1A-002). The
corrupted branch is decomposed and projected independently, then checked:

* identical base strings;
* identical authoritative content token ids;
* equal projection counts.

`padded_stage1_batch` re-checks the collated base ids and special-token masks.
Any failure raises `BaseInvarianceViolation` — **no heuristic repair**.

The reason for the extra work: sharing tensors between two branches is exactly
where a silent divergence would be invisible. The batch would look well-formed
while the two adapted representations described different strings.

---

## J. VARIABLE-LENGTH / POOLING CONTRACT

`L_ref` and `L_base` may differ and routinely will. Nothing requires them equal,
nothing truncates one to match the other, and no hidden states are compared token
by token. Each branch is pooled independently `[B, L_branch, d] → [B, d]`, and
only the pooled vectors are compared.

Pooling **reuses** `masked_mean_non_special` — no second implementation. Tests
assert no Stage-1 module defines a `masked_mean` or contains a `cumsum`
(which would signal a second position-id rule).

Zero content positions **fail loud** via `Stage1PoolingError`.

---

## K. COSINE OBJECTIVE

`representation_distance(a, b) = 1 − cos(a, b)`, over the **feature** dimension
(`dim=-1`), `[B, d] × [B, d] → [B]`. It **raises** on 3-D input, so a token-level
tensor cannot be passed by accident.

`COSINE_EPS = 1e-8` is documented as a denominator floor against a zero-norm
pooled vector — the same role `F.cosine_similarity`'s own `eps` plays. **It is
not an experiment knob** and is not exposed as configuration.

Verified: identical vectors → 0, opposite vectors → 2.

---

## L. LOSS AGGREGATION

**Mean over examples, never sum.** A summed loss would scale with batch size and
silently change the effective learning rate; a test asserts `loss_align` equals
the per-example mean and *not* the sum.

`Stage1LossResult` is structured, not one opaque scalar: `loss`, `loss_align`,
`loss_clean`, and per-example `[B]` distances for both terms. Pooled
representations are **not** retained by default — three `[B, d]` tensors held per
batch for diagnostics nobody asked for. **No raw text is stored**, since the loss
object travels into logs.

Non-finite components **fail loud** rather than propagating a `NaN` into a future
optimizer step.

---

## M. GRADIENT / FREEZING CONTRACT

| Branch | `no_grad` | detach |
|---|---|---|
| Reference | **yes** — it is a target | n/a |
| Adapted clean | **no** | **no** |
| Adapted corrupted | **no** | **no** |

Tests check this **structurally**, walking `ast.With` nodes: the reference method
must contain a `no_grad` block; `adapted_representation` and `forward` must not.
Only `to_dict` may detach, for logging.

A torch-gated test runs one backward and confirms every adapter parameter
receives a finite gradient with at least one nonzero, while **no frozen encoder
parameter receives any** and all stay `requires_grad=False`.

`Stage1Objective.train()` delegates to `UnmarkEncoder`, so the frozen encoder
stays in eval even two levels down (D-B4B-004). **One frozen `θ` serves all three
branches** — no second pretrained model is loaded for the reference.

`position_ids` is omitted so the wrapper derives and enforces the authoritative
values (D-B4B-002); Stage-1 does not reimplement position semantics.

---

## N. STAGE-1 DATA STRUCTURES

`Stage1Example` (clean text + **stable** `sample_id`, empty rejected),
`PreparedStage1Example`, and the batch dict. Field names are deliberately
unambiguous — `reference_input_ids` vs `base_input_ids`, `clean_tone_ids` vs
`corrupt_tone_ids` — so a wiring error is visually obvious. There is no bare
`input_ids`, `mask` or `tone` anywhere.

The two adapted branches **share** the base ids and special-token mask rather
than duplicating identical tensors — after §I proves they are identical.

**No downstream label field exists.** A test walks every dataclass annotation in
the package for `label`/`labels`/`target_label`/`y`/`gold`/`class_id`, and no
Stage-1 module imports downstream or baseline code.

---

## O. COLLATION / MASKING

**Two independent padding domains.** `L_ref_max` and `L_base_max` are computed
separately; a test with deliberately unequal branches asserts they differ and
that neither was padded to the other's width. Padding both to a common maximum
would add positions that exist only to match the other branch — and pooling
excludes padding, so it would be waste at best and a masking bug at worst.

Letter contributors stay `[B, L_base, K]` with `K` the batch maximum. Padding is
marked **both** unattended and special, so it is excluded twice over.

---

## P. CORRUPTION INTEGRATION

The B2 engine is authoritative; nothing about corruption is reimplemented.

`p ~ U(0,1)` per example is **locked** and implemented. The draw is a BLAKE2b
keyed digest over `(schema_version, seed, sample_id, visit)` — **no
module-global RNG**, because `random.uniform` would make the same batch differ
between processes and could not be reproduced from a run record. A test asserts
the module neither imports `random` nor calls `seed`/`uniform`/`random`.

`visit` is an **explicit argument with no default schedule attached**: the redraw
question is OPEN, and attaching a schedule would decide it by implementation
default.

A useful property, verified: B2 thresholds a per-unit score against `p`, so a
syllable corrupted at some `p` is also corrupted at any larger `p`. Corruption is
monotone in the rate, which is the intended "fraction of syllables" semantics.

Reproducibility is tested: the same `(seed, sample_id, visit)` gives the same rate
and the same corrupted text, and different `sample_id`s give different rates — so
reordering a corpus cannot change any example's corruption.

---

## Q. OPEN SCIENTIFIC VALUES

**The rule applied throughout: an API default is a scientific decision if it can
reach an experiment.**

| Item | Status |
|---|---|
| Corruption severity distribution | **LOCKED + IMPLEMENTED** — `p ~ U(0,1)` continuous |
| Distance function | **LOCKED + IMPLEMENTED** — cosine |
| Representation level | **LOCKED + IMPLEMENTED** — pooled only |
| Pooling | **LOCKED + IMPLEMENTED** — masked mean, non-special |
| Encoder freezing | **LOCKED + IMPLEMENTED** |
| `lambda_align` | **OPEN** — required argument, no default |
| `lambda_clean` | **OPEN** — required argument, no default |
| Stage-1 corpus | **OPEN** — not chosen, not loaded |
| `max_length` | **OPEN** — **no experiment-facing default**; `TruncationPolicy` has no no-argument form and `prepare_example` requires the policy |
| Truncation behaviour | **OPEN** value; interface decided — `FAIL`/`SKIP`/`NOT_APPLICABLE`, **never truncate**, neither `FAIL` nor `SKIP` selectable by omission |
| Corruption redraw schedule | **OPEN** — `visit` explicit |
| Optional letter-dropout rate | **OPEN** — not enabled |
| Stage-1 seed | **OPEN** — required explicitly |
| Batch size | **OPEN** |
| Optimizer / learning rate / epochs / warmup / grad accumulation / checkpoint selection | **OPEN — not implemented in this phase** |
| Backbone finalisation | **OPEN** — D-B3B0-002 |

`OPEN_STAGE1_VALUES` is the machine-readable register; `require_resolved(name)`
raises for any of them. Tests inspect the **signatures** to assert that
`truncation`, `visit`, `corruption_policy`, `purpose` and both lambdas have no
omitted defaults, and that no corpus is referenced.

**The existence of a config field does not mean a value is decided.**

### Diagnostic-only values (D-S1A-006)

`Stage1RunConfig` carries a required `Stage1Purpose`, mirroring B2's
`CorruptionPurpose`:

* **`DIAGNOSTIC`** — explicit wiring values for exercising a forward/backward
  path, no optimizer, no update. The upcoming dry run may pass
  `lambda_align = 1.0`, `lambda_clean = 1.0` and an explicit `max_length` on this
  footing. **It resolves nothing.** `to_dict()` stamps `diagnostic_only: true`
  and `values_are_scientific: false` into the run artifact.
* **`SCIENTIFIC`** — **cannot be constructed at all** until `resolved_values`
  names every entry of `SCIENTIFIC_REQUIRED_VALUES`. It raises
  `UnresolvedStage1Value` listing what is missing.

That is the strongest available guarantee that a diagnostic number does not drift
into a training run: **the scientific configuration does not exist as an object
yet.** An explicit diagnostic `max_length` does not lock the future training
value.

---

## R. DECISION LOG / PROPOSAL CONSISTENCY

Seven entries: **D-S1A-001** (reference tokenizes `canon(x)`), **D-S1A-002**
(base invariance verified, not assumed), **D-S1A-003** (no truncation, and **no
implicit length policy**), **D-S1A-004** (keyed-digest rate, explicit `visit`),
**D-S1A-005** (the OPEN register), **D-S1A-006** (diagnostic values cannot become
scientific configuration), **D-S1A-007** (the PRE-TRAIN audit runs after the
training runner exists).

**D-S1A-001 is not reopened.** I re-read the current editable proposal: §5.3
states corruption operates on `canon(x)`, and **D-B2-004** records that
clarification with its rationale — `hòa` and `hoà` must not be different examples
with different identities. The compiled PDF being stale with respect to that
change is not a contradiction in the source.

**Proposal updated: NO.** No scientific mismatch was exposed; §4.6 is implemented
as written and D-S1A-001 is the only reading consistent with §5.3. **PDF stale:
YES**, carried over from the earlier v1.4 source changes.

B4B decisions are unchanged. **D-B3B0-002 remains OPEN.**

---

## S. LOCAL TEST EVIDENCE

```
.venv/bin/python -m pytest -q -p no:cacheprovider
1938 passed, 46 skipped in 8.43s
```

Baseline before this task: 1874 passed, 32 skipped. `tests/test_stage1.py` holds
**64 local + 14 torch-gated**.

Torch-free tests genuinely exercise the data path — preparation is pure Python,
so branch separation, base invariance, channel semantics, padding domains,
corruption reproducibility and every OPEN-value guard run for real rather than
being inspected.

Static guards, all AST-based: no optimizer or training call; no second pooling or
position-id implementation; no hardcoded `768` or `pad_token_id = 1`; no
VnCoreNLP; no family-wide RoBERTa permission; no downstream label field; no
token-level hidden-state matching; `no_grad` only on the reference branch;
nothing detached on the loss path.

Torch-gated: distance shape and extremes; 3-D input rejected; scalar components
combining as `λ_a·L_align + λ_c·L_clean`; mean not sum; unequal `L_ref`/`L_base`;
reference not requiring grad while both adapted branches do; **channels actually
influencing the two adapted forwards** (if they were ignored, the branches would
coincide); gradients reaching the adapter and not the encoder; encoder eval
across mode transitions; zero-content and missing-field failing loud; non-finite
components failing loud; the result dict carrying no raw text; and end-to-end
collation.

**Added by the policy-explicitness repair (12 tests).** `TruncationPolicy()` and
`TruncationPolicy(max_length=...)` alone both refused; **signature inspection**
proving `truncation`, `visit`, `corruption_policy` and `purpose` have no omitted
defaults; explicit `unbounded()` behaving as a statement; explicit numeric
`max_length` accepted; `SKIP` and `FAIL` each unreachable by omission; all three
inconsistent length/overflow combinations failing loud; `OverflowBehaviour`
having no `TRUNCATE` member and no module defining `truncate`; a `SCIENTIFIC`
config refusing to construct; a `DIAGNOSTIC` config constructing and being
labelled; `resolved_values` rejecting names outside the OPEN register; and
`SCIENTIFIC_REQUIRED_VALUES` being a subset of it.

**Two of my first-draft static tests matched prose in the modules' own
docstrings** — a docstring saying the adapted branch is "**Not** under
`no_grad`", and one saying `max_length` is about "Stage-2 task datasets". Both
were rewritten as structural AST checks. This is the same failure mode as in
earlier phases, and raw-substring assertions about source keep producing it.

---

## T. TRAINING-PROHIBITION CHECK

Swept `unmark/stage1/*.py` structurally:

| Module | Banned calls | Banned imports |
|---|---|---|
| `contracts.py` | none | none |
| `data.py` | none | none |
| `objective.py` | none | none |
| `__init__.py` | none | none |

No `step`, `zero_grad`, `backward`, `save`, `save_pretrained`; no `optim`,
`datasets`, `wandb`, `tensorboard`; no `AdamW`, `SGD`, `lr_scheduler`,
`GradScaler`. **No `train.py` or runner exists** — there is no script that could
start learning. The only `.backward()` in the repository's Stage-1 surface is in
a torch-gated test, with no optimizer and no parameter update.

**The runner is deliberately absent, and its absence is now on the critical
path**: D-S1A-007 places the PRE-TRAIN audit *after* the runner is written, so
the audit can inspect it. Writing the runner is step 3 of §W, not this task.

---

## U. BLOCKING ISSUES

**None for this task.** Two block **scientific Stage-1 training** — and only
that:

1. **The OPEN scientific values in §Q**, above all `lambda_align` and
   `lambda_clean`. The code refuses to run without them rather than defaulting,
   and a `SCIENTIFIC` run configuration cannot be constructed at all.
2. **The PRE-TRAIN audit has not happened**, and cannot yet: the training runner
   it must inspect does not exist (D-S1A-007).

**What they do *not* block.** The real-model Stage-1 integration dry run (§W step
1) may proceed now, with explicit **diagnostic-only** values, no optimizer and no
parameter update. Nothing about that run resolves an OPEN value, and its artifact
is stamped `purpose = DIAGNOSTIC`.

The earlier wording said these blocked "Stage-1 execution", which was too broad —
it would have read as forbidding the dry run that is the immediate next step.

---

## V. NON-BLOCKING ISSUES

1. **One pre-existing guard was rescoped.**
   `test_no_stale_current_state_claims_in_b4a_docs` banned the phrase
   "OPEN — RESEARCHER DECISION REQUIRED" from the whole of `decisions.md`. That
   was over-broad: the guard's purpose is that no **B4A** item reads as currently
   open, and `decisions.md` accumulates every phase, so a new phase's legitimate
   open item tripped it. It now scans only the B4A block. I verified it still
   fails on a reintroduced B4A regression before keeping the change.

2. **`_regions` is duplicated.** The same "syllable spans plus the gaps" helper
   now exists in `unmark/stage1/data.py` and in both B3B probe scripts. I did not
   edit the probes — they are frozen evidence-producing artifacts, and changing
   them carries more risk than the duplication. Extraction into a shared
   alignment helper is the obvious follow-up.

3. **The local dry run used a stub tokenizer.** It exercises the contracts, not
   real BPE behaviour. `L_ref ≠ L_base` had to be forced with a stub option,
   because the naive stub tokenizes marked and stripped text identically — a real
   tokenizer will not.

4. **The collator is single-batch.** Bucketing, shuffling and epoch iteration do
   not exist; those belong with the training runner.

5. **Proposal PDF remains stale** (v1.4 source).

---

## W. NEXT VALIDATION STEP

**Corrected ordering (D-S1A-007).** The earlier version of this section put the
PRE-TRAIN audit *before* the training runner, which would have audited the
repository with the training code missing — everything except the part that
trains.

1. **Real-model Stage-1 dry run** on Colab — real PhoBERT, all three branches,
   one diagnostic backward permitted, **no optimizer and no parameter update**,
   `purpose = DIAGNOSTIC`. Confirms finite loss components, correct branch shapes
   with `L_ref ≠ L_base`, and gradient routing into `A_φ` with none into `θ`.
2. **Resolve the scientific values** needed to define the runner and its run
   configuration (§Q).
3. **Implement the Stage-1 training runner.** Optimizer, scheduler and
   checkpointing may exist *in code*; **no scientific training is run**, and no
   `optimizer.step()` is executed as an experiment.
4. **Run the mandatory repository-wide proposal-vs-code PRE-TRAIN audit**, which
   inspects the runner built in step 3.
5. **Only if that audit PASSes:** the first scientific Stage-1 training run.

**What step 4 must be able to see**, and why it must come after step 3: full
proposal vs repository; the Stage-1 objective; corpus and split discipline;
corruption sampling and redraw schedule; stable sample identity and seeds; lambda
values; `max_length` and overflow policy; batch size; **optimizer**; learning
rate; **scheduler/warmup**; epochs or steps; **gradient accumulation**; mixed
precision if used; the frozen/trainable partition; the encoder eval invariant;
**checkpoint save/resume semantics and the selection criterion**; leakage risks;
provenance; reproducibility; configs versus docs versus tests; and the
baseline/protocol commitments that must be fixed before any result is seen. The
emphasised items do not exist until step 3.

---

## X. GIT STATE

`HEAD = 4729ae7`.

```
 M docs/spec/decisions.md
 M tests/test_adapter_contract.py
?? docs/audits/018-stage1-objective-and-data-path-implementation.md
?? docs/experiments/stage1-objective-data-path-preflight.md
?? tests/test_stage1.py
?? unmark/stage1/
```

`git diff --check` is clean. Every change is left **unstaged**. No `add`, `commit`, `push`, `tag`, `stash`,
`reset` or `restore` was run. **No ML packages were installed locally; no model
weights were downloaded or loaded; no network was accessed; nothing was trained;
no optimizer exists.**

```text
AUDIT FILE WRITTEN: docs/audits/018-stage1-objective-and-data-path-implementation.md
OTHER AUDIT MODIFICATIONS MADE: NO
COMMIT CREATED: NO
```
