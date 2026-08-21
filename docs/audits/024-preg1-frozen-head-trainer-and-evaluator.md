# Audit 024 — pre-G1 frozen-encoder head trainer and evaluator

| | |
|---|---|
| **Audit id** | 024 |
| **Created (UTC)** | 2026-08-21 |
| **Baseline HEAD** | `57f70e373d46e919b754e24dd875f6d39e01e35c` |
| **Scope** | Implement the frozen-encoder linear-head trainer, evaluator, LR selector, paired measurement structure and representation cache. **Run none of it.** |
| **Predecessors** | [021](021-pre-g1-dataset-profile-and-protocol-precommit.md), [022](022-uit-vsfc-real-data-profile-integrity-closure.md), [023](023-pre-g1-internal-split-materializer-and-fail-closed-contract.md) |
| **Phase** | pre-G1 |
| **Type** | **Implementation + audit.** No real training, no LR sweep, no downstream score, no model download |
| **Revision 1b** | **2026-08-21** — **status-block consistency repair.** The final status block still asserted `PAIRED INITIALISATION: BIT-IDENTICAL…`, `ENCODER: FROZEN / EVAL / NO_GRAD / FP32`, `REPRESENTATION CACHE: FAIL-CLOSED…` and `EPOCHS: 30 / NO EARLY STOPPING` as unqualified facts, contradicting the evidence levels Revision 1a had just written into §§E, F, I and N. All four are now qualified; genuinely local-verified lines are marked `LOCAL-PASS`; §J's coverage list is tagged per group. **Documentation only — no code, test, constant or scientific value changed.** |
| **Revision 1a** | **2026-08-21** — **status-consistency repair.** §I claimed "each answer was checked by running the code" when two of the seventeen (Q4 paired init, Q5 frozen encoder) rest **only** on torch-gated tests that have never executed; every answer now carries an explicit LOCAL / PARTIAL / PENDING / INSPECTION status. Self-audit rows whose only proof is among the 28 unexecuted tests are corrected to *implementation present; test authored; runtime verification PENDING*. §O replaced with the real five-step sequence. **No code, constant or scientific value changed.** |
| **Revision 1** | **2026-08-21** — **fail-closed repair + honest test accounting.** (A) The local outcome was summarised as "87 passed" when 21 of those tests are torch-gated and did **not** execute; every such statement is corrected, and the torch-gated tests are now marked **PENDING** rather than passing. (B) Representations were passed as bare tensors beside a free `dev_role` argument, so the role was a *claim about* a tensor rather than a *property of* it — §M admitted this and the audit still answered Q1 with an unqualified "No". Repaired: `BoundRepresentations` carries the tensor with its `RepresentationKey`, the role comes from provenance, and **no role argument exists anywhere to contradict it**. |

---

## A. VERDICT

**IMPLEMENTATION PASS — LOCAL CONTRACT TESTS PASS; COLAB TORCH / REAL-MODEL /
REAL-DATA BOUNDARY VERIFICATION PENDING; NO DIAGNOSTIC RUN, NO DOWNSTREAM
SCORE**

**This implementation is not scientifically closed.** Three things remain
unverified — the torch runtime, the real-model integration and the real-data
boundary dry-run — and §O sets out the order they must happen in.

The mechanism for the pre-G1 Vanilla-vs-Base-only burden diagnostic exists. Its
contract logic is verified locally. **Nothing was run.** No learning rate was
tuned, no head was trained on real data, no score exists, and official TEST was
not loaded — it is not reachable from this code at all.

### Test accounting, stated exactly

| | |
|---|---|
| New tests **authored** | **134** |
| New tests **executed locally** | **106 passed** |
| New tests **skipped locally** (torch absent) | **28 — NOT passed, PENDING** |
| Full local suite | **2305 passed, 84 skipped** |
| Baseline at Audit 023 | 2199 passed, 56 skipped |

**The 28 torch-gated tests have not been executed anywhere.** They are not
claimed as passing, and this audit stays **provisional with respect to them**.
They cover head initialisation, the optimiser parameter groups, frozen-encoder
extraction, the cache round-trip and the end-to-end training path — including
the paired-initialisation guarantee, which is among the load-bearing claims
here. Colab verification will revise this audit **in place**.

This is not a failure of the milestone. It is where the ML-free local
environment stops.

**No new scientific decision was created**, and §L explains why none is
warranted — including for the Revision-1 repair, which makes the implementation
enforce an already-recorded boundary rather than changing one.

| Boundary | State |
|---|---|
| D-B3B0-002 | **OPEN** — nothing here locks the backbone |
| Final Stage-2 pooling | **OPEN** — `<s>` is scoped to this diagnostic only |
| Official TEST | **SEALED**, and unrepresentable in this module |
| Official validation | measurement-dev; cannot enter any selection path |
| Head trainer executed on real data | **NO** |
| Downstream score | **NONE** |
| Stage-1 training / HPO | **NOT RUN** |
| Compiled proposal PDF | **STALE** |

---

## B. WHAT THIS MEASURES, AND WHAT IT IS NOT

The diagnostic answers exactly one question: **how much does stripping
Vietnamese orthographic marks cost, on its own, in front of a frozen encoder?**

```
Vanilla     canon(x)          -> tokenizer -> frozen encoder -> <s> -> Linear(d,3)
Base-only   b(canon(x))       -> tokenizer -> frozen encoder -> <s> -> Linear(d,3)
```

Same tokenizer, same frozen encoder, same pooling, same head protocol, same
seeds, same learning rate. **The only difference is the stripping step.** That is
the entire design: anything else that differed would become a rival explanation
for the gap.

**This is not UNMARK.** No tone channel, no letter channel, no adapter, no
RESTORE, no ALIGN, no Stage-1 adaptation appears anywhere in the new code, and a
test asserts the module imports nothing from `unmark.modeling`, `unmark.stage1`
or `unmark.alignment`. It measures the burden UNMARK would later have to
recover. It says nothing about whether UNMARK recovers it.

**No word segmentation** in either pathway (RAW_BASE). A test asserts no
segmenter is importable from the module and that no pathway output contains the
underscore-joined compounds a segmenter would produce.

---

## C. WHAT WAS BUILT

| File | Role |
|---|---|
| `unmark/evaluation/preg1_head.py` | the diagnostic: roles, membership guards, representation cache, frozen extraction, head, optimiser, trainer, checkpoint selector, LR selector, paired report |
| `scripts/preg1_head_diagnostic.py` | Colab CLI — `tune` and `measure` subcommands |
| `tests/test_preg1_head.py` | 134 tests (106 pass locally, 28 torch-gated and pending) |
| `unmark/evaluation/pathways.py` | **one-line change**: `encode_split` gained an explicit `padding` parameter, default unchanged |

### Reuse rather than a parallel protocol

Every scientific constant is **imported** from `preg1_protocol`: LR grid, tuning
and measurement seeds, epochs, batch size, AdamW betas/eps/amsgrad, decay
groups, `max_length`, truncation, padding, pooling, checkpoint eligibility and
the checkpoint / aggregation rules. A test asserts by AST that the module
contains **no numeric literal** equal to any locked value — `30`, `128`, `256`,
`0.01`, `1e-8`, `0.9`, `0.999`, or any of the eight seeds.

`pathway_text`, `macro_f1` and `accuracy` are reused from the existing harness
rather than reimplemented, so the diagnostic cannot disagree with the pipeline
it is measuring.

**The one change to committed code** is additive and default-preserving:
`encode_split` previously hard-coded `padding=True` (pad to longest in batch).
The pre-G1 contract requires `padding="max_length"` because the representation
cache is keyed on a fixed shape. Rather than duplicate tokenisation or silently
change the Stage-2 default, `padding` became a parameter whose default is the
previous behaviour — so neither caller inherits the other's choice.

### `require_protocol_settings()`

The trainer hard-codes several behaviours because their values are locked: no
clipping, no accumulation, no dropout, constant LR. Implementing a knob for a
locked value invites someone to turn it. This function is the link back: if
`preg1_protocol` ever changes `GRADIENT_CLIPPING`, `GRADIENT_ACCUMULATION_STEPS`,
`HEAD_DROPOUT` or `PADDING`, the trainer stops matching the spec and raises —
instead of the mismatch surviving as a comment that used to be true. It is
called at the top of every training run.

---

## D. THE ROLE MODEL — OFFICIAL TEST IS UNNAMEABLE

```python
class Preg1Role(Enum):
    PROTOCOL_TRAIN      = "protocol-train"       # head training
    PROTOCOL_DEV        = "protocol-dev"         # checkpoint + LR selection
    OFFICIAL_VALIDATION = "official-validation"  # measurement, AFTER LR freeze
```

**There is no `OFFICIAL_TEST` member.** Official test is not forbidden by a
check that could be bypassed — it *cannot be named*, so no function signature can
accept it, no CLI flag can carry it and no code path can reach it. A test asserts
the member set is exactly these three and that `Preg1Role("official-test")`
raises.

Two derived properties carry the discipline: `may_train_head` is true only for
`PROTOCOL_TRAIN`, and `may_select` only for `PROTOCOL_DEV`.

### The Revision-1 repair — provenance-bound representations

**The weakness.** The first implementation passed a bare tensor beside a free
`dev_role` argument. The role was therefore a *claim about* the tensor, not a
*property of* it: a caller could hand official-validation features to checkpoint
selection while declaring `PROTOCOL_DEV`, and every guard would pass. §M
recorded that honestly — and the audit still answered Q1 with an unqualified
"No", which the evidence did not support. **Documenting a hole is not closing
it.**

**The repair.** `BoundRepresentations` carries the values together with the
`RepresentationKey` that was validated when they were produced or loaded:

```python
@dataclass(frozen=True)
class BoundRepresentations:
    values: "Tensor"
    key: RepresentationKey        # role, pathway, source identity, ordered ids,
                                  # tokenizer, revision, max_length, pooling, dtype
```

- **The role comes from `key.role`.** `train_head` no longer has a `dev_role`
  parameter, nor a `pathway` parameter — both are read from provenance. A test
  asserts by signature inspection that no such parameter exists, so the
  contradiction is not merely rejected, it is **inexpressible**.
- `require_training_roles(train, dev)` demands `PROTOCOL_TRAIN` and
  `PROTOCOL_DEV` respectively, plus matching pathway and matching geometry
  (tokenizer, revision, `max_length`, pooling, truncation, padding, dtype,
  hidden size, schema).
- `score_measurement` demands `OFFICIAL_VALIDATION` and refuses protocol-dev —
  the mirror-image guard: reporting the headline number on the set that chose
  the checkpoint would report the selection, not the pathway.
- Rebinding is not a loophole. `BoundRepresentations.__post_init__` requires the
  key to actually describe the values (`count`, `hidden_size`, `dtype`,
  `requires_grad`), so a contradictory key is rejected at construction.
- `RepresentationCache.load` now **returns** a `BoundRepresentations` carrying
  the exact key it validated, and `extract_bound_representations` produces the
  same shape — fresh and cached representations are interchangeable at the type
  level, and neither reaches the trainer without provenance.

Validation is duck-typed on `.shape`, `.dtype` and `.requires_grad`, so the
binding contract is **testable without torch**. That is why 20 of the 22 new
binding tests run in the ML-free environment rather than waiting for Colab.

The three earlier layers remain: `HeadRun` records `scored_on` (now taken from
`dev.role`), refuses construction if that role may not select, and `LrCandidate`
re-checks every run it aggregates — a run smuggled past the first two by
bypassing `__post_init__` is still caught by the third, and a test does exactly
that.

---

## E. DETERMINISM — STATED PRECISELY

**Intended, with evidence status per property** — the torch-dependent ones are
implemented and tested, but those tests have **not executed**:

| Property | How | Evidence |
|---|---|---|
| same seed → bit-identical head parameters | `build_head` resets the RNG from the seed **immediately before** initialising, and applies Xavier-uniform / zeros **explicitly** rather than relying on `nn.Linear`'s default (a Kaiming variant whose exact form has changed across PyTorch versions) | **PENDING** |
| Vanilla and Base-only start identically under a paired seed | the reset makes this hold **regardless of which arm ran first or what consumed RNG in between** — the test advances the global RNG by 1000 draws between the arms and still requires bitwise equality | **PENDING** |
| different seed → different initialisation | | **PENDING** |
| deterministic batch order | `deterministic_batches` uses its own `random.Random(seed)`, never the ambient global RNG; a test reseeds `random` between calls and requires identical output | **LOCAL** |
| batch order varies by epoch but is identical across arms | derived from `(seed, epoch)` | **LOCAL** |
| checkpoint tie-break | total order via `min(..., key=(-f1, -acc, epoch))` | **LOCAL** |
| LR tie-break | total order via `min(..., key=(-mean_f1, -mean_acc, sd, lr))` | **LOCAL** |
| no dependence on mapping insertion order | both selectors sort on **content**; tested with reversed inputs | **LOCAL** |

**The three head-initialisation rows are the ones that matter most and the ones
still unproven.** They are pure torch, so no amount of duck-typing moves them
into the local suite.

**Not guaranteed, and not claimed:** bitwise identity of trained parameters or
scores across different hardware, CUDA/cuBLAS versions or PyTorch builds.
Floating-point reduction order on a GPU is not fixed by a seed. A run reproduced
on different hardware should be expected to agree **closely, not exactly**, and
`DETERMINISM_SCOPE` says so in the artifact itself rather than leaving a reader
to assume more.

---

## F. THE REPRESENTATION CACHE

A cache is a correctness hazard here, not a convenience: **reusing Vanilla
vectors as Base-only would produce a burden of exactly zero and look like a
finding.** So `RepresentationKey` binds every input that could change the tensor
— dataset, version, task, role, pathway, source-file identity, ordered-id digest,
tokenizer id, model revision, `max_length`, truncation, padding, pooling, dtype,
hidden size, count, schema version — and comparison is **exact on every field**.
A mismatch raises; it never recomputes silently.

Sample identity travels as a digest of the **ordered** ids, so the key pins
*which examples in which order* without storing corpus text. Reordering changes
the digest, which is the point: row *i* must stay row *i*.

**No raw text.** A test asserts the serialised metadata contains no
Vietnamese-marked character and no `text` key.

**Fourteen incompatible-metadata cases are tested individually and executed
locally**, including the pathway swap — the key comparison is pure Python.

FP32 is enforced on both save and load, and the stored shape must match the
key's `(count, hidden_size)` — but those are **torch-gated and pending**, along
with the round-trip, the role-reinterpretation refusals on disk and the bound
return value. The provenance *logic* is verified here; the *file* path is not.

The tensor file is an experiment resource and is **never committed to git**.

---

## G. THE MEASUREMENT REPORT

Per-seed raw scores for both arms, per-seed deltas (Vanilla minus Base-only),
means, and **sample** standard deviations — `statistics.stdev`, not `pstdev`.
Three tuning seeds are a *sample*, not a population; the population form would
understate the spread by √(2/3), and that spread is exactly what the third LR
tie-break turns on. A test pins the distinction numerically.

**No significance machinery.** A test walks every key of the serialised report
and asserts none of `p_value`, `significant`, `threshold`, `confidence_interval`,
`test_statistic`, `verdict` or `passed` appears. Five seeds cannot support a
hypothesis test, and an acceptance threshold invented after seeing a burden
would be decorative. The report says this in `NO_SIGNIFICANCE_TEST`.

`PairedDiagnostic` refuses construction unless the seeds are exactly the five
precommitted measurement seeds and the reporting role is official validation.

---

## H. THE CLI BOUNDARY

Two subcommands in protocol order: `tune` (Vanilla only, on protocol-dev) and
`measure` (paired, on official validation).

- **No `--test` or `--official-test` flag** — the role does not exist.
- **No `--seeds`, `--grid`, `--epochs`, `--batch-size` or `--max-length` flag** —
  a command-line override of a precommitted constant is the hole the protocol
  exists to close.
- **`measure` requires `--frozen-lr`**, validated against the grid and forced to
  `selected_on = VANILLA`. By the time Base-only is encoded at all, the LR is
  already a value in a file, so it cannot influence the choice.

A test parses the CLI's `add_argument` calls by AST and asserts these
properties structurally rather than reading the help text.

---

## I. THE SEVENTEEN AUDIT QUESTIONS

**Evidence status is stated per answer.** Three levels, and they are not
interchangeable:

| Level | Meaning |
|---|---|
| **LOCAL** | proved by torch-free tests that **executed** in this environment |
| **PARTIAL** | the guard logic executed locally; the torch runtime path around it is **authored but unexecuted** |
| **PENDING** | the only executable proof is torch-gated and **has not run anywhere** |
| **INSPECTION** | established by reading or grepping the code, with no executable proof |

An earlier revision introduced this table with "each answer was checked by
running the code". That was true of most answers and false of several, and the
distinction is exactly what an audit is for.

| # | Question | Answer | Evidence |
|---|---|---|---|
| 1 | Can official validation influence LR selection in any code path? | **No, on the supported path — and now for a reason that holds.** *(Revision 1: the earlier "No" was not justified; see §D.)* The role is read from the representation's `RepresentationKey`; `train_head` has **no role parameter** to override it; `require_training_roles` demands `PROTOCOL_DEV`; `HeadRun.scored_on` records what was used; `LrCandidate` re-checks it; a contradictory key is rejected at construction. | **LOCAL** — the binding guards, the signature check and both selection layers are torch-free and executed |
| 2 | Can official TEST influence anything? | **No**, precisely: (a) `Preg1Role` has no `OFFICIAL_TEST` member, so it is unnameable in any signature; (b) the CLI exposes no test flag; (c) representations carry a validated role and source identity, so a supported-path reinterpretation fails closed. **Not claimed:** protection against deliberate tampering outside the validated boundary — someone who fabricates a key *and* a matching tensor is beyond what any in-process check can detect. | **LOCAL** — enum membership and CLI-AST tests executed |
| 3 | Can Base-only influence primary LR selection? | **No.** `LrCandidate` raises `SplitLeakage` on any non-VANILLA run; `FrozenLearningRate` refuses `selected_on != VANILLA`. | **LOCAL** |
| 4 | Can the arms start from different parameters for the same paired seed? | **The implementation reseeds from the seed immediately before initialising, so it should not** — but this is **not yet verified at runtime**. `test_paired_seed_gives_identical_starts_for_both_pathways` (which advances the global RNG by 1000 draws between the arms) is **authored and torch-gated; it has not executed anywhere**. | **PENDING** — one of the 28 |
| 5 | Can the encoder receive gradients or leave eval mode? | **The implementation checks every parameter and the mode before extraction, runs under `no_grad` and returns a detached FP32 tensor** — but this is **not yet verified at runtime**. The four frozen-encoder tests and the real-backward test are **authored and torch-gated; none has executed**. | **PENDING** — five of the 28 |
| 6 | Can a Vanilla cache be reused as Base-only? | **No.** Pathway is a key field and comparison is exact. | **PARTIAL** — the key-level refusal executed locally; the cache **file** round-trip refusal is torch-gated and pending |
| 7 | Can stale cache metadata silently pass? | **No.** Every field compared; missing keys, unknown enum values and unparseable JSON all raise. | **LOCAL** for the 14 field-mismatch cases and the malformed-metadata cases; **PENDING** for the on-disk load path |
| 8 | Can checkpoint tie-breaking depend on iteration accidents? | **No.** `min()` over an explicit key. A scan with `>` keeps the first maximum and `>=` the last — the rule would be decided by iteration order. | **LOCAL** — tested with reversed input |
| 9 | Can LR tie-breaking depend on mapping insertion order? | **No.** Sort key over candidate content. | **LOCAL** — tested forward and reversed |
| 10 | Is sample standard deviation used? | **Yes.** `statistics.stdev`; asserted to differ from `pstdev`. | **LOCAL** |
| 11 | Is there any early stopping? | **No.** No patience parameter exists anywhere, and `require_full_schedule` rejects a truncated score list. | **LOCAL** for the contract guard; the training loop itself is **PENDING** |
| 12 | Are all 30 epochs guaranteed? | **Yes** by construction: the loop is `range(1, epochs+1)` with no `break`, and `HeadRun` refuses construction unless epochs 1..30 are all present. | **LOCAL** for the `HeadRun` guard; the loop's actual execution is **PENDING** |
| 13 | Is `<s>` pooling scoped to this diagnostic only? | **Yes.** The function is named `PREG1_ONLY_first_token_pool` and `PREG1_POOLING_SCOPE` states final Stage-2 pooling remains **OPEN** (D-G1-005). | **LOCAL** for the naming and scope string; the pooling **arithmetic** is **PENDING** |
| 14 | Does anything close D-B3B0-002? | **No.** Nothing in the new code references or resolves it. The checkpoint is used as a **probe** revision. | **INSPECTION** — grep over the new modules |
| 15 | Is official TEST still sealed? | **Yes**, and structurally unreachable from this module. | **LOCAL** |
| 16 | Has any downstream score been produced? | **No.** | **fact of this task** |
| 17 | Has any real downstream training been run? | **No.** | **fact of this task** |

**Summary of evidence:** 8 answers are locally verified outright, 5 are partially
verified with a pending runtime path, **2 (Q4, Q5) are entirely pending**, 1 rests
on inspection, and 2 are statements about what this task did.

---

## J. TESTS

**134 tests** in `tests/test_preg1_head.py`: **106 pass locally**, **28 are
torch-gated and did not execute** — they are pending Colab verification and are
**not** counted as passing. No downloads, no network, no GPU, no real UIT-VSFC,
no real split artifacts.

Deliberately, **the contract logic is torch-free** — selection rules, membership
guards, cache provenance, report shape and the boundary checks are pure Python,
so the guarantees that matter most are verified in the ML-free environment
rather than deferred to a machine that may never run them.

Coverage by group — **`[LOCAL]` marks groups that executed, `[PENDING]` marks
groups that are authored and have not run**:

- `[LOCAL]` membership loading and every failure mode (duplicate, overlapping,
  incomplete, unknown id, malformed manifest, wrong schema, pinned digest
  mismatch);
- `[LOCAL]` pathway behaviour and the absence of segmentation;
- `[PENDING]` frozen encoder, eval mode, `no_grad`, FP32, real-backward
  isolation;
- `[PENDING]` head shape, Xavier bound, zero bias, seed determinism, paired
  identity, no dropout;
- `[PENDING]` AdamW settings, two decay groups, constant LR;
- `[LOCAL]` no scheduler / clipping / accumulation (AST);
- `[LOCAL]` epoch-0 ineligibility, full-schedule guard, all three checkpoint
  tie-breaks, macro-F1 edge cases, deterministic batching;
  `[PENDING]` the 30-epoch loop actually running;
- `[LOCAL]` LR grid and seed enforcement, all four aggregation tie-breaks,
  sample-vs-population SD, order invariance;
- `[LOCAL]` the five measurement seeds, delta direction, report contents,
  absence of significance fields;
- `[LOCAL]` fourteen cache-incompatibility cases and metadata hygiene;
  `[PENDING]` the on-disk save/load round-trip.

**Revision 1 added 22 provenance-binding tests**, 20 of which run locally:
training role must be `PROTOCOL_TRAIN`; checkpoint role must be `PROTOCOL_DEV`;
official validation refused for both training and selection; protocol-dev and
protocol-train both refused for the final measurement; **no role parameter
exists** (asserted by signature inspection); a tensor cannot be rebound under a
contradictory key; pathway mismatch; six geometry-mismatch cases; hidden-size
mismatch; cache load returns a bound object preserving the validated key;
protocol-dev↔official-validation cache reinterpretation fails both ways; fresh
extraction returns the same bound shape; and the full supported
train→select→measure path on synthetic data.

### Two test defects I found and fixed during the task

1. **A decorator on a helper.** `@requires_torch` was applied to
   `make_fake_encoder`, which is not a test — the mark does nothing there. Its
   callers are marked; the stray decorator was removed.
2. **A gradient test that could not fail.** It read
   `features.sum().backward() if features.requires_grad else None` — and features
   never require grad, so the backward never executed and the assertion was
   vacuous. Replaced with a test that fits a head on the features, always
   backpropagates, asserts the head **does** get gradients and the encoder does
   **not**.

The second is the more serious: it is precisely the class of defect this project
has hit repeatedly — a test that passes for a reason unrelated to what it claims
to check.

---

## K. WHAT HAS NOT HAPPENED

- **No LR sweep.** The grid is implemented and not executed.
- **No head trained on real data.** No real representation was extracted.
- **No downstream score.** No Vanilla-vs-Base-only number exists.
- **No model or dataset downloaded.** The local `.venv` remains ML-free;
  torch, transformers, datasets, numpy and sklearn are all absent.
- **Official TEST never loaded**, and now unreachable by construction.
- **No Stage-1 training or HPO.** Stage-1 is untouched.
- **No prohibited git operation.** Nothing staged, committed, tagged or pushed.

---

## L. DECISION-LOG REVIEW — NO NEW DECISION WARRANTED

`docs/spec/decisions.md` was reviewed. **No new decision entry was created**, and
none should be: every scientific value this implementation uses was already
decided and recorded.

- The head, loss, optimiser, decay groups, schedule, batch size, epochs,
  checkpoint rule and paired-initialisation rule are D-PREG1-009 and D-PREG1-010.
- The LR grid, tuning seeds, measurement seeds and Vanilla-only selection are
  D-PREG1-007 and D-PREG1-009.
- `max_length` 256 is D-PREG1-008b; split roles and the seal are D-PREG1-003 and
  D-PREG1-004b; `<s>` pooling for this diagnostic is D-PREG1-005.
- The derived pool and its membership are D-PREG1-011 and D-PREG1-014.

Writing code that obeys a decision is not a decision. **The one thing that came
close** — adding a `padding` parameter to `encode_split` — is not a scientific
choice either: `PADDING = "max_length"` was already locked in `preg1_protocol`,
and the parameter merely stops the pre-G1 contract and the general Stage-2
default from overwriting one another. It is recorded here rather than as a
decision entry.

If the Colab run later forces a genuine narrowing — as Audit 022's schema defect
and Audit 023's fail-open splitter did — that will warrant an entry then.

---

## M. LIMITATIONS

1. *(Repaired in Revision 1.)* The role guard was a **declaration, not a
   proof**: `train_head` verified a free `dev_role` argument that no code could
   check against the tensor. It is now read from the representation's validated
   `RepresentationKey`, and no role parameter exists to contradict it. **What
   remains true:** a caller who fabricates both a key and a matching tensor is
   outside the validated boundary, and no in-process check can detect that. The
   supported path — extract or load through the cache, then train — is bound;
   arbitrary construction is not, and is not claimed to be.
2. **Nothing here has met a real encoder.** Every torch test uses a fake
   embedding module. The extraction path is simple and its contract is checked,
   but PhoBERT's actual output object, tokenizer behaviour and `<s>` position
   are exercised only on Colab. Audit 022 is the standing reminder that a real
   run finds things fixtures do not.
3. **28 of 134 tests have not executed anywhere.** The head init, optimiser
   groups, frozen-encoder checks, cache round-trip and the end-to-end path are
   the torch-dependent ones — and they include the paired-initialisation
   guarantee, among the most load-bearing claims here. They are **written, not
   verified**. Revision 1 moved as much as possible out of this bucket by making
   the binding contract duck-typed, but the tensor-level claims genuinely
   require torch.
4. **The precommitted counts from Audit 023 are not re-verified by this module.**
   It trusts the membership files it is given, checking their internal
   consistency and optionally their digests. It does not re-derive the split.
5. **`build_head` sets the global torch RNG** as well as using a local
   generator. Deliberate, so the guarantee holds regardless of what a caller did
   first — but it is a global side effect, and a caller relying on ambient RNG
   state across a `build_head` call would be surprised.
6. **I answered an audit question more strongly than my own evidence allowed**
   (Revision 1). §M of the original audit described the role guard as "a
   declaration, not a proof", and §I still answered Q1 with an unqualified
   "No". Both sentences were written in the same pass. Recording a limitation
   does not license claiming it away three sections earlier — and the external
   review, not I, is what caught it.
7. **I reported skipped tests as passes** (Revision 1). "87 passed" described a
   run in which 21 of those tests did not execute. The pytest output said
   "84 passed, 21 skipped" and I compressed it wrongly in prose.
8. **I fixed the count and left the claim** (Revision 1a). Revision 1 corrected
   the accounting to "28 torch-gated, PENDING" and, in the same document, kept
   §I's header saying every answer "was checked by running the code" — with Q4
   and Q5 resting on nothing but those pending tests. Correcting a number is not
   the same as correcting what the number was used to assert, and this is the
   second consecutive revision in which external review caught that distinction
   rather than I.
9. **I qualified the body and left the summary** (Revision 1b). Revision 1a
   added per-answer evidence levels to §§E, F, I and N, and left the final
   status block asserting the same four properties as bare facts — the block a
   reader skims first. Three consecutive revisions have now corrected the same
   underlying error in a different location: the fix was applied where I was
   looking, not everywhere the claim lived.

---

## N. TASK-END SELF-AUDIT

| # | Check | Result |
|---|---|---|
| 1 | Audits 021, 022 and final 023 reread; decision log and experiment records read | **yes** |
| 2 | Audit 024 created and persisted | **yes** |
| 3 | No real downstream training | **yes** |
| 4 | No Vanilla/Base-only score produced | **yes** |
| 5 | No LR tuned | **yes** |
| 6 | Official TEST never loaded or scored | **yes** — unrepresentable |
| 7 | No Stage-1 training or HPO | **yes** |
| 8 | No model or dataset downloaded | **yes** — `.venv` verified ML-free |
| 9 | Official validation cannot enter selection | **yes** — three layers |
| 10 | Base-only cannot influence the primary LR | **yes** — two layers |
| 11 | Paired seeds give identical starts | **implementation present; test authored; runtime verification PENDING** — torch-gated, not executed |
| 12 | Encoder frozen, eval, `no_grad`, FP32, no gradient path | **implementation present; five tests authored; runtime verification PENDING** — torch-gated, not executed |
| 13 | Cache fails closed on every provenance field | **yes for the 14 key-field cases** (executed locally); the on-disk load path is **PENDING** |
| 14 | No raw corpus text in metadata, artifacts or errors | **yes** |
| 15 | Sample SD, not population | **yes** |
| 16 | All 30 epochs; no early stopping | **yes** for the `HeadRun` / `require_full_schedule` guards (executed locally); the training loop's execution is **PENDING** |
| 17 | Checkpoint and LR tie-breaks are total orders, order-invariant | **yes** |
| 18 | `<s>` pooling scoped to this diagnostic only | **yes** for naming and scope (executed locally); the pooling arithmetic is **PENDING** |
| 19 | No UNMARK/adapter/Stage-1 concept in the diagnostic | **yes** — AST-asserted |
| 20 | No word segmentation | **yes** |
| 21 | Locked constants imported, never restated | **yes** — AST-asserted |
| 22 | No parallel protocol definition created | **yes** — one additive parameter on existing code |
| 23 | D-B3B0-002 | **OPEN** |
| 24 | Final Stage-2 pooling | **OPEN** |
| 25 | Compiled PDF | **STALE** |
| 26 | No new decision invented | **yes** — §L |
| 27 | Tests | **2305 passed, 84 skipped** full suite; targeted file **106 passed, 28 skipped** — the skips are **not** counted as passes |
| 28 | `git diff --check` clean | **yes** |
| 29 | Everything unstaged | **yes** |
| 30 | No prohibited git operation | **yes** |
| | **— Revision 1 —** | |
| 31 | Audit 024 revised **in place**; **no Audit 025** | **yes** |
| 32 | Representation values are provenance-bound | **yes** — `BoundRepresentations(values, key)` |
| 33 | Role cannot be overridden independently of provenance | **yes** — no role/pathway parameter exists; asserted by signature inspection |
| 34 | Official-validation-bound features cannot enter checkpoint or LR selection | **yes** — `require_training_roles`, then `HeadRun`, then `LrCandidate` |
| 35 | Protocol-dev-bound features cannot be used as the final measurement | **yes** — the `require_role` guard executed locally; the `score_measurement` wrapper is **PENDING** |
| 36 | Protocol-train-bound features cannot be used as the final measurement | **yes** — same guard, executed locally |
| 37 | A tensor cannot be rebound under a contradictory key | **yes** — shape and dtype checked at construction, executed locally; the `requires_grad` branch is **PENDING** |
| 38 | Cache load returns the bound object with the validated key | **implementation present; test authored; PENDING** — torch-gated |
| 39 | Pathway and role cache reinterpretation fail closed, both directions | **yes at the key level** (executed locally); the on-disk reload refusal is **PENDING** |
| 40 | Base-only still cannot influence LR selection | **yes** — unchanged |
| 41 | Paired-initialisation contract unchanged | **yes** — `build_head` untouched by Revision 1; its runtime verification is **PENDING** (row 11) |
| 42 | Official TEST role still absent | **yes** — none added |
| 43 | Exact local pass/skip accounting stated, skips not reported as passes | **yes** — §A |
| 44 | Torch runtime verification still **pending** | **yes** — 28 tests |
| | **— Revision 1b —** | |
| 45 | Status block matches the evidence levels in §§E, F, I, N | **yes** — four unqualified lines corrected, locally-verified lines marked `LOCAL-PASS` |
| 46 | Q4 and Q5 remain runtime **PENDING** everywhere they appear | **yes** |
| 47 | Cache on-disk save/load path remains runtime **PENDING** | **yes** — §F, §I Q6/Q7, §J, status block |
| 48 | Real-model and real-data gates remain **PENDING** | **yes** — §O steps 2 and 3, status block |
| 49 | No scientific claim upgraded without evidence | **yes** — this pass only ever weakened claims |
| 50 | No code, test, constant or decision changed | **yes** — documentation only |

---

## O. REQUIRED NEXT ACTION — FIVE STEPS, IN ORDER

**None of these was run in this task.** Steps 1–3 produce no classifier result
and no score; only step 5 does, and it comes after a separate approval.

### 1. Colab torch-runtime verification

Execute **all** tests in `tests/test_preg1_head.py` with torch available, and
record the **exact** pass/skip accounting. The 28 currently-unexecuted tests must
actually run. Until they do, Q4 and Q5 stay **PENDING** and the paired-init and
frozen-encoder guarantees are written, not verified.

### 2. Real PhoBERT-base integration smoke — **no optimizer, no training**

Against the exact diagnostic checkpoint and revision, with the real tokenizer:

- Vanilla and Base-only preprocessing through the **same** tokenizer contract;
- `max_length = 256`, `padding = "max_length"`, `truncation = True`;
- the real frozen encoder, in `eval`, under `no_grad`;
- first-`<s>` pooling;
- FP32 `BoundRepresentations`;
- representation cache save **and** load, with the provenance/key checks firing;
- **no downstream score**.

This is where a fake embedding module stops being adequate: PhoBERT's actual
output object, tokenizer behaviour and `<s>` position have never been exercised
here. Audit 022 is the standing reminder that a real run finds what fixtures do
not.

### 3. Real approved-data / membership boundary dry-run — **still no training**

- reconstruct or verify the approved derived TRAIN identity;
- recover the safe Audit-023 membership artifacts;
- verify the locked hashes and the assignment identity;
- prove protocol-train / protocol-dev binding end to end;
- verify official validation is measurement-only;
- **do not touch official TEST**;
- **produce no classifier result**.

### 4. Revise Audit 024 **in place** with the step 1–3 Colab evidence

No Audit 025. The runtime and integration evidence closes this audit's pending
half; the verdict may then drop the pending clause.

### 5. Only after that audit closes **and** the researcher approves the exact
implementation, and it is committed

- Vanilla-only LR tuning across the precommitted grid;
- freeze the winning LR;
- paired Vanilla-vs-Base-only measurement on official validation.

That paired result will be the **first downstream number in this project**. It
measures a burden. It does not test UNMARK, and it must not be reported as if it
did.


---

```
AUDIT 024 CREATED:
YES

VERDICT:
IMPLEMENTATION PASS — LOCAL CONTRACT TESTS PASS;
COLAB TORCH / REAL-MODEL / REAL-DATA BOUNDARY VERIFICATION PENDING;
NO DIAGNOSTIC RUN, NO DOWNSTREAM SCORE

SCIENTIFICALLY CLOSED:
NO

REVISION:
1a

REPRESENTATION BINDING:
PROVENANCE-BOUND (BoundRepresentations = values + RepresentationKey)

FREE ROLE ARGUMENT:
REMOVED — ROLE COMES FROM PROVENANCE

BASELINE HEAD:
57f70e373d46e919b754e24dd875f6d39e01e35c

OFFICIAL TEST:
SEALED / UNREPRESENTABLE IN Preg1Role AND CLI — LOCAL-PASS

OFFICIAL VALIDATION:
CANNOT ENTER LR OR CHECKPOINT SELECTION — LOCAL-PASS

BASE-ONLY -> PRIMARY LR:
BLOCKED — LOCAL-PASS

ROLE PROVENANCE BINDING:
ENFORCED — LOCAL-PASS

CHECKPOINT / LR SELECTORS:
TOTAL-ORDER, ORDER-INVARIANT — LOCAL-PASS

PAIRED INITIALISATION:
IMPLEMENTED / TEST AUTHORED / TORCH RUNTIME PENDING

ENCODER FREEZE / EVAL / NO_GRAD / FP32:
IMPLEMENTED / TESTS AUTHORED / TORCH RUNTIME PENDING

REPRESENTATION CACHE:
PROVENANCE LOGIC LOCAL-PASS /
TORCH SAVE-LOAD RUNTIME PENDING

EPOCH CONTRACT:
30 / NO EARLY STOPPING — CONTRACT LOCAL-PASS /
TRAINING-LOOP RUNTIME PENDING

SAMPLE SD:
YES (n-1) — LOCAL-PASS

SIGNIFICANCE TEST:
NONE BY DESIGN

LR SWEEP:
IMPLEMENTED / NOT RUN

PAIRED MEASUREMENT:
IMPLEMENTED / NOT RUN

DOWNSTREAM SCORE:
NONE

STAGE-1 TRAINING:
NOT RUN

NEW SCIENTIFIC DECISION:
NONE WARRANTED

D-B3B0-002:
OPEN

STAGE-2 POOLING:
OPEN

PDF:
STALE

LOCAL TESTS:
full suite 2305 passed, 84 skipped
targeted file: 134 authored / 106 passed locally / 28 torch-gated NOT EXECUTED

TORCH RUNTIME VERIFICATION:
PENDING (28 tests, never executed)

REAL-MODEL INTEGRATION:
PENDING

REAL-DATA BOUNDARY DRY-RUN:
PENDING

COMMIT CREATED:
NO
```
