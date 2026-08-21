# Audit 024 — pre-G1 frozen-encoder head trainer and evaluator

| | |
|---|---|
| **Audit id** | 024 |
| **Created (UTC)** | 2026-08-21 |
| **Baseline HEAD** | `57f70e373d46e919b754e24dd875f6d39e01e35c` |
| **Scope** | Implement the frozen-encoder linear-head trainer, evaluator, LR selector, paired measurement structure and representation cache. **Run no scientific diagnostic with it.** (Its *test suite* has since executed on GPU — §P.) |
| **Predecessors** | [021](021-pre-g1-dataset-profile-and-protocol-precommit.md), [022](022-uit-vsfc-real-data-profile-integrity-closure.md), [023](023-pre-g1-internal-split-materializer-and-fail-closed-contract.md) |
| **Phase** | pre-G1 |
| **Type** | **Implementation + audit.** No real training, no LR sweep, no downstream score. **No model or dataset was downloaded during the local implementation/audit work** — the local `.venv` stayed ML-free. *(Separately: an accidental Colab probe execution occurred under a broken negative test; whether it downloaded or cache-loaded assets is **NOT ESTABLISHED** — §Q.4 Group C.)* |
| **Revision 4b** | **2026-08-21** — **four residual current-state items.** §K said "no real representation was extracted", false after C24-4-R2 (which extracted real-PhoBERT representations from **synthetic** smoke inputs); §N rows 48 and 83 still read as current pending gates; and row 80's rationale preserved a wrong claim — I had written that `results/` is unignored, having checked the *directory* (re-included by `!results/**/`) rather than a file beneath it. **Verified here:** `.gitignore:15 `` results/** `` matches `results/b3b1/…/*`, so the accidental probe **is** gitignored. All four repaired as historical/superseded; no accepted evidence touched. **Documentation only.** |
| **Revision 4a** | **2026-08-21** — **current-state consistency repair.** Four defects found by external review: §A still denied that the real PhoBERT smoke and real-data dry-run had run — **they had** (C24-4-R2, C24-5B-R1), as *verification*, not diagnostic execution; §J still said the Colab runs used "no real model and no real corpus at any point" and that the Revision-3 repository repairs were unexecuted; the current local count still read 2 310 / 89 instead of **2 312 / 89**; and the status block's `SCIENTIFIC DIAGNOSTIC EXECUTION: NONE … MODEL SMOKE` contradicted its own `REAL PHOBERT INTEGRATION: PASS / CLOSED`. All repaired; the same false claim in §K was repaired too. Historical §§P/Q/R untouched. **Documentation only.** |
| **Revision 4 — FINALIZATION** | **2026-08-21** — **all four gates closed.** At `767fb8ee30fa9b5df344a3629b3295c1fd03c379` the **full repository Torch suite returned 2 401 passed / 0 failed / 0 errors** (C24-1-R3C), closing the repository-wide gate. C24-4-R2 then closed the **real PhoBERT integration** gate — `FIRST_TOKEN` extraction proved **exactly equal** to `last_hidden_state[:, 0, :]` on the real `RobertaModel`, with a real cache round-trip and a real cross-pathway refusal. C24-5B-R1 closed the **real approved-data / role boundary** gate — the derived TRAIN digest `a20c0f77…` reproduced from the official source, real Audit-023 membership bound row-for-row, and both role-substitution directions refused with `SplitLeakage`. **Official TEST never selected, downloaded, read or scored.** No head, optimizer, training, LR tuning or downstream score. **No production change; no new scientific decision.** |
| **Revision 3b** | **2026-08-21** — **C24-1-R3B: repairs pass, a shared fixture did not.** At `f6ad4eef21cff9826f6ba58191a3dbcfcbed491a` the targeted suite reproduced **136/0/0** and the **exact eight R3 repairs passed 8/8**. The broader related/safety group then **failed 8/328**, so the full repository suite was **NOT REACHED**. Root cause (§R): Revision 3's accent-sensitive `StubTokenizer` emitted ids up to 4099 while `stub_encoder` declared a 64-token vocabulary — **my own fixture change broke an implicit agreement**, and eight unrelated tests died in `nn.Embedding` before reaching their named properties. **Test-fixture regression; no production defect.** Repaired with one single-sourced vocabulary contract plus two mutation-verified local guards. **No production line changed.** |
| **Revision 3a** | **2026-08-21** — **evidence-consistency sweep after C24-1-R2.** §F still called the on-disk cache path "torch-gated and pending"; §I's rows still showed Q4/Q5 `PENDING` and Q6/Q7/Q11–Q13 `PARTIAL`/`PENDING` while its own summary already said otherwise; §A still listed "the torch runtime" among the unverified; and the no-download claim was unqualified beside the accidental-probe uncertainty. All corrected. **Every upgraded label was checked against an actual `@requires_torch` test in `tests/test_preg1_head.py` before relabelling** — 14 tests confirmed. **Documentation only.** |
| **Revision 3** | **2026-08-21** — **C24-1-R2 targeted PASS + repository-wide regression repair.** At HEAD `52ebca0657dc39b8e8e25ebd71c90ae7a4687501` the targeted suite returned **136 passed / 0 failed / 0 skipped** — that gate is **closed**. The repository-wide suite then returned **2 383 passed / 8 failed** after the pinned inventory was recovered (a resource requirement, not a defect). All eight were inspected and classified: **none was an implementation defect** — six stale fixtures, one non-discriminative stub, one environment-sensitive test. Repaired in §Q. **One production change: an additive `ChannelContractViolation` grid-agreement guard (+26/−0), classified as implementation hardening**; the verified-position rule was **not** weakened and `VERIFIED_POSITION_PROFILES` still holds exactly one entry. **Repository-wide Torch regression remains PENDING REPAIR.** |
| **Revision 2a** | **2026-08-21** — **temporal / evidence wording repair.** §A still said "Nothing was run" after C24-1-R1 had executed 134 tests on GPU; §I defined PENDING as "has not run anywhere" when **26 of the 28** torch-gated tests had executed and passed; §E called unexecuted tests "tested". All corrected. **No label was upgraded** — Q4 and Q5 remain PENDING, now for the accurate reason (a passing test inside a failing run is not *accepted* evidence) rather than the false one (never executed). **Documentation only.** |
| **Revision 2** | **2026-08-21** — **first real Torch execution (C24-1-R1) and its two failures.** At HEAD `b43cca829ca163b9d6a818c980dfbf6fdaea651f` the targeted suite ran on GPU Colab: **132 passed, 2 failed, 0 skipped**. Both failures were in **tests**, not in the implementation — one a test-isolation defect, one a stale assertion left over from the Revision-1 API strengthening (§P). Both repaired; **no implementation line changed**. **C24-1 runtime closure remains PENDING** until a new committed revision executes all 136 tests successfully on Colab. |
| **Revision 1b** | **2026-08-21** — **status-block consistency repair.** The final status block still asserted `PAIRED INITIALISATION: BIT-IDENTICAL…`, `ENCODER: FROZEN / EVAL / NO_GRAD / FP32`, `REPRESENTATION CACHE: FAIL-CLOSED…` and `EPOCHS: 30 / NO EARLY STOPPING` as unqualified facts, contradicting the evidence levels Revision 1a had just written into §§E, F, I and N. All four are now qualified; genuinely local-verified lines are marked `LOCAL-PASS`; §J's coverage list is tagged per group. **Documentation only — no code, test, constant or scientific value changed.** |
| **Revision 1a** | **2026-08-21** — **status-consistency repair.** §I claimed "each answer was checked by running the code" when two of the seventeen (Q4 paired init, Q5 frozen encoder) rest **only** on torch-gated tests that have never executed; every answer now carries an explicit LOCAL / PARTIAL / PENDING / INSPECTION status. Self-audit rows whose only proof is among the 28 unexecuted tests are corrected to *implementation present; test authored; runtime verification PENDING*. §O replaced with the real five-step sequence. **No code, constant or scientific value changed.** |
| **Revision 1** | **2026-08-21** — **fail-closed repair + honest test accounting.** (A) The local outcome was summarised as "87 passed" when 21 of those tests are torch-gated and did **not** execute; every such statement is corrected, and the torch-gated tests are now marked **PENDING** rather than passing. (B) Representations were passed as bare tensors beside a free `dev_role` argument, so the role was a *claim about* a tensor rather than a *property of* it — §M admitted this and the audit still answered Q1 with an unqualified "No". Repaired: `BoundRepresentations` carries the tensor with its `RepresentationKey`, the role comes from provenance, and **no role argument exists anywhere to contradict it**. |

---

## A. VERDICT

**IMPLEMENTATION PASS — ALL FOUR VERIFICATION GATES CLOSED;
NO DIAGNOSTIC RUN, NO DOWNSTREAM SCORE**

| Gate | State |
|---|---|
| Audit-024 targeted Torch contract | **PASS / CLOSED** — 136/136 |
| Repository-wide Torch regression | **PASS / CLOSED** — **2 401 / 0 / 0** |
| Real PhoBERT integration | **PASS / CLOSED** |
| Real approved-data / role boundary | **PASS / CLOSED** |

**Audit 024 is closed as an implementation / pre-diagnostic audit** (§S). Its
implementation, runtime, integration and data-boundary verification is complete.

**That is the whole of the claim.** It does **not** mean the Vanilla-vs-Base-only
burden result exists, that pre-G1 is finished, that Stage-1 is approved for
training, that Stage-2 pooling is locked, or that D-B3B0-002 is resolved — see
§S.9.

The mechanism for the pre-G1 Vanilla-vs-Base-only burden diagnostic exists. Its
**No scientific downstream diagnostic was executed.** Precisely — what did
**not** happen: no real-data head training, no LR tuning, no downstream score,
no Vanilla-vs-Base-only burden diagnostic, and no Stage-1 training or HPO.

**What did happen, as verification only:** the real PhoBERT integration smoke
(C24-4-R2, §S.5) and the real TRAIN/DEV integrity, reconstruction and
role-boundary dry-run (C24-5B-R1, §S.8). **Neither is a downstream scientific
result** — the first extracted representations and exercised the cache with no
head and no optimizer; the second read source bytes and bound membership with
dummy tensors and no model. Official TEST content was never selected,
downloaded, read or scored, and the role is not reachable from this code at
all.

**Tests, however, have now run on GPU.** C24-1-R1 (§P) executed the targeted
suite at HEAD `b43cca82…`: 132 passed, 2 failed, 0 skipped. Saying "nothing was
run" stopped being true at that moment, and this audit does not say it.

### Test accounting, stated exactly

| | |
|---|---|
| New tests **authored** | **136** (134 + 2 added by Revision 2) |
| New tests **executed locally** | **108 passed** |
| New tests **skipped locally** (torch absent) | **28 — all passed on GPU in C24-1-R2** |
| Full local suite | **2 312 passed, 89 skipped** |
| Baseline at Audit 023 | 2199 passed, 56 skipped |
| **First Colab Torch run (C24-1-R1)** | **132 passed, 2 FAILED, 0 skipped** — all 134 then-current tests executed; see §P |
| **Targeted Colab rerun (C24-1-R2)** | **136 passed / 0 failed / 0 skipped** at `52ebca06…` — **PASS**; see §Q |
| **Repository-wide Colab run (C24-1-R2B)** | **2 383 passed / 8 failed** — see §Q |
| **Revision-3 verification (C24-1-R3B)** | targeted **136/0/0 PASS**; the eight R3 repairs **8/8 PASS**; broader group **8 failed / 328 passed**; full suite **NOT REACHED** — see §R |
| **Final verification (C24-1-R3C)** | **full repository Torch suite 2 401 passed / 0 failed / 0 errors** at `767fb8ee…` — **CLOSED**; see §S.1 |

**The Audit-024 targeted Torch gate is CLOSED.** C24-1-R2 reran the whole file
at HEAD `52ebca06…`: **136 passed / 0 failed / 0 skipped**. All 28 torch-gated
tests executed and passed — including the two repaired after C24-1-R1, the
paired-initialisation guarantee, the frozen-encoder checks, the optimiser
parameter groups, the on-disk cache path and the end-to-end training loop.
`tests/test_preg1_head.py` is **unmodified since that commit**, so the result
applies to the file as it stands today.

**Repository-wide health is now established too.** C24-1-R2B returned 2 383 / 8
(§Q); C24-1-R3B confirmed those eight repairs pass but exposed a fixture
interaction that failed eight evaluation-harness tests (§R); **C24-1-R3C then
returned 2 401 passed / 0 failed / 0 errors** (§S.1). Both gates are closed, and
the path there is preserved in §§Q, R and S rather than smoothed over.

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
| `tests/test_preg1_head.py` | **136** tests — 108 pass locally, 28 torch-gated; **all 136 passed on GPU in C24-1-R2** |
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

**Evidence status per property.** The torch-dependent rows were **verified on
GPU** in the clean C24-1-R2 targeted run (136/0/0 at `52ebca06…`) — not merely
authored. `TORCH-PASS` means executed and passed on Colab; `LOCAL` means proved
by torch-free tests in the ML-free environment.

| Property | How | Evidence |
|---|---|---|
| same seed → bit-identical head parameters | `build_head` resets the RNG from the seed **immediately before** initialising, and applies Xavier-uniform / zeros **explicitly** rather than relying on `nn.Linear`'s default (a Kaiming variant whose exact form has changed across PyTorch versions) | **TORCH-PASS** — C24-1-R2 at `52ebca06…` |
| Vanilla and Base-only start identically under a paired seed | the reset makes this hold **regardless of which arm ran first or what consumed RNG in between** — the test advances the global RNG by 1000 draws between the arms and still requires bitwise equality | **TORCH-PASS** — C24-1-R2 at `52ebca06…` |
| different seed → different initialisation | | **TORCH-PASS** — C24-1-R2 at `52ebca06…` |
| deterministic batch order | `deterministic_batches` uses its own `random.Random(seed)`, never the ambient global RNG; a test reseeds `random` between calls and requires identical output | **LOCAL** |
| batch order varies by epoch but is identical across arms | derived from `(seed, epoch)` | **LOCAL** |
| checkpoint tie-break | total order via `min(..., key=(-f1, -acc, epoch))` | **LOCAL** |
| LR tie-break | total order via `min(..., key=(-mean_f1, -mean_acc, sd, lr))` | **LOCAL** |
| no dependence on mapping insertion order | both selectors sort on **content**; tested with reversed inputs | **LOCAL** |

**The three head-initialisation rows matter most, and they are now verified.**
They are pure torch, so no amount of duck-typing could move them into the local
suite; the clean C24-1-R2 run is what settled them. *(Historically: C24-1-R1
executed them too, but inside a run that failed as a whole, so they were held at
PENDING until R2.)*

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

**LOCAL** — proved by torch-free tests in the ML-free environment: the
provenance/key comparison logic, and fourteen incompatible-metadata cases tested
individually (including the pathway swap), plus the malformed-metadata and
unknown-enum refusals.

**TORCH-PASS** — executed and passed on GPU in C24-1-R2 at `52ebca06…`: the on-disk
save/load round-trip; FP32 and shape enforcement on both save and load; the
bound return value carrying the validated key; and role/pathway reinterpretation
refused on disk in both directions.

Both halves are now verified — the *logic* locally and the *file path* on GPU.

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
| **TORCH-PASS** | executed and passed on GPU in the clean C24-1-R2 targeted run (136/0/0 at `52ebca06…`), on a file unmodified since that commit |
| **LOCAL + TORCH-PASS** | the guard logic proved locally **and** the torch runtime path around it proved on GPU |
| **PENDING** | no accepted executable proof yet — never executed, or executed only inside a run that failed as a whole. *(After C24-1-R2 no question carries this label; the repository-wide repairs in §Q do.)* |
| **INSPECTION** | established by reading or grepping the code, with no executable proof |

An earlier revision introduced this table with "each answer was checked by
running the code". That was true of most answers and false of several, and the
distinction is exactly what an audit is for.

| # | Question | Answer | Evidence |
|---|---|---|---|
| 1 | Can official validation influence LR selection in any code path? | **No, on the supported path — and now for a reason that holds.** *(Revision 1: the earlier "No" was not justified; see §D.)* The role is read from the representation's `RepresentationKey`; `train_head` has **no role parameter** to override it; `require_training_roles` demands `PROTOCOL_DEV`; `HeadRun.scored_on` records what was used; `LrCandidate` re-checks it; a contradictory key is rejected at construction. | **LOCAL** — the binding guards, the signature check and both selection layers are torch-free and executed |
| 2 | Can official TEST influence anything? | **No**, precisely: (a) `Preg1Role` has no `OFFICIAL_TEST` member, so it is unnameable in any signature; (b) the CLI exposes no test flag; (c) representations carry a validated role and source identity, so a supported-path reinterpretation fails closed. **Not claimed:** protection against deliberate tampering outside the validated boundary — someone who fabricates a key *and* a matching tensor is beyond what any in-process check can detect. | **LOCAL** — enum membership and CLI-AST tests executed |
| 3 | Can Base-only influence primary LR selection? | **No.** `LrCandidate` raises `SplitLeakage` on any non-VANILLA run; `FrozenLearningRate` refuses `selected_on != VANILLA`. | **LOCAL** |
| 4 | Can the arms start from different parameters for the same paired seed? | **No.** `build_head` reseeds from the seed immediately before initialising. `test_paired_seed_gives_identical_starts_for_both_pathways` — which advances the global RNG by 1000 draws between the arms and then requires bitwise equality — **executed and passed on GPU** in the clean C24-1-R2 run. | **TORCH-PASS** — C24-1-R2 at `52ebca06…` |
| 5 | Can the encoder receive gradients or leave eval mode? | **No.** Every parameter and the mode are checked before extraction; extraction runs under `no_grad` and returns a detached FP32 tensor. The four frozen-encoder tests and the real-backward test **all executed and passed on GPU** in C24-1-R2. | **TORCH-PASS** — C24-1-R2 at `52ebca06…` |
| 6 | Can a Vanilla cache be reused as Base-only? | **No.** Pathway is a key field and comparison is exact, and the on-disk reload refusal was exercised. | **LOCAL** (key-level refusal) **+ TORCH-PASS** (on-disk reload refusal, C24-1-R2) |
| 7 | Can stale cache metadata silently pass? | **No.** Every field compared; missing keys, unknown enum values and unparseable JSON all raise; the on-disk load path enforces FP32 and shape. | **LOCAL** (14 field-mismatch + malformed-metadata cases) **+ TORCH-PASS** (on-disk load path, C24-1-R2) |
| 8 | Can checkpoint tie-breaking depend on iteration accidents? | **No.** `min()` over an explicit key. A scan with `>` keeps the first maximum and `>=` the last — the rule would be decided by iteration order. | **LOCAL** — tested with reversed input |
| 9 | Can LR tie-breaking depend on mapping insertion order? | **No.** Sort key over candidate content. | **LOCAL** — tested forward and reversed |
| 10 | Is sample standard deviation used? | **Yes.** `statistics.stdev`; asserted to differ from `pstdev`. | **LOCAL** |
| 11 | Is there any early stopping? | **No.** No patience parameter exists anywhere, `require_full_schedule` rejects a truncated score list, and a real 30-epoch run produced all 30. | **LOCAL** (contract guard) **+ TORCH-PASS** (the loop ran, C24-1-R2) |
| 12 | Are all 30 epochs guaranteed? | **Yes.** The loop is `range(1, epochs+1)` with no `break`, `HeadRun` refuses construction unless epochs 1..30 are present, and the loop was executed end to end. | **LOCAL** (`HeadRun` guard) **+ TORCH-PASS** (execution, C24-1-R2) |
| 13 | Is `<s>` pooling scoped to this diagnostic only? | **Yes.** The function is named `PREG1_ONLY_first_token_pool` and `PREG1_POOLING_SCOPE` states final Stage-2 pooling remains **OPEN** (D-G1-005); the position-0 arithmetic and its rank guard were exercised. | **LOCAL** (naming, scope) **+ TORCH-PASS** (arithmetic, C24-1-R2) |
| 14 | Does anything close D-B3B0-002? | **No.** Nothing in the new code references or resolves it. The checkpoint is used as a **probe** revision. | **INSPECTION** — grep over the new modules |
| 15 | Is official TEST still sealed? | **Yes**, and structurally unreachable from this module. | **LOCAL** |
| 16 | Has any downstream score been produced? | **No.** | **fact of this task** |
| 17 | Has any real downstream training been run? | **No.** | **fact of this task** |

**Summary of evidence (Revision 3).** After C24-1-R2, **no question is
PENDING**: 8 are `LOCAL`, 2 (Q4, Q5) are `TORCH-PASS`, 5 (Q6, Q7, Q11–Q13) are
`LOCAL + TORCH-PASS`, 1 rests on inspection, and 2 state what this task did.

**This is the Audit-024 targeted contract only.** The eight repository-wide
regression failures (§Q) are a separate gate, and their repairs remain
unverified on Torch.

---

## J. TESTS

**136 tests** in `tests/test_preg1_head.py`: **108 pass locally** and **28 are
torch-gated**, which locally skip. On GPU, **C24-1-R2 ran all 136 and all
passed** (`52ebca06…`), so the torch-gated 28 are `TORCH-PASS`, not pending.
*(Historically, C24-1-R1 ran the pre-repair file and returned 132/2/0 — §P.)*

**Locally**: no downloads, no network, no GPU, no real UIT-VSFC, no real split
artifacts.

**What the Colab runs used, in order.** The C24-1 runs (R1, R2, R2B, R3B, R3C)
were **test and runtime verification** — a GPU for the suite, no real model and
no real corpus. **C24-4-R2 later used the real PhoBERT tokenizer and encoder**
for integration verification (§S.5). **C24-5B-R1 later read the exact official
TRAIN/DEV source bytes transiently** for integrity, reconstruction and
role-boundary verification (§S.8). **Neither produced downstream training or a
downstream score, and official TEST remained untouched.**

**The Revision-3 repository repairs have since run.** The exact eight passed
**8/8** in R3B; the complete evaluation harness passed **74/74** and the full
repository **2 401 / 0 / 0** in R3C. *(They were unexecuted at Revision 3 — §§Q
and R record that state at that time.)* They are not part of the 28.

Deliberately, **the contract logic is torch-free** — selection rules, membership
guards, cache provenance, report shape and the boundary checks are pure Python,
so the guarantees that matter most are verified in the ML-free environment
rather than deferred to a machine that may never run them.

Coverage by group — **`[LOCAL]`** executed in the ML-free environment;
**`[TORCH-PASS]`** executed and passed on GPU in C24-1-R2:

- `[LOCAL]` membership loading and every failure mode (duplicate, overlapping,
  incomplete, unknown id, malformed manifest, wrong schema, pinned digest
  mismatch);
- `[LOCAL]` pathway behaviour and the absence of segmentation;
- `[TORCH-PASS]` frozen encoder, eval mode, `no_grad`, FP32, real-backward
  isolation;
- `[TORCH-PASS]` head shape, Xavier bound, zero bias, seed determinism, paired
  identity, no dropout;
- `[TORCH-PASS]` AdamW settings, two decay groups, constant LR;
- `[LOCAL]` no scheduler / clipping / accumulation (AST);
- `[LOCAL]` epoch-0 ineligibility, full-schedule guard, all three checkpoint
  tie-breaks, macro-F1 edge cases, deterministic batching;
  `[TORCH-PASS]` the 30-epoch loop actually running;
- `[LOCAL]` LR grid and seed enforcement, all four aggregation tie-breaks,
  sample-vs-population SD, order invariance;
- `[LOCAL]` the five measurement seeds, delta direction, report contents,
  absence of significance fields;
- `[LOCAL]` fourteen cache-incompatibility cases and metadata hygiene;
  `[TORCH-PASS]` the on-disk save/load round-trip, FP32/shape refusals and
  role/pathway reinterpretation refusals.

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
- **No head trained on real UIT-VSFC**, and **no real UIT-VSFC representation
  was extracted for downstream training or measurement.** *(C24-4-R2 did extract
  real-PhoBERT representations — from three **synthetic** smoke inputs, for
  integration verification only; §S.5.)*
- **No downstream score.** No Vanilla-vs-Base-only number exists.
- **No model or dataset downloaded by this task.** The local `.venv` remains
  ML-free — torch, transformers, datasets, numpy and sklearn are all absent, and
  nothing here fetched a model or a corpus. **Scope matters:** the Colab test
  C24-1 runs used a GPU for the test suite only; **C24-4-R2 later loaded the
  real PhoBERT model from the repo-local HF cache** for integration verification
  (§S.5), and **C24-5B-R1 downloaded the four official TRAIN/DEV content files**
  for the data-boundary verification (§S.8) — neither trained anything nor
  produced a score, and both are verification, not diagnostic execution.
  Separately, the accidental probe under the broken negative test (§Q.4 Group C)
  reached the model-loading path: **whether it downloaded assets or loaded them
  from cache is NOT ESTABLISHED**, and this audit does not guess.
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
2. *(Discharged by C24-4-R2.)* This audit's tests use a fake embedding module,
   so for most of its life nothing here had met a real encoder. **C24-4-R2
   settled it**: on the real `RobertaModel`, `extract_bound_representations`
   returns values **exactly equal** to `last_hidden_state[:, 0, :]` for both
   pathways, with a real cache round-trip and a real cross-pathway refusal
   (§S.5). **What remains true:** the local suite still cannot see this, so a
   future change to extraction is only caught on Colab.
3. *(Closed by C24-1-R3C — 2 401 / 0 / 0.)* It took **three** Colab rounds to
   get there: R2 closed the targeted file, R2B exposed eight older regressions
   elsewhere (§Q), R3B confirmed those repairs but exposed eight more from a
   fixture interaction my own repair introduced (§R). **What remains true:** a
   green file was never evidence of a green repository, and every round revealed
   something the ML-free local environment could not see. That asymmetry has not
   gone away — it has only been paid down for this commit.
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
10. **My tests had two defects that only a real runtime could surface**
   (Revision 2). One asserted a property its own fixture prevented it from
   reaching; the other encoded an API I had deliberately changed in the same
   session. Both would have kept passing locally forever, because locally they
   never ran. The 28-test gap was not merely an accounting inconvenience — it
   was hiding two broken tests, which is exactly what a pending bucket is
   dangerous for.
11. **I described executed tests as never executed** (Revision 2a). Revision 2
   recorded "132 passed, 2 failed, 0 skipped" and, in the same document, left §I
   defining PENDING as "has not run anywhere" and §A saying "Nothing was run".
   Zero skips means all 134 ran. The correction is not an upgrade — Q4 and Q5
   stay PENDING — but the *reason* had to change from a falsehood to the truth:
   a passing test inside a failing run is **unaccepted**, not **unexecuted**.
12. **Six of my fixtures were stale and I did not notice for four revisions**
   (Revision 3). The B4B fail-closed hardening was mine; the fixtures it broke
   were mine; and every local run passed because torch was absent, so nothing
   surfaced until a GPU ran the *whole repository*. The targeted suite passing
   136/136 was not evidence that the repository was healthy — I had only ever
   run the file I was working on.

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
| 8 | No model or dataset downloaded **by the local implementation/audit work** | **yes** — `.venv` verified ML-free. *(The accidental Colab probe's download/cache status is **NOT ESTABLISHED** — §Q.4 Group C.)* |
| 9 | Official validation cannot enter selection | **yes** — three layers |
| 10 | Base-only cannot influence the primary LR | **yes** — two layers |
| 11 | Paired seeds give identical starts | **TORCH-PASS** — C24-1-R2, 136/0/0 at `52ebca06…` |
| 12 | Encoder frozen, eval, `no_grad`, FP32, no gradient path | **TORCH-PASS** — all five tests, C24-1-R2 |
| 13 | Cache fails closed on every provenance field | **LOCAL** (14 key-field cases) **+ TORCH-PASS** (on-disk load path, C24-1-R2) |
| 14 | No raw corpus text in metadata, artifacts or errors | **yes** |
| 15 | Sample SD, not population | **yes** |
| 16 | All 30 epochs; no early stopping | **LOCAL** (guards) **+ TORCH-PASS** (the loop ran, C24-1-R2) |
| 17 | Checkpoint and LR tie-breaks are total orders, order-invariant | **yes** |
| 18 | `<s>` pooling scoped to this diagnostic only | **LOCAL** (naming, scope) **+ TORCH-PASS** (arithmetic, C24-1-R2) |
| 19 | No UNMARK/adapter/Stage-1 concept in the diagnostic | **yes** — AST-asserted |
| 20 | No word segmentation | **yes** |
| 21 | Locked constants imported, never restated | **yes** — AST-asserted |
| 22 | No parallel protocol definition created | **yes** — one additive parameter on existing code |
| 23 | D-B3B0-002 | **OPEN** |
| 24 | Final Stage-2 pooling | **OPEN** |
| 25 | Compiled PDF | **STALE** |
| 26 | No new decision invented | **yes** — §L |
| 27 | Tests | local **2 312 passed, 89 skipped**; targeted file **108 passed, 28 skipped locally** — and **136/0/0 on GPU**, with the **full repository at 2 401 / 0 / 0** in C24-1-R3C |
| 28 | `git diff --check` clean | **yes** |
| 29 | Everything unstaged | **yes** |
| 30 | No prohibited git operation | **yes** |
| | **— Revision 1 —** | |
| 31 | Audit 024 revised **in place**; **no Audit 025** | **yes** |
| 32 | Representation values are provenance-bound | **yes** — `BoundRepresentations(values, key)` |
| 33 | Role cannot be overridden independently of provenance | **yes** — no role/pathway parameter exists; asserted by signature inspection |
| 34 | Official-validation-bound features cannot enter checkpoint or LR selection | **yes** — `require_training_roles`, then `HeadRun`, then `LrCandidate` |
| 35 | Protocol-dev-bound features cannot be used as the final measurement | **LOCAL** (`require_role` guard) **+ TORCH-PASS** (`score_measurement` wrapper, C24-1-R2) |
| 36 | Protocol-train-bound features cannot be used as the final measurement | **yes** — same guard, executed locally |
| 37 | A tensor cannot be rebound under a contradictory key | **LOCAL** (shape, dtype at construction) **+ TORCH-PASS** (`requires_grad` branch, C24-1-R2) |
| 38 | Cache load returns the bound object with the validated key | **TORCH-PASS** — C24-1-R2 |
| 39 | Pathway and role cache reinterpretation fail closed, both directions | **LOCAL** (key level) **+ TORCH-PASS** (on-disk reload refusal, C24-1-R2) |
| 40 | Base-only still cannot influence LR selection | **yes** — unchanged |
| 41 | Paired-initialisation contract unchanged | **yes** — `build_head` untouched by Revision 1; runtime **TORCH-PASS** in C24-1-R2 (row 11) |
| 42 | Official TEST role still absent | **yes** — none added |
| 43 | Exact local pass/skip accounting stated, skips not reported as passes | **yes** — §A |
| 44 | *(historical, Rev 1b)* Torch runtime verification still pending | **correct then; SUPERSEDED by Revision 3** — C24-1-R2 returned 136/0/0 |
| | **— Revision 1b —** | |
| 45 | Status block matches the evidence levels in §§E, F, I, N | **yes** — four unqualified lines corrected, locally-verified lines marked `LOCAL-PASS` |
| 46 | *(historical, Rev 1b)* Q4 and Q5 remain runtime PENDING everywhere | **correct then; SUPERSEDED by Revision 3** — both are **TORCH-PASS** after C24-1-R2 |
| 47 | *(historical, Rev 1b)* Cache on-disk save/load path remains runtime PENDING | **correct then; SUPERSEDED by Revision 3** — **TORCH-PASS** after C24-1-R2 |
| 48 | *(historical, Rev 1b)* Real-model and real-data gates remain PENDING | **correct then; SUPERSEDED by Revision 4** — real PhoBERT **CLOSED** by C24-4-R2 (§S.5); real-data boundary **CLOSED** by C24-5B-R1 (§S.8) |
| 49 | No scientific claim upgraded without evidence | **yes** — this pass only ever weakened claims |
| 50 | No code, test, constant or decision changed | **yes** — documentation only |
| | **— Revision 2: C24-1-R1 —** | |
| 51 | C24-1-R1 failure preserved in the audit history, not hidden | **yes** — §P, with HEAD, environment and both test names |
| 52 | Finding 1 classified after inspecting the code, not from the hypothesis | **yes** — the independent FP32 guard was located in `train_head` before classifying |
| 53 | Finding 1 **isolated** the FP32 contract rather than weakening the assertion | **yes** — both sets made mutually-consistent float64; the regex is unchanged; the test asserts `require_training_roles` does **not** raise first |
| 54 | Validation order not reordered to satisfy an error-message match | **yes** — agreement-before-absolute is principled and was left alone |
| 55 | Train/dev dtype mismatch still fails closed as a geometry mismatch | **yes** — `dtype` added to the parameterised geometry test; **executes locally** |
| 56 | `cache.load` still returns provenance-bound representations | **yes** — not reverted |
| 57 | No bare-Tensor cache regression introduced, and one is now caught locally | **yes** — torch-free AST test on `load`'s return annotation and every `return` |
| 58 | Provenance binding intact; no free role override | **yes** — unchanged |
| 59 | Zero implementation lines changed by Revision 2 | **yes** — `git diff` on `unmark/` and `scripts/` is empty |
| 60 | *(historical, Rev 2)* Repaired torch tests not described as passing | **correct then; SUPERSEDED by Revision 3** — both passed in C24-1-R2 |
| 61 | *(historical, Rev 2)* C24-1 runtime closure remains PENDING | **correct then; SUPERSEDED by Revision 3** — targeted gate closed |
| | **— Revision 2a —** | |
| 62 | No false "nothing was run" remains | **yes** — §A, scope row, §I definition and §J corrected |
| 63 | Unexecuted tests not described as runtime-tested | **yes** — §E now reads "covered by authored tests … not accepted at closure" |
| 64 | Tests that **did** execute not described as unexecuted | **yes** — 26 of the 28 passed on GPU and the audit says so |
| 65 | *(historical, Rev 2a)* No PENDING label upgraded | **correct then; SUPERSEDED by Revision 3** — C24-1-R2 earned the upgrade for Q4/Q5 |
| 66 | C24-1-R1 result, classifications and counts unchanged | **yes** — 132/2/0, both classifications, 136/108/28 intact |
| 67 | No code, test, script, constant or scientific value changed | **yes** — documentation only |
| | **— Revision 3: C24-1-R2 / R2B —** | |
| 68 | C24-1-R1 history preserved | **yes** — §P intact |
| 69 | C24-1-R2 targeted **136/136 PASS** preserved | **yes** — §Q.1 |
| 70 | Inventory recovery recorded as a resource requirement | **yes** — §Q.2; no logic touched, nothing committed |
| 71 | R2B **2383/8** failure preserved, not hidden | **yes** — §Q.3 |
| 72 | All eight failures individually classified with evidence | **yes** — §Q.4; none an implementation defect |
| 73 | Production position-profile fail-closed rule **not weakened** | **yes** — resolver patched in tests only |
| 74 | No fake encoder added to the verified-profile registry | **yes** — AST-verified `(PHOBERT_BASE_POSITION_PROFILE,)`, one entry |
| 75 | A test proves the Group-A seam cannot become permanent | **yes** — unpatched rejection + registry assertions |
| 76 | Vanilla/Base pathway semantics unchanged; no segmentation | **yes** — the test asserts `canon(x)` / `b(canon(x))` and no underscore |
| 77 | Stage-1 adapted-grid contract unchanged | **yes** — fixture repaired; the guard rejects only what already crashed |
| 78 | Missing-transformers negative test is deterministic | **yes** — meta-path blocker, not PATH |
| 79 | The negative test cannot enter the probe success path | **yes** — asserts no output dir and no load message |
| 80 | *(historical, Rev 3)* Generated Colab probe artifacts not committed | **yes, and still true** — but the Rev-3 **rationale was wrong**: I wrote that `results/` is not ignored, having checked the *directory* (re-included by `!results/**/`) rather than a file under it. H1 established that `.gitignore:15 `` results/** `` **does** match `results/b3b1/20260821T104151Z/*`, so the known accidental probe **is gitignored** (mtime 2026-08-21 10:41:51 UTC). The R3C and H1 worktrees were clean; its download-vs-cache status remains **NOT ESTABLISHED** |
| 81 | One production change, additive, classified as hardening | **yes** — `+26/−0`, zero removed lines |
| 82 | *(historical, Rev 3)* Local counts reported honestly | **yes** — **2 310 / 89 at that revision**; 1 of 8 repairs ran locally. Current: 2 312 / 89 |
| 83 | *(historical, Rev 3)* Real-model smoke and real-data dry-run still pending | **correct then; SUPERSEDED by Revision 4** — **CLOSED** by C24-4-R2 and C24-5B-R1 |
| 84 | **Targeted** Audit-024 Torch gate | **PASS / CLOSED** — 136/0/0 at `52ebca06…`, file unmodified since |
| 85 | **Repository-wide** Torch regression | *(historical, Rev 3)* PENDING then — **CLOSED by C24-1-R3C: 2 401 / 0 / 0** |
| 86 | The two gates are never conflated | **yes** — §A, §I, §J, §O and the status block state them separately |
| 87 | No claim upgraded beyond what C24-1-R2 executed | **yes** — every `TORCH-PASS` maps to a test in `tests/test_preg1_head.py`, verified unmodified since `52ebca06` |
| | **— Revision 3a: evidence-consistency sweep —** | |
| 88 | §F distinguishes LOCAL provenance logic from TORCH-PASS on-disk path | **yes** |
| 89 | Every §I row agrees with the §I summary | **yes** — Q4/Q5 `TORCH-PASS`; Q6/Q7/Q11/Q12/Q13 `LOCAL + TORCH-PASS`; no row says PENDING |
| 90 | Each upgraded label verified against an actual torch-gated test before relabelling | **yes** — 14 named tests confirmed `@requires_torch` and inside the 28 |
| 91 | §A no longer calls "the torch runtime" unverified | **yes** — names the **repository-wide** regression |
| 92 | No-download claim scoped; accidental-probe status left **NOT ESTABLISHED** | **yes** — type row, §K, §N row 8, status block |
| 93 | Historical rows scoped, history not rewritten | **yes** — rows 44, 46, 47, 60, 61, 65 marked SUPERSEDED |
| 94 | No code or test changed in this pass | **yes** — documentation only |
| | **— Revision 3b: C24-1-R3B —** | |
| 95 | Root cause classified from the actual code | **yes** — encoder vocab 64 vs ids to 4099, recomputed on the fixture |
| 96 | No production change | **yes** — `git diff` on `unmark/` and `scripts/` is empty |
| 97 | `StubTokenizer` remains accent-sensitive | **yes** — `Tôi`≠`Toi`, `đang`≠`dang`, `học`≠`hoc` |
| 98 | Tokenizer ids and encoder vocabulary share an explicit contract | **yes** — one `STUB_VOCAB_SIZE`; ids in-range by construction |
| 99 | Both new guards mutation-verified, and both run locally | **yes** — hard-coded 64 fails the AST test; `% 4093` fails the range test |
| 100 | The prior exact eight repairs preserved and passing on Torch | **yes** — 8/8 in C24-1-R3B |
| 101 | Targeted 136/0/0 history preserved and reproduced | **yes** — §Q.1 and §R |
| 102 | Full R3 suite recorded as **NOT REACHED**, not failed | **yes** — §R step 4 |
| 103 | The eight harness repairs **not** claimed to work | **yes** — torch-gated, unverified; only the fixture accident is excluded |
| 104 | No new scientific decision | **yes** — a fixture repair is not a decision |
| | **— Revision 4: FINALIZATION —** | |
| 105 | C24-1-R3C **2 401 / 0 / 0** recorded; repository-wide gate **CLOSED** | **yes** — §S.1 |
| 106 | The R3C `results/` stop classified as stale historical probe hygiene, **not** a test failure | **yes** — mtime 2026-08-21 10:41:51 UTC, gitignored, worktree clean |
| 107 | Historical accidental probe download/cache status still **NOT ESTABLISHED** | **yes** — §S.1 and §Q.4 Group C |
| 108 | C24-2 committed-API discovery recorded; `train_head` has no free role/pathway | **yes** — §S.2 |
| 109 | C24-3 Drive membership verified **by locked SHA, not filename** | **yes** — five digests, 9139/2285/11424, assignment digest |
| 110 | C24-4-R1 classified as **external smoke-harness metadata-discovery failure** | **yes** — §S.4; not a production defect |
| 111 | C24-4-R2 real PhoBERT integration **PASS**; exact model/revision/tokenizer/class recorded | **yes** — §S.5 |
| 112 | `FIRST_TOKEN` proved **exactly equal** to real `last_hidden_state[:, 0, :]`, both pathways | **yes** — §S.5 |
| 113 | Real cache round-trip **and** cross-pathway refusal recorded | **yes** — §S.5 |
| 114 | C24-5B attempt 1 classified as a **one-character external-harness SHA typo**, not a dataset mismatch | **yes** — §S.7; no repository code changed |
| 115 | C24-5B-R1: four official TRAIN/DEV raw hashes recorded | **yes** — §S.8 |
| 116 | Derived TRAIN SHA / rows / bytes / labels recorded; `load_derived_pool` gate passed | **yes** — `a20c0f77…`, 11 424, 1 067 331, 5 324/458/5 642 |
| 117 | Derived official DEV SHA / rows / bytes recorded | **yes** — `9c475c89…`, 1 583, 139 001 |
| 118 | Audit-023 membership 9 139 / 2 285 / 11 424 + assignment digest recorded | **yes** |
| 119 | Membership → real-row exact binding recorded | **yes** — every id resolves to one row; none missing, none duplicated |
| 120 | Official validation refused where protocol-dev is required | **yes** — `SplitLeakage`, on real membership |
| 121 | Protocol-dev refused where official validation is required | **yes** — `SplitLeakage` |
| 122 | Official TEST content never selected, downloaded, read or scored | **yes** |
| 123 | No optimizer, head training, LR tuning or downstream score in any C24 run | **yes** |
| 124 | Current verdict no longer says repository / model / data verification pending | **yes** — §A |
| 125 | Historical failed attempts preserved in §§P, Q, R | **yes** — history not rewritten |
| 126 | No new scientific decision invented by finalization | **yes** — §S.9 |
| 127 | No production source or test changed in this pass | **yes** — documentation only |
| | **— Revision 4a: current-state consistency —** | |
| 128 | §A no longer denies C24-4 / C24-5 execution | **yes** — separates *did not happen* (head training, LR tuning, score, burden diagnostic, Stage-1) from *did happen as verification* (real PhoBERT smoke, real-data role-boundary dry-run) |
| 129 | §J no longer says no real model or corpus was ever used | **yes** — chronology: C24-1 = test/runtime only; C24-4-R2 = real PhoBERT; C24-5B-R1 = transient official TRAIN/DEV bytes |
| 130 | §J no longer says the Revision-3 repository repairs are unexecuted | **yes** — 8/8 in R3B, 74/74 and 2 401/0/0 in R3C |
| 131 | The same false "no real model" claim repaired in §K | **yes** |
| 132 | Current local count is **2 312 passed / 89 skipped** | **yes** — §A and §N row 27 |
| 133 | Historical 2 310 / 89 retained only where labelled historical | **yes** — §N row 82 and §Q.6, both scoped to Revision 3 |
| 134 | Status block separates downstream diagnostic from verification-only execution | **yes** — and the duplicate `STAGE-1` key was removed |
| 135 | Neither C24-4 nor C24-5 is implied to have produced a downstream result | **yes** — stated explicitly in §A, §J and the status block |
| 136 | Accepted Revision-4 evidence preserved unchanged | **yes** — 27-point gate: 2 401/0/0, 136/136, all hashes, membership, refusals, classifications, NOT ESTABLISHED |
| 137 | No production or test source changed in this pass | **yes** — documentation only |
| | **— Revision 4b: four residual items —** | |
| 138 | §K no longer says no real representation was ever extracted | **yes** — C24-4-R2 extracted from three **synthetic** smoke inputs, verification only |
| 139 | §K still correctly states no real-data head training occurred | **yes** — and no real UIT-VSFC representation for downstream training or measurement |
| 140 | §N row 48 marked historical / SUPERSEDED, not "still current" | **yes** |
| 141 | §N row 83 marked historical / SUPERSEDED | **yes** — same style as rows 82 and 85 |
| 142 | §N row 80 no longer claims `results/` was unignored | **yes** — and the wrong rationale is recorded rather than deleted |
| 143 | The gitignore correction was **verified**, not accepted on assertion | **yes** — `git check-ignore` on a file under a run directory returns `.gitignore:15 results/**` |
| 144 | Historical accidental-probe facts preserved; download-vs-cache still **NOT ESTABLISHED** | **yes** |
| 145 | All four verification gates remain **CLOSED**; local count remains **2 312 / 89** | **yes** |
| 146 | No accepted Revision-4 evidence altered | **yes** — 20-point gate re-run |

---

## O. NEXT ACTION — THE PRE-G1 DIAGNOSTIC

**All four verification gates are closed** (§S). Steps 1–3 of the earlier
sequence — repository-wide Torch, real-model integration, real-data boundary —
are discharged; their evidence is in §S.1, §S.5 and §S.8, and the failed
attempts that preceded them are preserved in §§P, Q and R.

**Nothing below was run in this task.**

### Only after researcher review and commit of this finalization

The locked pre-G1 Vanilla-vs-Base-only burden diagnostic:

| Phase | Runs |
|---|---|
| Vanilla LR tuning — 5 learning rates × 3 tuning seeds | **15** |
| *(freeze the winning Vanilla LR)* | — |
| Paired final measurement — 5 Vanilla + 5 Base-only | **10** |
| **Total** | **25 frozen-representation linear-head training runs** |

**These train only the linear head. The pretrained encoder stays frozen.**

The paired result will be the **first downstream number in this project**. It
measures a burden. It does not test UNMARK, and it must not be reported as if it
did.

---

## P. C24-1-R1 — FIRST REAL TORCH EXECUTION, AND ITS TWO FAILURES

**This attempt failed and is recorded, not hidden.** It is the first time any of
the 28 torch-gated tests executed, and it earned its place in the evidence
history by finding two real defects — both in the tests.

| | |
|---|---|
| **Run id** | `C24-1-R1` |
| **Execution HEAD** | `b43cca829ca163b9d6a818c980dfbf6fdaea651f` |
| **Python / torch** | 3.12.13 / 2.11.0+cu128 |
| **transformers / pytest** | 4.57.6 / 9.1.1 |
| **CUDA** | available |
| **GPU** | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| **Targeted outcome** | **132 passed, 2 failed, 0 skipped** |
| **Status** | **FAILED — deliberately fail-fast; no closure** |

**Evidence status.** Externally observed on Colab and supplied to this session.
The run was not reproduced here; the local environment has no torch.

Note what the zero skips mean: with torch present, all 134 tests executed, so
**the torch-gated bucket is no longer hypothetical** — it runs, and 132 of it
passed. That is genuinely more than was known before, and it is still not
closure.

### Failure 1 — `test_train_head_refuses_non_fp32_representations`

**Observed.** Train representations `torch.float64`, dev `torch.float32`. The
exception raised was

> `representation sets disagree on dtype: 'torch.float64' vs 'torch.float32'`

while the test matched on `"FP32"`.

**Classification: TEST ISOLATION DEFECT.** Not an implementation defect —
verified by reading `train_head`, which contains an independent and explicit
FP32 guard *after* role and geometry validation:

```
require_training_roles(train, dev)          # includes dtype AGREEMENT
...
if train_features.dtype is not torch.float32 or dev_features.dtype is not torch.float32:
    raise EvaluationContractViolation("features must be FP32 (no AMP)")
```

The fixture violated **two** rules at once — the sets must agree on dtype, and
the dtype must be FP32 — and the first fired. The test therefore proved the
agreement rule while claiming to prove the FP32 rule. It could have passed for
the wrong reason on a different day.

**Repair — the fixture, not the assertion.** Both sets are now **mutually
consistent float64**, and the repaired test *first asserts that
`require_training_roles` does **not** raise* before expecting the FP32 refusal.
That positive assertion is what makes the isolation checkable rather than
asserted: if a future change made the pair disagree again, the test fails at the
isolation step instead of silently passing for the old reason.

**The regex was not loosened.** Widening it to accept any dtype-related
exception would have made the test pass while proving less. **The validation
order was not changed either**: agreement between the two sets is a precondition
for comparing either against an absolute contract, so checking it first is
principled — reordering it to satisfy an error-message match would have been the
tail wagging the dog.

**Coverage added, not lost.** The property the broken test was accidentally
proving now has its own home: `dtype` was added to the parameterised
geometry-mismatch test, so a **train/dev dtype mismatch still fails closed** as a
geometry disagreement. That test is torch-free and **executes locally**.

### Failure 2 — `test_cache_saves_and_reloads_under_matching_metadata`

**Observed.** The test asserted `torch.equal(cache.load(sample_key()), tensor)`;
`load` returned a `BoundRepresentations`.

**Classification: STALE TEST after intentional API strengthening.** Revision 1
deliberately changed `RepresentationCache.load` to return the values *with* the
`RepresentationKey` it validated — the whole point being that a caller must not
carry values away from their provenance. The assertion predated that change.

**Repair.** The test now verifies the current contract: the return is **not** a
bare tensor, it is a `BoundRepresentations`, its key equals the validated key,
pathway and source/ordered-id identity are preserved, the values round-trip
bit-exactly, and a **second** load is equally bound — so the binding is not a
one-shot wrapper.

**`load` was not reverted to a bare tensor.** That would have been the easy fix
and would have undone the Revision-1 repair. To make the temptation
self-defeating, a **torch-free AST test** now asserts that `load`'s return
annotation is `BoundRepresentations` and that every `return` in it constructs
one — so a regression is caught in the **local** suite, not deferred to the next
Colab run.

### What C24-1-R1 changed, and what it did not

| | |
|---|---|
| Implementation lines changed | **zero** — `git diff` on `unmark/` and `scripts/` is empty |
| Test file | repaired; **136** tests now (134 + 2) |
| Scientific values | **none touched** |
| Verdict | **unchanged** — still not runtime-closed |

*(Written at Revision 2, and now discharged.)* **At that point** C24-1 runtime
closure remained PENDING, requiring a new committed revision on which all 136
tests executed successfully on Colab. **C24-1-R2 did exactly that** — 136/0/0 at
`52ebca06…`, including both repaired tests (§Q.1). The targeted gate is closed;
what remains open is the *repository-wide* regression, which is a different
gate.


---

## Q. C24-1-R2 / R2B — TARGETED PASS, REPOSITORY-WIDE REGRESSION

### Q.1 Targeted Audit-024 suite — **PASS**

| | |
|---|---|
| **Run id** | `C24-1-R2` |
| **Execution HEAD** | `52ebca0657dc39b8e8e25ebd71c90ae7a4687501` |
| **Python / torch** | 3.12.13 / 2.11.0+cu128 |
| **transformers / pytest** | 4.57.6 / 9.1.1 |
| **CUDA / GPU** | available / NVIDIA RTX PRO 6000 Blackwell Server Edition |
| **`tests/test_preg1_head.py`** | **136 passed / 0 failed / 0 skipped** |
| **Load-bearing subset** | 22 passed / 114 deselected |

**This is a real pass and it closes the targeted gate.** Both C24-1-R1 failures
are fixed, and the 28 previously torch-gated tests — including the
paired-initialisation guarantee and every frozen-encoder check — executed and
passed on GPU.

### Q.2 Resource recovery — pinned inventory

The repository-wide suite first failed partly because the pinned Vietnamese
syllable inventory was absent. It was fetched **only through the repository's
own pinned fetcher** and verified:

| | |
|---|---|
| Revision | `135a4d9716e49a981624474156d6f247b9b46f6a` |
| SHA-256 | `78eeb840d50455b14bd564da5aed7318d96468b8deaad5986b77bf5c538315d2` |
| Raw / unique canonical / unique stripped | 17 974 / 17 954 / 2 489 |
| License | `NO_EXPLICIT_LICENSE` |

**Classification: ENVIRONMENT / RESOURCE REQUIREMENT, not a code defect.** Every
inventory and corruption error disappeared once the exact pinned bytes were
present. No inventory logic was touched, the cache stays git-ignored, the raw
file is **not committed**, and **no decision is warranted** by a resource
recovery.

### Q.3 Repository-wide suite — **NOT YET PASS**

`C24-1-R2B`, after inventory recovery: **2 383 passed / 8 failed / 0 errors.**

**Neither failed attempt is hidden.** R1 (132/2/0) is §P; R2B (2383/8) is here.

---

### Q.4 The eight failures — classified after inspecting the code

**Not one was an implementation defect.** Six were fixtures that predated
fail-closed hardening, one was a non-discriminative stub, one was an
environment-sensitive test.

| # | Test | Classification |
|---|---|---|
| 1 | `test_runtime_wrapper_supplies_positions_automatically` | **stale test fixture** |
| 2 | `test_runtime_pathways_produce_different_token_ids` | **test-isolation defect** (non-discriminative stub) |
| 3 | `test_probe_refuses_locally_without_transformers` | **environment-sensitive test defect** |
| 4 | `test_runtime_train_mode_invariants` | **stale test fixture** |
| 5 | `test_runtime_encoder_stays_frozen_across_mode_changes` | **stale test fixture** |
| 6 | `test_runtime_gradients_reach_the_adapter_through_a_stand_in_encoder` | **stale test fixture** |
| 7 | `test_runtime_gate_weight_gradient_exists_despite_zero_init` | **stale test fixture** |
| 8 | `test_runtime_unequal_reference_and_base_lengths_work` | **stale test fixture** |

#### Group A (1, 4, 5, 6, 7) — verified position profile vs synthetic encoders

**Root cause, read from the code.** `UnmarkEncoder.__init__` calls
`resolve_position_profile(encoder)`, which matches on the **whole** profile —
checkpoint, model type **and** class. `VERIFIED_POSITION_PROFILES` has exactly
one entry, because exactly one backbone was measured. The synthetic
`RobertaLike` / `TinyEncoder` stand-ins match nothing, so construction raises
`UnsupportedPositionSemantics`. **That is production working as designed** —
D-B4B-002, with D-B3B0-002 OPEN. The fixtures simply predate the hardening.

**Repair — test-only, narrowest seam.** A helper builds a
`VerifiedPositionProfile` **locally inside the test** and monkeypatches only
`resolve_position_profile`. The object is **never added to
`VERIFIED_POSITION_PROFILES`**, which still contains exactly one entry
(verified by AST: `(PHOBERT_BASE_POSITION_PROFILE,)`).

**Nothing was weakened.** No fallback rule, no family-wide acceptance, no fake
encoder registered, D-B3B0-002 untouched. The fail-closed tests
(`…rejects_an_unverified_backbone`, `…other_family_is_rejected`,
`…unexpected_model_class_is_rejected`) run **unpatched** and still pass.

**Isolation preserved, and checked.** For test 1 the property is the position
*arithmetic*, so the injected profile must carry
`position_rule="roberta_input_ids_offset"` — the test asserts that before
proceeding, and still asserts the exact ids `[[2, 3, 4, 5, 1]]`. Only the
*permission* is stubbed; the computation is untouched. Tests 4–7 concern
train/eval mode, freezing and gradient connectivity, which are independent of
which backbone is permitted. The stand-ins also gained a real
`config`/`padding_idx`, so `detect_padding_index` is genuinely exercised rather
than bypassed.

**A new test stops the seam becoming permanent**:
`test_runtime_the_patched_fixture_is_still_rejected_unpatched` constructs the
same `RobertaLike` **without** the patch, requires
`UnsupportedPositionSemantics`, and asserts the registry still has one entry and
contains neither stand-in.

#### Group B (2) — the stub tokenizer could not represent the difference

**Root cause.** `StubTokenizer` computed ids as `7 + (len(token) % 5)` — keyed
on **character count alone**. NFC Vietnamese diacritics do not change a token's
length, so `"Tôi"`→`"Toi"` and `"đang"`→`"dang"` produced *identical* ids.
Verified directly: the pathway **texts** differ correctly
(`'Tôi đang học nghiên cứu'` vs `'Toi dang hoc nghien cuu'`) while the id lists
were equal. **Production preprocessing was right; the fixture could not observe
it.**

**Repair.** Ids now derive from the token's codepoints
(`crc32(token.encode())`), so the stub is accent-sensitive by construction. More
importantly the test now proves the scientific claim **independently of the
tokenizer**: for every fixture example it asserts Vanilla text `== canon(x)`,
Base-only text `== b(canon(x))`, that the two differ, and that neither contains
an underscore (**no segmentation**). The token-id assertion is retained as a
plumbing check, and a separate torch-free test pins that the stub really is
accent-sensitive — so the fixture cannot silently regress.

#### Group C (3) — the "without transformers" test ran the probe

**Root cause.** The subprocess used `sys.executable` and constrained only
`PATH`. On Colab `sys.executable` was `.venv-colab/bin/python`, which **has**
transformers — so the assumed environment never existed, the probe took its
success path, returned **0** instead of 2, and printed a results directory
`results/b3b1/20260821T104151Z`. Constraining `PATH` cannot help when the chosen
interpreter already owns the package.

**Classification: environment-sensitive test defect.**

**Repair — deterministic, and it never reaches the success path.** The probe's
dependency guard is inline in `main()`, so there is no callable to test
directly; option 2 was taken. The subprocess now runs a bootstrap that installs
a `sys.meta_path` finder raising `ImportError` for `transformers`, then executes
the probe via `runpy`. Project imports still resolve; the guarded import fails
**before any model is named, fetched or loaded**. No skip-if-installed, no
accepting 0, no uninstalling, no model download.

**Two new tests protect the mechanism.**
`test_the_negative_dependency_test_cannot_enter_the_success_path` runs with
`--output-root` in a temp dir and asserts the directory **was never created**,
that `"Loading slow tokenizer"` never appears, and that the exit code is 2.
`test_the_import_blocker_works_on_a_package_that_is_actually_installed` blocks
**pytest** — certainly installed — and asserts it becomes unimportable, because
locally `transformers` is absent anyway and blocking it would prove nothing.

**What the unintended Colab execution can and cannot be established to have
done.** From the available evidence — exit code 0 and a printed path — the probe
ran past its dependency guard and reported an output directory. **The directory
is not present in this repository**, so its contents were not inspected here.
Therefore:

- **Established:** the guard was not reached; the run proceeded; a path under
  `results/b3b1/` was reported.
- **Not established:** whether a model or tokenizer was downloaded or loaded from
  cache, what the directory contains, or whether it holds a meaningful probe
  result. **This audit does not guess.**
- **Not treated as a scientific result.** It is an accidental artifact of a
  broken negative test. `results/` is **not** git-ignored, so a generated run
  directory appears as untracked and **must not be staged**. None is present
  here (`git status` is clean of it).
- **No scientific state changed:** the probe is tokenizer-only, writes only under
  its output root, and produces no downstream score.

#### Group D (8) — the unequal-length fixture broke the adapted grid

**Root cause.** `synthetic_batch(reference_len, base_len)` parameterised the
reference and base **ids** but left the orthography channels hard-coded at
length 5. So `synthetic_batch(reference_len=11, base_len=4)` produced base ids at
L=4 beside tone/letter channels at L=5, and the adapter failed inside
`torch.cat`. The test appeared to say "unequal reference/base lengths do not
work" — the opposite of the Stage-1 contract.

**Repair.** The channels now follow `base_len`; `reference_len` stays
independent; the test still proves `reference_len != base_len` works.

**One production change — implementation hardening, not science.** The adapter
validated each channel's *rank* but never that base embeddings, tone and letter
channels share the same `[B, L]` grid, so a misalignment surfaced only as a raw
`torch.cat` size error. A `ChannelContractViolation` now names both grids and
states the rule: only the Stage-1 **reference** branch may differ in length;
everything on the adapted grid must agree.

**Why this is not a scientific change:** it rejects only inputs that would
already have crashed, alters no equation, value, gate, objective or contract, and
**broadens nothing**. Four parameterised tests cover it, plus the aligned case
still passing. The diff is **26 insertions, 0 deletions**.

---

### Q.5 What changed

| File | Change | Kind |
|---|---|---|
| `tests/test_b4b_provenance_and_positions.py` | test-only profile seam + anti-permanence test | test |
| `tests/test_neural_adapter.py` | test-only profile seam ×4, real `config`/`padding_idx`, grid-agreement tests | test |
| `tests/test_evaluation_harness.py` | accent-sensitive stub, pathway-text assertions, stub guard | test |
| `tests/test_manual_alignment.py` | meta-path blocker + 2 protective tests | test |
| `tests/test_stage1.py` | channels follow `base_len` | test |
| `unmark/modeling/adapter.py` | **grid-agreement guard (additive, +26/−0)** | **implementation hardening** |

**No scientific value changed**: architecture, adapter equations, gate init,
Stage-1 objective, RAW_BASE, tokenizer policy, model revision, `max_length`, LR
grid, seeds, optimiser, epochs, batch size, split membership, Audit-023 hashes,
official-validation role and the TEST seal are all untouched.

### Q.6 Local verification — honest counts

| | |
|---|---|
| Full local suite *(at Revision 3)* | **2 310 passed, 89 skipped** (2 307 / 84 before; **2 312 / 89** after Revision 3b) |
| `tests/test_preg1_head.py` | **108 passed, 28 skipped** — unaffected |
| Of the eight repaired tests | **1 runs locally and passes** (Group C); **7 remain torch-gated** |

**The seven torch-gated repairs are not claimed to work.** They are written
against failures that were actually observed, which is better evidence than
guesswork, and they have not executed. Only Colab can settle them.

### Q.7 Evidence state — do not conflate these

| | |
|---|---|
| **Audit-024 targeted Torch contract tests** | **PASS / CLOSED** — 136/136 at `52ebca06…` |
| **Repository-wide Torch regression** | *(PENDING at Revision 3)* — **CLOSED by C24-1-R3C**, 2 401 / 0 / 0 (§S.1) |
| **Real PhoBERT integration smoke** | *(PENDING at Revision 3)* — **CLOSED by C24-4-R2** (§S.5) |
| **Real-data boundary dry-run** | *(PENDING at Revision 3)* — **CLOSED by C24-5B-R1** (§S.8) |

**"C24-1 fully passed" and "all Torch verification closed" are both false** and
appear nowhere in this audit.

### Q.8 Next Colab rerun requirements

On a **new committed revision**:

1. `tests/test_preg1_head.py` → **136 passed / 0 failed / 0 skipped** (regression).
2. Repository-wide suite → **0 failed**, with the pinned inventory present via
   the repository fetcher only.
3. Confirm the eight repaired tests pass **and** that the four fail-closed
   position tests, the anti-permanence test, the stub-sensitivity test and the
   negative-dependency protection tests all pass unpatched.
4. Confirm no `results/…` directory is staged.

Only then do the real-model smoke and the real-data dry-run follow.

**NO NEW SCIENTIFIC DECISION WARRANTED.** `docs/spec/decisions.md` was reviewed:
test repairs, fixture repairs, a resource recovery and additive implementation
hardening are none of them scientific or specification decisions.


---

## R. C24-1-R3B — THE REPAIRS PASS, A SHARED FIXTURE DID NOT

| | |
|---|---|
| **Run id** | `C24-1-R3B` |
| **Execution HEAD** | `f6ad4eef21cff9826f6ba58191a3dbcfcbed491a` |
| **Evidence status** | externally observed on Colab; not reproduced here |

| Step | Result |
|---|---|
| 1. `tests/test_preg1_head.py` | **136 passed / 0 failed / 0 skipped — PASS** |
| 2. the exact eight R3 repairs | **8 passed — PASS** |
| 3. broader R3 related/safety group | **FAILED — 8 failed / 328 passed** |
| 4. full repository suite | **NOT REACHED** — step 3 failed first |

**Step 4 is *not reached*, not *failed*.** The repository-wide gate has no
result at this HEAD, and this audit does not manufacture one.

**Both prior gates held.** The targeted suite reproduced 136/0/0, and all eight
Revision-3 repairs — six stale fixtures, the non-discriminative stub and the
environment-sensitive negative test — passed on GPU. The eight new failures are
**different tests**, in `tests/test_evaluation_harness.py`:

`test_runtime_extraction_returns_unpooled_hidden_states`,
`…masks_travel_with_the_hidden_states`,
`…scientific_path_cannot_reach_masked_mean`,
`…test_only_pooling_works_for_diagnostics`,
`…encoder_is_not_mutated_by_extraction`,
`…cross_pathway_head_reuse_is_refused`,
`…dev_representations_cannot_train_a_head`,
`…task_mismatch_is_refused`.

All eight died identically: `IndexError: index out of range in self`, inside the
synthetic encoder's `nn.Embedding` lookup.

### Root cause — verified from the code, and it is mine

`stub_encoder` declared `nn.Embedding(64, d, padding_idx=1)`. My Revision-3
change made `StubTokenizer` accent-sensitive with
`7 + crc32(token) % 4093`, emitting ids up to **4099**. The previous rule,
`7 + len(token) % 5`, emitted ids in **[7, 11]** — always inside 64.

Recomputed here from the actual fixture: the old rule's maximum id is **11**;
the new rule produces **12, 14, 233, 291, 1070, 1097, 2239, 2282, 2433, 2536,
2593, 2886, 2905, 2987, 3399, 3739** — matching the values the run reported.

**The two halves of one test double had an implicit, undocumented vocabulary
agreement, and I changed one side.** The eight tests never reached the property
each of them names; they died in the fixture.

**Classification: TEST-FIXTURE INTERACTION / REGRESSION** — introduced by
Revision 3's own fixture change. **Not a production defect and not a scientific
defect.** No production code was inspected into suspicion and none was changed.

### Repair — one explicit, single-sourced contract

The agreement is no longer implicit:

```python
STUB_VOCAB_SIZE = 512          # the one number both halves are built from
STUB_PAD_ID = 1                # also the encoder's padding_idx
STUB_FIRST_CONTENT_ID = 7      # below this: special/pad

def stub_token_id(token):
    span = STUB_VOCAB_SIZE - STUB_FIRST_CONTENT_ID
    return STUB_FIRST_CONTENT_ID + zlib.crc32(token.encode("utf-8")) % span
```

`StubTokenizer` produces every id through `stub_token_id`; `stub_encoder` builds
`nn.Embedding(STUB_VOCAB_SIZE, …)`. **Ids are in-vocabulary by construction** —
the modulus is *derived from* the bound, so the function cannot emit an id the
encoder cannot embed.

**Why this design.** The alternative fixes — enlarging the encoder, or clamping
ids at the call site — would each have restored the *behaviour* while leaving
the *coupling* implicit and free to drift again. A single constant that both
halves are built from removes the class of defect, not the instance. It also
keeps the double small: the encoder still emulates nothing.

**Accent sensitivity is retained**, which was the whole point of the Revision-3
change. Verified on the fixture: `Tôi`≠`Toi`, `đang`≠`dang`, `học`≠`hoc`, and
the per-sentence id sequences still differ between pathways. Maximum id emitted
by the fixture is **473**, comfortably inside 512.

### Two new fixture-level tests — and both run locally

- `test_every_stub_id_is_inside_the_stub_encoder_vocabulary` walks the **actual
  fixture texts under both pathways**, plus the special and pad ids, and asserts
  each is in `[STUB_FIRST_CONTENT_ID, STUB_VOCAB_SIZE)`.
- `test_the_two_halves_of_the_test_double_share_one_vocabulary_constant` asserts
  by AST that the embedding's size argument is the **name** `STUB_VOCAB_SIZE`,
  not a literal — a hard-coded size is precisely how the halves drifted.

Both were **mutation-verified**: restoring `nn.Embedding(64, …)` fails the AST
test; restoring `% 4093` fails the range test. Neither needs torch, so **this
class of mismatch is now caught in the ML-free local suite** rather than costing
a GPU round trip.

### Do the eight tests now reach their named properties?

They are torch-gated, so they still skip locally and **their repair is
unverified**. What changed is that the *fixture accident* that stopped them is
now excluded by an invariant checked locally — the embedding lookup can no
longer be the reason they fail. Whether each then proves its named property is
for the rerun to establish.

### Scope

| | |
|---|---|
| Files changed | `tests/test_evaluation_harness.py` only |
| Production changed | **none** — `git diff` on `unmark/` and `scripts/` is empty |
| Grid guard from Revision 3 | untouched; not implicated |
| Scientific values | none touched |
| New decision | **none warranted** — a fixture repair is not a decision |

### Current evidence state

| Gate | State |
|---|---|
| Audit-024 targeted Torch | **PASS / CLOSED** — 136/136, reproduced at `f6ad4eef…` |
| The exact eight R3 repairs | **PASS** — 8/8 on Torch |
| Broader R3 related/safety group | **FAILED 8/328** at R3B — repaired; **C24-1-R3C then passed 74/74** |
| Full repository R3 suite | **NOT REACHED** at R3B — **C24-1-R3C: 2 401 / 0 / 0** |
| Real PhoBERT integration | *(PENDING then)* — **CLOSED by C24-4-R2** (§S.5) |
| Real-data boundary dry-run | *(PENDING then)* — **CLOSED by C24-5B-R1** (§S.8) |
| Scientifically closed as an implementation / pre-diagnostic audit | **YES** (§S.9) — the burden result still does not exist |


---

## S. FINALIZATION — ALL FOUR GATES CLOSED

**Final verification commit:** `767fb8ee30fa9b5df344a3629b3295c1fd03c379`
("test: align evaluation fixture vocabulary").

Environment for every run below: Python 3.12.13, pytest 9.1.1,
torch 2.11.0+cu128, transformers 4.57.6, CUDA available, NVIDIA RTX PRO 6000
Blackwell Server Edition. Pinned syllable inventory
`78eeb840d50455b14bd564da5aed7318d96468b8deaad5986b77bf5c538315d2`.

**Evidence status.** Externally observed on Colab and supplied to this session;
not reproduced here — the local environment has no torch and holds no corpus.

### S.1 C24-1-R3C — repository-wide Torch closure

| Step | Result |
|---|---|
| `tests/test_preg1_head.py` | **136 passed / 0 failed / 0 skipped** |
| the eight evaluation-harness tests that failed in R3B | **8 passed** |
| complete `tests/test_evaluation_harness.py` | **74 passed** |
| **full repository Torch suite** | **2 401 passed / 0 failed / 0 errors** |

**This closes the repository-wide Torch regression gate.** Every repair from
Revisions 2, 3 and 3b now has runtime evidence.

**The `results/` stop was not a test failure.** The first R3C shell halted after
the suite on an over-broad `results/`-directory-existence guard. Classification
established: the worktree is clean; tracked `results/**/.gitkeep` files are
expected; `results/b3b1/20260821T104151Z/*` is ignored by `.gitignore`; its mtime
is **2026-08-21 10:41:51 UTC** — it is the already-known historical accidental
B3B1 probe from Revision 3 (§Q.4 Group C), **not an artifact R3C created**. The
test result is **2401 / 2401 PASS**.

**Unchanged:** whether that historical accidental probe downloaded assets or
loaded them from cache remains **NOT ESTABLISHED**. Nothing in R3C bears on it.

### S.2 C24-2 — committed API discovery (source only)

**PASS.** No model, data, training or score.

`Preg1Role` is exactly `PROTOCOL_TRAIN` / `PROTOCOL_DEV` /
`OFFICIAL_VALIDATION` — **no `OFFICIAL_TEST`**. `RepresentationKey` carries
dataset, dataset_version, task, role, pathway, source_identity,
ordered_id_digest, tokenizer_id, model_revision, max_length, truncation,
padding, pooling, dtype, hidden_size, count, schema_version.
`BoundRepresentations` is `values` + `key`.

Committed signatures confirmed:

```
extract_bound_representations(encoder, input_ids, attention_mask, key) -> BoundRepresentations
train_head(train, train_labels, dev, dev_labels, *, learning_rate, seed,
           epochs=30, batch_size=128, on_epoch=None) -> HeadRun
score_measurement(head, measurement, labels) -> tuple[float, float]
RepresentationCache.save(key, tensor) / .load(key) -> BoundRepresentations
```

**`train_head` has no free `role`, `dev_role`, `pathway` or `dev_pathway`** — the
Revision-1 provenance binding is confirmed on the committed source, not just in
this audit's prose.

Locked values reconfirmed: `ENCODER_REVISION` `01daacda68afe13d83023d16ec647239e344a1e6`,
`MAX_LENGTH` 256, `PADDING` `max_length`, `TRUNCATION` True, `PREG1_POOLING`
`FIRST_TOKEN`, `EPOCHS` 30, `BATCH_SIZE` 128, LR grid and both seed sets
unchanged.

### S.3 C24-3 — Audit-023 Drive membership verification

**PASS.** Directory
`…/preg1-uit-vsfc-internal-split/preg1-split-v1-66f4522a-7bd5d189`.

**Five files identified by locked SHA, not by filename** — the right way round,
since a filename is an assertion and a digest is evidence:

| File | SHA-256 |
|---|---|
| `protocol-train.ids.txt` | `275ae66d16582418093a1f4500904faefedd5936bb5cf383c52be302e151172e` |
| `protocol-dev.ids.txt` | `d342950ae183e6c08bfeecaeacfb0e42aaf3751c12dec0baf0ca515922ca5e31` |
| `split-manifest.json` | `225b109ea5fa58476e98bdf050a42ca89f12c6df02b37a882dc09cdc958b3685` |
| `report.md` | `17a9a6f116b1277bc063ff53d0840e20cae1b034177d6d2ae014a6428ee20459` |
| `runtime-environment.json` | `0ed15fc3f717e1d316194021969ec6fc8288073de99a051980d94f6b86bc2c6e` |

protocol-train **9 139**, protocol-dev **2 285**, union **11 424**; no
duplicates, no cross-split overlap; ids in the `train:NNNNN` namespace; the
conflicting group `train:11293` / `train:11417` excluded. Assignment digest
`7bd5d1892e23b96035c376936d2168f547661da07b8abc769f5656e9438f4f84`, and the
manifest parsed and contains that exact digest. **No CSV persisted** in the
membership directory. No model, training or score.

### S.4 C24-4-R1 — first PhoBERT smoke attempt

**Classification: external smoke-harness metadata-discovery failure. Not a
production defect, and not a model failure.**

Before stopping it established: repository import from repo root;
`vinai/phobert-base` @ `01daacda68afe13d83023d16ec647239e344a1e6`; real
`PhobertTokenizer` and `RobertaModel`; hidden size 768; `max_length` 256;
padding `max_length`; position 0 = BOS/`<s>`; Vanilla and Base-only token grids
differ; encoder frozen and in eval.

It stopped because the **external harness** tried to infer the
`RepresentationKey` dtype spelling and **refused to guess** — which is the
harness behaving correctly, not the committed code failing.

### S.5 C24-4-R2 — real PhoBERT integration **PASS**

`vinai/phobert-base` @ `01daacda68afe13d83023d16ec647239e344a1e6`, loaded from
the repo-local HF cache. Schema `preg1-head-v1`. Real `PhobertTokenizer`, real
`RobertaModel`, hidden size **768**.

**RAW_BASE held on real text:** Vanilla and Base-only strings differ, no
segmentation. Real tokenization gave both pathways shape **(3, 256)**, token
position 0 = BOS/`<s>`, and **different token-id grids**.

The **committed** validator selected the dtype convention `torch.float32` — the
thing C24-4-R1's harness had declined to guess.

Real bound representations, both pathways: shape **(3, 768)**, dtype
`torch.float32`, `requires_grad` **False**.

**FIRST_TOKEN verified against the real model.** For **both** pathways, the
values returned by the committed `extract_bound_representations` path are
**exactly equal** to `real_encoder(...).last_hidden_state[:, 0, :]`. This is the
claim a fake embedding module could never settle — §M limitation 2 said so, and
it is now discharged.

Encoder invariants after extraction: **eval**, every parameter
`requires_grad=False`, every parameter `grad` **None**.

Real cache: **save PASS**, **load PASS**, exact key preserved, tensor preserved,
and a **Vanilla cache loaded under a Base-only key is refused** with
`EvaluationContractViolation` — the single most dangerous reuse, refused against
real 768-dimensional tensors.

No head, no optimizer, no training, no LR tuning, no downstream score, no real
UIT-VSFC in this smoke, official TEST untouched. **This closes the real-model
integration gate.**

### S.6 C24-5A — real-data interface discovery

**PASS**, discovery only. Confirmed derived TRAIN SHA
`a20c0f7760f32dc48263a79d73ddf5363526c17e9de2afc32d8346b23444d301`; the excluded
conflict `train:11293` / `train:11417`; and the committed
`load_derived_pool(path, text_column, label_column, id_column, *,
expected_sha256=a20c0f77…, expected_rows=11424, expected_label_counts=…)` and
`load_membership(...)`. No raw dataset present in the runtime before C24-5B. No
model, training or score.

### S.7 C24-5B attempt 1 — a one-character harness typo

**Not a dataset mismatch.** The external harness omitted the final character `3`
from the expected SHA of `dev/sentiments.txt`. The authoritative,
repository-recorded value is
`a9584a22c926a54c6042236380c9a65ab8c41467477f7a5d794fb2505c96a9c3`, and **the
downloaded official file matched it**. The typo was fixed in the external Colab
harness only; **no repository code changed**. No training, no score.

Worth stating plainly: the digest gate did exactly its job. A wrong expectation
produced a refusal rather than a silent acceptance.

### S.8 C24-5B-R1 — real approved-data / role boundary **PASS**

All locked SHAs passed a 64-hex-character preflight before use.

Official public UIT-VSFC source folder `1xclbjHHK58zk2X6iqbvMPS2rcy9y9E0X`.
**Only these four content files were selected and downloaded:**

| File | SHA-256 |
|---|---|
| `train/sents.txt` | `5481dc1fa51f2fe72f22afd89b8aeb7f8945a126af7e66ac622e2ab0291130cb` |
| `train/sentiments.txt` | `480480c3b9a6bc8bdf53339c721e93f8cc30472ac8f0bd21d4440ce1171aefac` |
| `dev/sents.txt` | `fb7c3cc3173e1383edc03779883d91bb4d6110c8dd881612572a256878aa23b4` |
| `dev/sentiments.txt` | `a9584a22c926a54c6042236380c9a65ab8c41467477f7a5d794fb2505c96a9c3` |

**Official TEST content was not selected, not downloaded, not read and not
scored.**

Raw counts — TRAIN **11 426** (`0` 5 325, `1` 458, `2` 5 643); official DEV
**1 583** (`0` 705, `1` 73, `2` 805).

Adapter reconstruction used the official `sents.txt` + `sentiments.txt` with **no
NFC normalisation, no whitespace normalisation, no label transformation**,
`csv.DictWriter`, fields `id,text,label`, `newline=""`, LF, zero-based ids
`split:NNNNN`. The entire conflicting canonical group (`train:11293`,
`train:11417`) was excluded, **with no relabel**.

| Derived | Rows | Bytes | SHA-256 |
|---|---|---|---|
| TRAIN | **11 424** | 1 067 331 | `a20c0f7760f32dc48263a79d73ddf5363526c17e9de2afc32d8346b23444d301` |
| official DEV | **1 583** | 139 001 | `9c475c8998871c0c7317ee200b3e7db827128cd2dfec9de5c689aca299acc8d0` |

Derived TRAIN labels: `0` 5 324, `1` 458, `2` 5 642. **The committed
`load_derived_pool()` accepted it and its locked SHA/count gate passed** — the
Audit-022 digest reproduced from the official source a third time.

Real Audit-023 membership loaded: protocol-train **9 139**, protocol-dev
**2 285**, union **11 424**, assignment digest
`7bd5d1892e23b96035c376936d2168f547661da07b8abc769f5656e9438f4f84`. Unique
within each split, disjoint, union exactly equal to the real derived TRAIN ids,
**every membership id resolves to exactly one real row**, none missing, none
duplicated.

**Role-boundary dry-run** — dummy tensors only, purely to exercise provenance:

| Case | Result |
|---|---|
| real protocol-train membership → `PROTOCOL_TRAIN` | **PASS** |
| real protocol-dev membership → `PROTOCOL_DEV` | **PASS** |
| official validation → `OFFICIAL_VALIDATION` | **PASS** |
| official validation supplied where protocol-dev is required | **REFUSED — `SplitLeakage`** |
| protocol-dev supplied where `OFFICIAL_VALIDATION` is required | **REFUSED — `SplitLeakage`** |
| official TEST role | **absent — PASS** |

**Official validation is therefore measurement-only on the supported path, and
protocol-dev cannot be reported as the measurement** — both directions refused
against *real* membership rather than synthetic ids.

Execution boundary: real TRAIN/DEV integrity read **yes**; real model **no**;
representation extraction **no**; head **no**; optimizer **no**; training **no**;
LR tuning **no**; downstream score **no**; official TEST content **untouched**.
Raw and derived files existed only inside a `TemporaryDirectory` and were not
persisted; the repository stayed clean. **This closes the real-data boundary
gate.**

### S.9 What "scientifically closed" means here — and what it does not

Audit 024's **implementation, runtime, integration and data-boundary
verification is complete**. That is the whole of the claim.

**It does not mean** the Vanilla-vs-Base-only burden result exists, that pre-G1
is finished, that Stage-1 is approved for training, that Stage-2 pooling is
locked, or that D-B3B0-002 is resolved.

| Still open | State |
|---|---|
| D-B3B0-002 | **OPEN** |
| Final Stage-2 pooling | **OPEN** |
| Compiled proposal PDF | **STALE** |
| Official TEST | **SEALED** |
| Real downstream head training | **NOT RUN** |
| LR sweep | **NOT RUN** |
| Downstream score | **NONE** |
| Stage-1 training / HPO | **NOT RUN** |

**No new scientific decision was created by this finalization.** Runtime
verification of a precommitted implementation is not a decision.


---

```
AUDIT 024 CREATED:
YES

VERDICT:
IMPLEMENTATION PASS — ALL FOUR VERIFICATION GATES CLOSED;
NO DIAGNOSTIC RUN, NO DOWNSTREAM SCORE

SCIENTIFICALLY CLOSED AS AN IMPLEMENTATION / PRE-DIAGNOSTIC AUDIT:
YES — implementation, runtime, integration and data-boundary verification complete
NOT A CLAIM THAT: the burden result exists, pre-G1 is finished,
Stage-1 is approved for training, Stage-2 pooling is locked,
or D-B3B0-002 is resolved

REVISION:
4 (FINALIZATION)

FINAL VERIFICATION COMMIT:
767fb8ee30fa9b5df344a3629b3295c1fd03c379

AUDIT-024 TARGETED TORCH CONTRACT TESTS:
PASS / CLOSED — 136 passed / 0 failed / 0 skipped at HEAD 52ebca06
FILE UNMODIFIED SINCE THAT COMMIT

REPOSITORY-WIDE TORCH REGRESSION:
PASS / CLOSED — C24-1-R3C AT 767fb8ee: 2401 passed / 0 failed / 0 errors
THE R3C `results/` STOP WAS A STALE HISTORICAL PROBE-HYGIENE GUARD,
NOT A TEST FAILURE (gitignored; mtime 2026-08-21 10:41:51 UTC)
HISTORY: R2B 2383/8; R3B eight repairs 8/8 PASS, broader group 8/328,
full suite NOT REACHED — root cause a test-fixture regression
(stub vocab 64 vs ids to 4099), repaired via one shared STUB_VOCAB_SIZE

REAL PHOBERT INTEGRATION:
PASS / CLOSED — C24-4-R2
vinai/phobert-base @ 01daacda68afe13d83023d16ec647239e344a1e6
PhobertTokenizer / RobertaModel / hidden 768 / (3,256) / FP32 (3,768)
FIRST_TOKEN EXACTLY EQUALS last_hidden_state[:, 0, :] FOR BOTH PATHWAYS
REAL CACHE SAVE+LOAD PASS; VANILLA-AS-BASE-ONLY REFUSED
(C24-4-R1 WAS AN EXTERNAL SMOKE-HARNESS METADATA-DISCOVERY FAILURE)

REAL APPROVED-DATA / ROLE BOUNDARY:
PASS / CLOSED — C24-5B-R1
DERIVED TRAIN a20c0f77... 11424 rows / 1067331 bytes / 5324-458-5642
DERIVED DEV 9c475c89... 1583 rows / 139001 bytes
MEMBERSHIP 9139 / 2285 / 11424; DIGEST 7bd5d189...
EVERY MEMBERSHIP ID RESOLVES TO EXACTLY ONE REAL ROW
OFFICIAL VALIDATION -> SELECTION: REFUSED (SplitLeakage)
PROTOCOL-DEV -> MEASUREMENT: REFUSED (SplitLeakage)
OFFICIAL TEST CONTENT: NOT SELECTED / NOT DOWNLOADED / NOT READ / NOT SCORED
(C24-5B ATTEMPT 1 WAS A ONE-CHARACTER EXTERNAL-HARNESS SHA TYPO)

PINNED INVENTORY:
RECOVERED VIA REPOSITORY FETCHER — RESOURCE REQUIREMENT, NOT A DEFECT
NOT COMMITTED

PRODUCTION CHANGE:
ONE — ADDITIVE ChannelContractViolation GRID GUARD (+26/-0)
IMPLEMENTATION HARDENING; VERIFIED-POSITION RULE NOT WEAKENED

VERIFIED_POSITION_PROFILES:
ONE ENTRY — UNCHANGED

FIRST COLAB TORCH RUN (C24-1-R1):
FAILED — 132 passed / 2 failed / 0 skipped at HEAD b43cca82
ALL 134 THEN-CURRENT TESTS EXECUTED; 26 OF THE 28 TORCH-GATED PASSED
BOTH FAILURES TEST-SIDE; ZERO IMPLEMENTATION LINES CHANGED

SCIENTIFIC DOWNSTREAM DIAGNOSTIC EXECUTION:
NONE — NO REAL-DATA HEAD TRAINING / LR TUNING / VANILLA-vs-BASE SCORE

VERIFICATION-ONLY EXECUTION:
REAL PHOBERT SMOKE = PASS (C24-4-R2)
REAL APPROVED-DATA / ROLE BOUNDARY = PASS (C24-5B-R1)
NEITHER PRODUCED A DOWNSTREAM RESULT

MODEL / DATASET DOWNLOAD:
NONE BY THE LOCAL IMPLEMENTATION/AUDIT WORK (.venv ML-FREE)
ACCIDENTAL COLAB PROBE: DOWNLOAD-vs-CACHE STATUS **NOT ESTABLISHED**

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
TORCH-PASS IN C24-1-R2

ENCODER FREEZE / EVAL / NO_GRAD / FP32:
TORCH-PASS IN C24-1-R2

REPRESENTATION CACHE:
PROVENANCE LOGIC LOCAL-PASS + TORCH SAVE/LOAD PASS IN C24-1-R2

EPOCH / TRAINING-LOOP CONTRACT:
30 / NO EARLY STOPPING — CONTRACT LOCAL-PASS + TORCH EXECUTION PASS IN C24-1-R2

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
full suite 2312 passed, 89 skipped
targeted file: 136 authored / 108 passed locally / 28 torch-gated
OF THE 8 REPAIRED TESTS: 1 RUNS AND PASSES LOCALLY, 7 REMAIN TORCH-GATED

TORCH RUNTIME VERIFICATION:
TARGETED AUDIT-024 = PASS (136/0/0)
REPOSITORY-WIDE = PASS (2401/0/0, C24-1-R3C)

ACCIDENTAL HISTORICAL B3B1 PROBE:
DOWNLOAD-vs-CACHE STATUS STILL NOT ESTABLISHED

NEXT SCIENTIFIC STEP (NOT RUN):
PRE-G1 VANILLA-VS-BASE DIAGNOSTIC — 15 TUNING + 10 PAIRED = 25 LINEAR-HEAD RUNS
ENCODER REMAINS FROZEN

COMMIT CREATED:
NO
```
