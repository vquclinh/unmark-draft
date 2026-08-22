# Audit 029 — Stage-1 runner implementation

| | |
|---|---|
| **Audit id** | 029 |
| **Created (UTC)** | 2026-08-22 |
| **Baseline HEAD** | `5b07430` (`docs: lock Stage-1 scientific configuration`) |
| **Scope** | Implement the complete pre-training Stage-1 execution stack from the locked Audit 028 configuration. **Execute none of it.** |
| **Predecessor** | [028](028-stage1-scientific-config-review.md) Revision 2 — the authoritative config lock |
| **Type** | Implementation + tests. **No real Stage-1 run, no corpus download, no model load, no optimizer step on real data** |
| **NOT** | **This is not the PRE-TRAIN audit.** That happens after this is reviewed, committed, the proposal/PDF are synchronised, and a no-update real-model smoke is available for review |

---

## A. VERDICT

**IMPLEMENTATION PASS — STAGE-1 STACK COMPLETE; NOT EXECUTED**

**2 513 local tests pass, 97 skip** (2 398 / 91 before — **+115**, of which 6 are
torch-gated and run on Colab).

The blocking defect Audit 028 found is **fixed**: `scope_for` exists, the
run-global scope is gone, and STRIP-ALL now has training support that is
measured rather than assumed. The corpus pin is **closed** with all three
parquet digests. Nothing was downloaded, no model was loaded, and no optimizer
step ran on real data.

**Two real defects were found and fixed during implementation**, both by tests
that were written to fail first (§I.2). Neither was a specification problem.

**Compiled proposal PDF: STALE.** The editable source `unmark-proposal.md` was
updated (new §5.1.1, changelog v1.5); the PDF is **not** regenerated here.

---

## B. FILES CHANGED

**New — implementation (11)**

| File | Purpose |
|---|---|
| `configs/data/uvw_2026.json` | The committed corpus pin: revision, three filenames, byte sizes, sha256, concatenation order |
| `unmark/stage1/protocol.py` | The locked Stage-1 protocol. One source of truth, torch-free |
| `unmark/stage1/corpus.py` | Pin verification, schema, concatenation, contamination screen, document split |
| `unmark/stage1/chunking.py` | Deterministic tokenizer-aware pre-chunker |
| `unmark/stage1/manifest.py` | Prepared-corpus manifest + fail-closed compatibility check |
| `unmark/stage1/sampler.py` | Deterministic training order and exact mid-pass resume |
| `unmark/stage1/selection.py` | Score, tie-breaks, budget rule, run schedule |
| `unmark/stage1/optim.py` | Weight-decay groups (torch-free) + AdamW construction |
| `unmark/stage1/validation.py` | Held-out unlabeled evaluator on the fixed condition grid |
| `unmark/stage1/trainer.py` | The training loop, model contract, monitoring, checkpointing |
| `unmark/stage1/execute.py` | Stage orchestration + the step-incapable `smoke_check` |
| `scripts/stage1_runner.py` | CLI: `prepare-corpus`, `lr-pilot`, `r-phase1`, `final-main`, `smoke` |

**Modified (5)**

| File | Change |
|---|---|
| `unmark/stage1/contracts.py` | `CorruptionRatePolicy` gains `scope_for` + `pi_strip`; OPEN/LOCKED registers updated; SCIENTIFIC config requires the locked mixture |
| `unmark/stage1/data.py` | `prepare_example` draws **both** streams; extracted `prepare_with_condition` so validation reuses one implementation; `letter_channels_differ` added |
| `configs/corruption/default.yaml` | Stale `UNRESOLVED`/`provisional` eligibility block corrected to the resolved policy |
| `docs/spec/decisions.md` | D-S1B-002 pin CLOSED; D-S1B-003 IMPLEMENTED; new D-S1B-005 … D-S1B-008 |
| `unmark-proposal.md` | New §5.1.1 Stage-1 protocol; §5 open-items row closed; changelog v1.5 |

**New — tests (6):** `test_stage1_corruption_scope.py`, `test_stage1_corpus.py`,
`test_stage1_chunking.py`, `test_stage1_schedule.py`,
`test_stage1_runner_contract.py`, `test_stage1_torch_contracts.py`.
`tests/test_stage1.py` had two tests **replaced** (§I.1).

---

## C. THE CORPUS PIN — CLOSED

```
dataset  : undertheseanlp/UVW-2026
revision : a0a79294e4568137e25828bb3f2a4cde8546e1fb
```

| # | File (concatenation order) | Bytes | sha256 |
|---|---|---|---|
| 1 | `train.parquet` | 608 316 204 | `524374d4…a8c955f2` |
| 2 | `validation.parquet` | 78 047 554 | `d3da5989…9936a83f` |
| 3 | `test.parquet` | 79 550 587 | `60fcbe70…d7c4e323` |

Committed to `configs/data/uvw_2026.json`, following the established
`vncorenlp_v1.2.json` pin convention. **Parquet bytes are not in git.**

`verify_corpus_root` checks **filename, exact byte size and exact sha256 for all
three files, before a single row is read** — verifying after reading would mean a
wrong revision had already influenced the run. `CorpusPin` refuses any revision
that is not a full 40-character sha, so a moving `main` cannot be a pin.

**All three shards are used.** `shard_labels_are_a_split: false` is recorded in
the manifest, the protocol and the pin. The shard named `test.parquet` is an
upstream unlabeled Wikipedia source shard with **no relation** to UIT-VSFC
official TEST.

---

## D. SEEDS — SEVEN ROLES, ALL VERIFIED

Derived with the repository's `derive_seeds(tag, count)` (`sha256(tag)` read as
2-byte big-endian ints) and **verified by the helper, not copied**. The same
call reproduces the committed `TUNING_SEEDS = (5509, 19422, 11800)`.

| Role | Namespace tag | Seed |
|---|---|---|
| Selection (pilot + Phase 1) | `UNMARK-STAGE1-v1\|selection` | **21230** |
| Final main run 0 | `UNMARK-STAGE1-v1\|train\|0` | **36930** |
| Final main run 1 | `UNMARK-STAGE1-v1\|train\|1` | **7309** |
| Final main run 2 | `UNMARK-STAGE1-v1\|train\|2` | **5993** |
| Training corruption | `UNMARK-STAGE1-v1\|corruption` | **35422** |
| **Document split** | `UNMARK-STAGE1-v1\|split` | **51733** |
| **Validation corruption** | `UNMARK-STAGE1-v1\|validation-corruption` | **19225** |

All seven **distinct**, asserted at import time in `protocol.py` — a future
collision breaks the build rather than quietly coupling two roles. The two new
roles are recorded as **D-S1B-005**; Audit 028 was not reopened for prose.

---

## E. `scope_for` — THE STRIP-ALL FIX

**The defect (Audit 028 §F):** a run-global `"TONE"` scope left the corrupted
branch's letter channel **bit-identical** to the clean branch's in **0 / 18**
prepared examples, so the headline evaluation condition had zero training
support.

**The implementation.** `CorruptionRatePolicy` now draws two **domain-separated**
digests from one shared helper:

```python
def _unit_draw(self, namespace, sample_id, visit):
    payload = "|".join((namespace, self.schema_version, str(self.seed),
                        str(sample_id), str(visit)))
    return int.from_bytes(blake2b(payload.encode(), digest_size=8).digest(), "big") / 2**64

def rate_for (self, sid, visit=0): return self._unit_draw(RATE_NAMESPACE,  sid, visit)
def scope_for(self, sid, visit=0):
    return "TONE_AND_LETTER" if self._unit_draw(SCOPE_NAMESPACE, sid, visit) < self.pi_strip else "TONE"
```

`pi_strip = 0.25`, locked. `prepare_example` calls **both**. The audited
corruption engine's removal semantics are **unchanged** — the policy only chooses
which existing scope to invoke, and no per-syllable letter-dropout `q` was added.

**There is no run-global scope any more.** Asking a mixture policy for `.scope`
**raises** rather than answering, because returning one would be a lie a caller
could act on. A pinned scope is diagnostic-only, and `Stage1RunConfig` refuses a
SCIENTIFIC purpose unless the policy is the locked mixture.

**Evidence** (14 ML-free tests):

| Property | Result |
|---|---|
| Deterministic and repeatable | pass |
| `visit` changes both streams | pass |
| `sample_id` / seed change the stream | pass |
| `scope_for` never calls `rate_for`, and vice versa | AST-asserted |
| Each draw uses its **own** namespace | AST-asserted on the `_unit_draw` call |
| `P(TONE_AND_LETTER)` over 6 000 ids | **0.2523** vs locked 0.25 |
| `p \| scope` covers all ten deciles, **both** scopes | pass |
| conditional mean of `p`, both scopes | ≈ 0.50 |
| **Letter-channel degradation now occurs** | pass — was **0/N** |
| **Tone-only degradation survives** | pass |
| Base invariance holds under both scopes | pass |

**No statistical independence proof is claimed from a finite sample.** What is
tested is the *construction* — separate namespaces, no shared scalar, no
functional dependence either way — plus deterministic finite-sample sanity.

---

## F. CORPUS PIPELINE

```
verify pin  ->  read + concatenate  ->  schema  ->  contamination screen
            ->  DOCUMENT-LEVEL SPLIT  ->  CHUNK  ->  chunks inherit partition
```

**Schema.** `id` and `content` required; null/empty ids or text rejected with the
offending row; **duplicate document ids FAIL and are reported** — never renamed
or de-duplicated, because document identity keys the corruption stream.
Concatenation follows the locked order regardless of dict ordering.

**Contamination.** Exact/canonical only: `sha256(canon(x))` equality. Accepts
**only** `uitvsfc_derived_train` and `uitvsfc_official_validation`; any other key
— including anything naming TEST — **raises**. Near-duplicates are deliberately
**not** excluded (a fuzzy match needs a threshold, and thresholds are choices).
The report carries digests and ids, **never UIT-VSFC text**, and its `claim`
field states explicitly that this is *not* a claim of zero overlap with sealed
TEST.

**Split.** `stratified_group_split` was **not** reused: it is fraction-based and
label-stratified, and this corpus is unlabeled and needs an exact count of 5 000.
Reusing it would have meant inventing a label and a fraction that only
approximates the locked count. Instead: stable `blake2b` hash-ranking over
`tag|seed|document_id` with the id as tie-break — no global RNG, order-independent
by construction (verified against a shuffled input).

**Split before chunk — structural.** `chunk_document` *takes* a partition and
copies it onto every chunk; there is no code path that assigns one. `chunk_corpus`
refuses any document without a partition. `verify_no_parent_spans_partitions`
asserts the invariant anyway and is itself tested against a hand-built violating
chunk set.

---

## G. CHUNKING CONTRACT

| # | Requirement | How |
|---|---|---|
| 1 | Preserve text order | contiguous segments, emitted in order; `" ".join(chunks) == content` asserted |
| 2 | No extra normalization | AST-asserted: the chunker calls no `canon`, `decompose`, `corrupt`, `normalize`, `lower` |
| 3 | Stable ids | `{document_id}#{chunk_index}`, re-derived and asserted in `PreparedChunk.__post_init__` |
| 4 | Fits **both** paths | reference and base length functions both checked; the test's mock base path is deliberately *longer* |
| 5 | Never split a syllable | cuts land only on whitespace boundaries |
| 6 | Runs after the split | partition is an argument |
| 7 | Inherits parent partition | copied, never assigned |

**No truncation.** An indivisible span raises `ChunkingViolation` carrying
document id, shard, source row, the segment, and both measured lengths.

The length functions are **injected**, so the whole contract is proved with a
lightweight mock — the real pinned tokenizer is not needed and was not
downloaded.

---

## H. TRAINER, VALIDATION, ORCHESTRATION

**Model contract** (`verify_model_contract`): no encoder parameter requires
grad, the encoder stays in eval, and the adapter has exactly **3 551 232**
trainable parameters at `d = 768`. `build_optimizer` **refuses** any parameter
that does not require grad — silently filtering the encoder out would hide a
wiring error instead of surfacing it.

**Weight decay:** `0.01` on `fusion.weight` and `gate.weight` only; `0.0` on
biases, LayerNorm and **both embedding tables**.

**Objective:** unchanged. Cosine, pooled, **attention-masked mean over
non-special content**. Asserted that `FIRST_TOKEN` appears in no Stage-1 module —
it was pre-G1-only.

**Precision: FP32** (D-S1B-007). AST-asserted that `autocast`, `GradScaler`,
`half` and `bfloat16` are called nowhere.

**Validation.** Fixed grid `FULL, P50, P100, STRIP_ALL`, taken from the audited
B2 condition set rather than re-derived. Keyed on the **validation** corruption
seed `19225`, at a fixed visit — AST-asserted that `prepare_condition_batch`
mentions no run seed, so a training seed cannot change a validation corruption.
The held-out realization is built **once** and reused by every candidate. Score =
`max` over the grid; checkpoint tie-break `d_clean` then **earliest** update;
`r` tie-break `d_clean` then **smaller r**. **Update 0 is required** — `select_checkpoint`
raises without it, so the initial clean-path distance is measured, not assumed.

**Monitoring.** `MonitorWindow` accumulates channel-degradation and
embedding-gradient evidence over a **window**, not per batch: with `pi_strip = 0.25`
a single batch may legitimately contain no letter-degraded syllable, and failing
on that would be a false alarm. Non-finite loss or gradient is a hard stop.
Encoder gradients raise. The runner **never** changes clipping mid-run.

**Budget** (`budget_decision`): inside → stop; exactly at 20 000 → continue the
**same** run to 40 000 preserving adapter, optimizer, `visit`, cursor and
streams; exactly at 40 000 → **STOP, `BUDGET_LIMITED`**. A third cap raises.
Selection then considers the whole 0→40 000 trajectory.

**Resume.** Because `pass_order` is a pure function of `(chunk_ids, seed, visit)`,
the payload is a **cursor**, not an opaque RNG blob. Tests assert: order is
deterministic and seed/visit dependent; each chunk is consumed once per pass;
`visit` advances only at the boundary; a straddling batch carries both visits;
mid-pass resume continues from the exact next position; resume does not re-serve
consumed chunks or bump `visit`; and **uninterrupted vs. resumed streams are
identical**. Resume fails closed on a different chunk set or schema.

**CLI.** `prepare-corpus`, `lr-pilot`, `r-phase1`, `final-main`, `smoke`.
Verified by AST: **no flag contains "test"**; none of `--lr`, `--r`,
`--batch-size`, `--epochs`, `--max-updates`, `--pi-strip`, `--scope`, `--seed`,
`--precision`, `--amp` … exists; the only UIT-VSFC flags are the two opened
sources, reachable **only** from `prepare-corpus` (resolved through the
subparser variable, not a substring scan). `smoke_check` calls no `backward`,
`step`, `zero_grad`, `build_optimizer` or `AdamW` — **structurally incapable of
an update**.

---

## I. DEFECTS FOUND DURING IMPLEMENTATION

### I.1 Two tests encoded the old contract — replaced, not loosened

`test_corruption_redraw_schedule_is_open` and
`test_letter_dropout_scope_is_not_enabled_by_default` asserted that the redraw
schedule and letter dropout were OPEN, and that the default scope was `"TONE"` —
i.e. they asserted **the defect**. Both were **replaced** with tests of the new
contract (locked registers; the default is the mixture; `.scope` raises; a
pinned scope cannot go SCIENTIFIC), not weakened to pass.

### I.2 A real chunker defect, caught by a test written to fail first

The first chunker checked whether a segment fitted when *appended* to the current
chunk, but not when it *started* a new one. An oversized segment would therefore
have become its own **oversized chunk** — silently violating `max_length` instead
of failing. `test_no_truncation_ever_happens` caught it. Fixed with
`require_fits_alone`, and **mutation-verified**: removing the new check makes the
test fail again.

### I.3 My own verification was briefly prose-matching

Checking "smoke cannot step" with a substring scan matched the **docstring**
saying it calls no `.backward()`. Replaced with an AST check over the call
graph — the same defect class this project has hit repeatedly.

---

## J. TESTS

| File | Result |
|---|---|
| `test_stage1.py` | 65 passed, 14 skipped |
| `test_stage1_corruption_scope.py` | **14 passed** |
| `test_stage1_corpus.py` | **24 passed** |
| `test_stage1_chunking.py` | **13 passed** |
| `test_stage1_schedule.py` | **25 passed** |
| `test_stage1_runner_contract.py` | **39 passed** |
| `test_stage1_torch_contracts.py` | **6 skipped** (torch-gated; run on Colab) |
| **Full repository** | **2 513 passed, 97 skipped** (was 2 398 / 91) |

Static checks: every Stage-1 module AST-verified to import no torch at module
scope; CLI imports without torch; `configs/corruption/default.yaml` now agrees
with `active_eligibility_policy()`; corpus pin manifest structurally validated.

---

## K. BOUNDARIES

| Boundary | Status |
|---|---|
| UIT-VSFC official TEST | **SEALED** — no flag, no path, no API, screen refuses any such key |
| Downstream scores in Stage-1 selection | **absent** — held-out unlabeled only |
| Raw corpus text in scientific reports | **absent** — manifests/artifacts carry ids, digests, counts. The prepared-corpus data file contains text because it **is** the dataset |
| UIT-VSFC text in the contamination report | **absent** — digests only |
| Scientific CLI overrides | **none** |
| Optimizer step in `smoke` | **structurally impossible** |
| Encoder training | refused by `verify_model_contract` and `build_optimizer` |

---

## L. WHAT DID **NOT** RUN

**No real Stage-1 scientific execution occurred.**

- UVW-2026 **not** downloaded (766 MB); `prepare-corpus` **not** run
- PhoBERT **not** loaded; no real forward, no real backward
- LR pilot, `r` sweep, final main: **not run**
- No optimizer step on real data or a real model
- No downstream task; official TEST **not** touched
- Local `.venv` remains **ML-free** (no torch, transformers, datasets, numpy, sklearn)

Everything above is synthetic fixtures, AST/contract assertions, and toy-tensor
tests that skip locally.

---

## M. LIMITATIONS

1. **Nothing has executed.** Every claim is structural or synthetic. The real
   parquet schema has been *asserted* (`id`, `content`) but never *observed* —
   `read_shard` fails closed if the columns are absent, which is the correct
   posture, but the first `prepare-corpus` run is where that is confirmed.
2. **Chunk-size behaviour on real Wikipedia is unknown.** The chunker is proved
   against mock length functions; real article length distribution, chunk counts
   and any indivisible-span failures are unmeasured.
3. **The 6 torch-gated tests have never run.** They skip in the ML-free venv.
4. **`execute_stage` has never been exercised end to end.** Its orchestration is
   tested only through the units it calls.
5. **The budget continuation path is tested as bookkeeping**, not as a real
   20 k → 40 k training continuation.
6. **`d_clean` is computed inside the FULL-condition pass** for efficiency; that
   coupling is correct but is an implementation choice worth re-reading during
   the PRE-TRAIN audit.

---

## N. REMAINING BLOCKERS BEFORE A REAL RUN

1. **Review and commit this implementation.**
2. **Regenerate the compiled proposal PDF** — currently **STALE**.
3. **Run `prepare-corpus` once** on the pinned bytes and review its manifest:
   real schema, duplicate-id behaviour, contamination count, chunk counts, and
   that no article spans both partitions.
4. **Run `smoke`** (no-update real-model check) and review it.
5. **PRE-TRAIN audit** of the whole stack.
6. Only then: LR pilot → `r` Phase 1 → the three final main runs.

---

## O. SELF-AUDIT

| # | Check | Result |
|---|---|---|
| 1 | Audit 029 created and persisted | **yes** |
| 2 | No real Stage-1 run, no corpus download, no model load, no optimizer step | **yes** |
| 3 | Official UIT-VSFC TEST untouched and unreachable | **yes** |
| 4 | Corpus pin recorded with revision + 3 filenames + sizes + digests + order | **yes** — `configs/data/uvw_2026.json` |
| 5 | Parquet bytes not committed | **yes** |
| 6 | All three shards used; upstream labels not treated as a split | **yes** |
| 7 | Split seed 51733 and validation-corruption seed 19225 verified by the helper | **yes** — §D |
| 8 | All seven role seeds distinct, asserted at import | **yes** |
| 9 | `scope_for` implemented; `pi_strip = 0.25` locked | **yes** |
| 10 | Rate/scope domain-separated; neither derived from the other | **yes** — AST-asserted |
| 11 | Corruption engine removal semantics unchanged | **yes** |
| 12 | STRIP-ALL support exists; tone-only support survives | **yes** — measured |
| 13 | Stale corruption YAML corrected; semantics unchanged | **yes** |
| 14 | Corpus pipeline is a separate path from the optimizer loop | **yes** |
| 15 | Verify-before-read; all three files; name+size+digest | **yes** |
| 16 | Duplicate document ids fail and are reported | **yes** |
| 17 | Contamination exact/canonical only; no fuzzy; no TEST route | **yes** |
| 18 | Split before chunk; chunks inherit partition; no article spans both | **yes** — structural + tested |
| 19 | No truncation; indivisible span fails with provenance | **yes** |
| 20 | Manifest binds the protocol and fails closed | **yes** — 14 refusal cases |
| 21 | Validation grid + seed locked; training seed cannot alter it | **yes** |
| 22 | Update 0 recorded before any step | **yes** — enforced |
| 23 | Exactly 3 + 5 + 3 = 11 runs; no post-hoc grid expansion | **yes** |
| 24 | Final three are the final adapters; nothing follows | **yes** |
| 25 | Budget 20 k → one continuation → 40 k → BUDGET_LIMITED | **yes** |
| 26 | Resume exact: cursor, visit, optimizer, adapter; fails closed | **yes** |
| 27 | Encoder frozen/eval, excluded from optimizer; adapter 3 551 232 | **yes** |
| 28 | Masked-mean pooling; no FIRST_TOKEN anywhere | **yes** |
| 29 | FP32; no AMP introduced | **yes** — AST-asserted |
| 30 | No scientific CLI overrides; no TEST argument | **yes** |
| 31 | `smoke` structurally cannot step | **yes** |
| 32 | No raw text in scientific report artifacts | **yes** |
| 33 | Tests behavioural/AST, not prose-matching | **yes** — one lapse caught and fixed (§I.3) |
| 34 | Decisions recorded | **yes** — D-S1B-002/003 updated, 005–008 added |
| 35 | Editable proposal updated; **PDF STALE** | **yes** — §5.1.1, changelog v1.5 |
| 36 | `.venv` still ML-free | **yes** |
| 37 | Everything unstaged; no prohibited git operation | **yes** |

---

**STATUS: IMPLEMENTATION PASS — STAGE-1 STACK COMPLETE; NOT EXECUTED**
**STRIP-ALL SUPPORT: IMPLEMENTED AND MEASURED — NO LONGER BLOCKING**
**CORPUS PIN: CLOSED (all three shards, verified at load)**
**NO REAL STAGE-1 SCIENTIFIC EXECUTION OCCURRED**
**COMPILED PROPOSAL PDF: STALE**
**THIS IS NOT THE PRE-TRAIN AUDIT**
**ALL CHANGES UNSTAGED**
