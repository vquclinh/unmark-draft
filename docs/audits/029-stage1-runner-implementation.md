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
| **Revision 3a** | **2026-08-23** — **Revision 3's own micro-probe FAILED on the real pinned tokenizer** (`composed 5, exact 7`, rc 1). The runtime verifier worked and is kept. Per-run token composition removed; the 956x claim is withdrawn; 192x remains and is exact by construction. See §S. |
| **Revision 3** | **2026-08-23** — **third post-commit real-corpus defect**: stage-6 pre-chunking re-canonicalised every growing prefix (~250x document length) and produced no 1 % line in 13 minutes. Repaired by memoised, composable transforms and per-run token composition; `chunking.py` untouched. See §R. |
| **Revision 2** | **2026-08-23** — **second post-commit real-corpus defect**: the stage-4 contamination screen applied full `canon()` to all 1 118 224 documents and ran >7 h. Repaired with two proven necessary-condition prefilters; the criterion is unchanged. See §Q. |
| **Revision 1a** | **2026-08-22** — evidence-accuracy cleanup. Audit-only: §L/§M labelled historical, three overstated §P claims corrected, and the uncaptured original exception recorded. No code, test or decision changed. See §P.11. |
| **Revision 1** | **2026-08-22** — **post-commit real-corpus defect and repair**, revised in place. The chunker could not prepare the locked corpus: "never split a syllable" had been implemented as "cut only at whitespace", and real UVW-2026 contains oversized **non-whitespace** units. See §P. |

---

## A. VERDICT

**REVISION 1 (2026-08-22): REPAIR PASS — READY FOR REAL CORPUS RE-RUN**

The implementation below passed every local and torch-gated test, was committed
as `0a34083`, and then **failed against the real corpus**: the chunker treated a
maximal non-whitespace unit as orthographically atomic, and UVW-2026 contains
such units up to **1 707 tokens**. Repaired in §P. **PRE-TRAIN remains BLOCKED**
until real `prepare-corpus` succeeds in Colab and the real-PhoBERT no-update
smoke passes.

---

## A. VERDICT (as first written)

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
| 1 | Preserve text order | **STRENGTHENED in Revision 1** — was `" ".join(chunks) == content`, which assumes single-space separation. Now: chunks are contiguous half-open slices tiling `[0, len(content))`, `"".join(texts) == content` byte-exact. See §P.4 |
| 2 | No extra normalization | AST-asserted: the chunker calls no `canon`, `decompose`, `corrupt`, `normalize`, `lower` |
| 3 | Stable ids | `{document_id}#{chunk_index}`, re-derived and asserted in `PreparedChunk.__post_init__` |
| 4 | Fits **both** paths | reference and base length functions both checked; the test's mock base path is deliberately *longer* |
| 5 | Never split a syllable | **CORRECTED in Revision 1** — was "cuts land only on whitespace boundaries", which is a stronger rule than the science requires and could not prepare the real corpus. Now: cuts land on offsets that `decompose` reports as outside every `SyllableSpan`. See §P |
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

## L. WHAT DID **NOT** RUN *(pre-commit state — partly SUPERSEDED by §P)*

> **HISTORICAL.** This section records what was true when Audit 029 was first
> written, **before** commit `0a34083`. Two of its claims have since been
> overtaken by real-corpus evidence and are annotated inline. **§P is the
> current state.** The original text is preserved rather than deleted, because
> the audit's value depends on showing what was known when.

**No real Stage-1 scientific execution occurred.** *(Still true: no training, no
optimizer step, no downstream task, and official TEST untouched — then and now.)*

- UVW-2026 **not** downloaded (766 MB); `prepare-corpus` **not** run
  — **SUPERSEDED by §P:** the researcher subsequently ran `prepare-corpus`
  against the real pinned bytes (it exited 2) and ran a real-tokenizer
  preflight. The corpus was **never downloaded into this environment**, which
  remains true, and no *successful* preparation has occurred.
- PhoBERT **not** loaded; no real forward, no real backward
- LR pilot, `r` sweep, final main: **not run**
- No optimizer step on real data or a real model
- No downstream task; official TEST **not** touched
- Local `.venv` remains **ML-free** (no torch, transformers, datasets, numpy, sklearn)

Everything above is synthetic fixtures, AST/contract assertions, and toy-tensor
tests that skip locally. *(Still true of everything done **in this environment**;
§P adds real-corpus evidence gathered **outside** it.)*

---

## M. LIMITATIONS *(pre-commit — items 1-2 SUPERSEDED by §P)*

> **HISTORICAL.** Written before commit `0a34083`. Limitation 1 correctly
> predicted the defect §P then found; limitation 2 named exactly the unknown
> that broke the run. Both are now answered — see §P.1 and §P.9 for the current
> limitations.

1. **Nothing has executed.** Every claim is structural or synthetic. The real
   parquet schema has been *asserted* (`id`, `content`) but never *observed* —
   `read_shard` fails closed if the columns are absent, which is the correct
   posture, but the first `prepare-corpus` run is where that is confirmed.
   — **ANSWERED in §P.1:** the schema **is** now observed on the real bytes:
   1 118 224 documents, 0 duplicate ids, 0 null/empty ids or content, `id` and
   `content` present in all three shards.
2. **Chunk-size behaviour on real Wikipedia is unknown.** The chunker is proved
   against mock length functions; real article length distribution, chunk counts
   and any indivisible-span failures are unmeasured.
   — **PARTLY ANSWERED, and this is where the defect was:** §P.1 shows the real
   corpus contains maximal non-whitespace units up to 1 707 tokens, which the
   committed chunker could not handle. Real chunk **counts** and the length
   **distribution** remain unmeasured (§P.9.4).
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

## P. REVISION 1 — POST-COMMIT REAL-CORPUS DEFECT AND REPAIR

**Date:** 2026-08-22 **Baseline commit:** `0a34083af184a6d25db845b36fb4f0a6a18792be`

### P.1 What the real corpus showed

The committed runner passed **every local test** and **6/6 torch-gated tests in
Colab**. The pinned bytes were then inspected directly:

```
dataset  : undertheseanlp/UVW-2026
revision : a0a79294e4568137e25828bb3f2a4cde8546e1fb
train.parquet      608 316 204 B  sha256 524374d40fc7b25501c9d3c7420d9d6f41973d24476418523d3f0536a8c955f2
validation.parquet  78 047 554 B  sha256 d3da59890851b2e68698b4fd8e67aef5439dbad8132e1a264d98eaab9936a83f
test.parquet        79 550 587 B  sha256 60fcbe70fd7c110fae09b34ee9306960f8db3738c5a7ce679753060bd7c4e323
```

**The parts of the contract that held.** Every schema and identity assumption
Audit 029 listed as *asserted but never observed* (§M.1) is now **observed**:

| Property | Real value |
|---|---|
| Total documents | **1 118 224** |
| Unique ids | **1 118 224** |
| Duplicate ids | **0** |
| Null / empty ids | **0 / 0** |
| Null / empty content | **0 / 0** |
| `id` and `content` present in all three shards | **yes** |

So the pin, the schema check and the duplicate-id contract are all confirmed
against reality.

**The chunker is the identified blocking defect; schema and identity assumptions
were independently confirmed by the real-corpus preflight.** The original
30-minute `prepare-corpus` invocation returned exit status 2, but its inner
stderr — and therefore the actual `Stage1ContractViolation` it raised — **was not
captured**. This audit does not claim to have seen that exception, and does not
claim the chunker was the *only* thing that could have failed. It claims what the
evidence supports: the chunker contains a deterministic blocking defect for real
oversized non-whitespace units, and the other pipeline stages were independently
verified against the real bytes.

**What failed.** A real-tokenizer preflight at the locked revision
`01daacda68afe13d83023d16ec647239e344a1e6` found oversized maximal
**non-whitespace** units immediately — at least 20 before the diagnostic stopped
collecting:

| Document | Characters | Clean tokens (incl. specials) |
|---|---|---|
| `Iraq` | 501 | 298 |
| `Quần_đảo_Hoàng_Sa` | 263 | 260 |
| `Viên_Chiếu` | 876 | 873 |
| `Đội_tuyển_bóng_đá_quốc_gia_Afghanistan` | 2 647 | **1 707** |

**What this evidence does and does not establish.** The preflight *measured*
these spans with the locked tokenizer; it did **not** execute `chunk_document`
on each listed example. The chain is therefore:

1. maximal non-whitespace spans exist in the real pinned corpus (**measured**);
2. the locked PhoBERT tokenizer puts them above `max_length = 256` (**measured**);
3. inspection of the committed whitespace-only chunker shows that such a unit,
   once reached, **necessarily** fails its fit-alone contract (**contract
   inspection**);
4. the defect was then **reproduced** against the committed code on a
   constructed fixture of the same shape (§P.6).

Together these establish that **the committed implementation cannot robustly
prepare the locked corpus**. That is a statement about the code's contract, not
a transcript of every listed document failing.

### P.2 Root cause

D-S1B-002 requires that chunking "never split a syllable span". Audit 029
implemented that as **"cuts land only on whitespace boundaries"** — §G rule 5,
now corrected in place. Whitespace *is* safe, so the premise is true; but it is
**sufficient, not necessary**, and the implementation silently promoted it to the
definition of indivisibility. A maximal non-whitespace unit was therefore treated
as atomic.

The repository's own orthography contradicts that.
`unmark/orthography/decompose.py::_segment_syllables` splits the unit stream on
`unit.is_letter`, so a `SyllableSpan` is a **maximal alphabetic run**. Underscores,
hyphens and punctuation are *between* spans. Measured below on a fixture built from the observed identifier, and on
that identifier itself:

```
# CONSTRUCTED fixture: the observed identifier repeated x12 by this audit
decompose("Đội_tuyển_bóng_đá_quốc_gia_Afghanistan" x12) -> 84 SyllableSpans
# REAL observed identifier, measured on its own
decompose("Đội_tuyển_bóng_đá_quốc_gia_Afghanistan")     ->  7 SyllableSpans
```

**This measurement is on a CONSTRUCTED fixture, not on real corpus content.**
The *document identifier* `Đội_tuyển_bóng_đá_quốc_gia_Afghanistan` is real
evidence from the preflight; the string measured above is that identifier
**repeated twelve times by this audit** to reach 467 characters. The real
document is reported as 2 647 characters / 1 707 tokens, and **its content is not
available in this environment** — nothing was downloaded.

The real identifier **on its own** is 38 characters and `decompose` reports
**7** spans in it, with boundaries at the underscores:
`(0,3,'Đội') (4,9,'tuyển') (10,14,'bóng')`. That is the load-bearing fact, and it
is measured on real observed text: a single maximal non-whitespace unit already
contains multiple orthographic spans, so **legal interior boundaries exist**. The
×12 repetition only makes the unit large enough to exceed `max_length` locally so
the defect can be reproduced without the corpus.

**A conflated fact.** B3B alignment reconstructs PhoBERT BPE over maximal
non-whitespace units (D-B3B1B-001). That is about **tokenization alignment**, is
authoritative, and is **unchanged**. It never implied the *document chunker* must
regard those units as scientifically atomic. Recorded as **D-S1B-009**.

### P.3 The repaired contract

| | Rule |
|---|---|
| Fast path | unchanged greedy accumulation over whitespace-delimited segments |
| Fallback | only when a single non-whitespace unit cannot fit alone |
| Safe offset | a character-unit boundary **not strictly inside a `SyllableSpan`**, obtained by *querying* `decompose` — `safe_cut_offsets` introduces no second syllable parser and no new linguistic rule |
| Vietnamese candidate spans | **indivisible** |
| Combining sequences | never split — candidates are unit boundaries, so a base and its marks stay together |
| Non-canonical text | `decompose` returns *canonical* offsets; when `canon(text) != text` they do not address the original, so **no interior boundary is offered** and the chunker stays fail-closed rather than normalising the corpus |
| Truly atomic oversized region | **fail closed** with document id, shard, source row, range, char count and both measured lengths |
| Every emitted chunk | re-measured; must satisfy `max_length = 256` on **both** paths |
| Truncation / dropping / normalisation / synthetic whitespace | still forbidden |
| Runtime overflow | still `FAIL`, still a guard |

Correctness does not depend on token counts growing monotonically with text:
the greedy scan records the last *fitting* candidate, and every emitted chunk is
independently length-checked.

### P.4 Text preservation is now actually lossless

The old invariant `" ".join(chunks) == content` assumed single-space separation.
It would have masked collapsed runs of whitespace, tabs and newlines, and cannot
describe an internal non-whitespace cut at all.

`PreparedChunk` now carries **`source_start` / `source_end`**, and
`verify_tiles_source` asserts that chunks are contiguous half-open slices tiling
`[0, len(content))` — no gaps, no overlaps, in order — and that
`"".join(texts) == content` **byte-exact**. `PreparedChunk.__post_init__` also
refuses any chunk whose text length disagrees with its range, so a chunk cannot
be a rewrite of its source. Chunks now retain the whitespace that separates them
rather than being stripped.

### P.5 Files changed in Revision 1

| File | Change |
|---|---|
| `unmark/stage1/chunking.py` | Rewritten: `safe_cut_offsets`, two-tier chunking, range-based `PreparedChunk`, `verify_tiles_source` |
| `scripts/stage1_runner.py` | Persists `source_start`/`source_end`; `[n/6]` stage banners; chunking progress every 1 %; chunking failures name the stage and re-raise the original error unchanged |
| `tests/test_stage1_chunking.py` | Rewritten — 13 → **35** tests |
| `docs/spec/decisions.md` | **D-S1B-009** added; status table updated |
| `docs/audits/029-…md` | This section; §G rules 1 and 5 corrected at the point of the original error |

`unmark-proposal.md` **unchanged and no change required** — it states the
contract at the right abstraction ("split first, chunk second … no truncation")
and never asserted a whitespace-only rule. The over-strong claim existed only in
this audit's prose.

### P.6 Regression tests

The defect was **reproduced against the committed code first**: a 467-character
underscored title raised `ChunkingViolation`; after the repair it yields 2 chunks
with exact reconstruction.

`tests/test_stage1_chunking.py` — **35 passed**. Coverage against Task D's list:

| Required case | Test |
|---|---|
| Oversized non-whitespace span | `test_an_oversized_maximal_non_whitespace_span_is_subdivided` |
| Base path longer, forces earlier cut | `test_a_longer_base_path_forces_an_earlier_cut` |
| Internal punctuation boundaries | `test_internal_punctuation_offers_legal_boundaries` (8 separators) |
| Candidate span never bisected | `test_no_vietnamese_candidate_span_is_ever_bisected` (4 fixtures, checked against `decompose`) |
| Atomic Vietnamese span fails closed | `test_a_genuinely_atomic_vietnamese_span_still_fails_closed` |
| Multiple spaces / tabs / newlines / leading+trailing | `test_exact_reconstruction_and_range_tiling` (6 fixtures) |
| Exact reconstruction + range tiling | same, plus `test_the_tiling_verifier_actually_catches_a_gap` |
| Stable chunk ids | `test_stable_chunk_ids_across_repeated_calls` |
| Order independence | `test_document_order_does_not_change_any_document_s_chunks` |
| ≤256 on both injected lengths | `test_every_emitted_chunk_fits_both_paths` |
| No truncation / no dropping / no synthetic whitespace | `test_no_truncation_and_no_dropping`, `test_whitespace_is_never_collapsed_or_synthesised` |
| Split-before-chunk, partition inheritance | 4 tests, unchanged in substance |
| Long non-Vietnamese no-whitespace surface | `test_a_long_non_vietnamese_no_whitespace_surface_is_subdividable` |
| B3 alignment/channel tests still green | full suite below |

**Mutation-verified** (each injected violation caught):

| Injected violation | Caught by |
|---|---|
| Revert to whitespace-only cutting | `test_a_long_non_vietnamese_no_whitespace_surface_is_subdividable` |
| Allow cuts inside a `SyllableSpan` | `test_no_vietnamese_candidate_span_is_ever_bisected` |
| Strip chunk text (collapse whitespace) | `test_no_truncation_and_no_dropping` |

Two further mutations — removing the final per-chunk length assertion and
removing the internal `verify_tiles_source` call — were **not** caught, because
the greedy algorithm already satisfies both properties. They are
**defence-in-depth guards, not independently verified**, and this audit does not
claim otherwise. `verify_tiles_source` itself *is* directly tested.

No real document name is hard-coded in production logic; the fixtures live only
in tests.

### P.7 Failure visibility (Task E)

The real `prepare-corpus` ran ~30 minutes and exited 2, surfacing only
`CalledProcessError`. Two reasons: the six pipeline stages printed nothing until
each completed, and the chunking loop — the slow stage, one tokenization per
segment across 1.1 M documents — was silent.

Small, semantics-free changes: `[1/6]`…`[6/6]` stage banners; a chunking progress
line every ~1 % of documents with the running chunk count; and a chunking failure
that prints **which stage failed and the contract message** to stderr before
**re-raising the original error unchanged**. Nothing is swallowed, and no corpus
text is written to logs — the violation carries ids, shard, row, ranges and
lengths only.

### P.8 Test results

| Suite | Result |
|---|---|
| `tests/test_stage1_chunking.py` | **35 passed** (was 13) |
| Full repository | **2 535 passed, 97 skipped** (was 2 513 / 97) |
| Torch-gated | 6 skipped locally; unchanged, run on Colab |

`.venv` remains ML-free.

### P.9 Limitations of Revision 1

1. **The real corpus has NOT been re-prepared.** I do not have the 766 MB of
   pinned bytes and did not download them. Every claim here rests on synthetic
   fixtures built to the shape of the reported evidence.
2. **The repair is not proven closed by these tests.** Only a real
   `prepare-corpus` run can show whether *other* documents contain regions that
   are genuinely atomic and oversized — a single alphabetic run longer than 256
   tokens would still, correctly, fail closed.
3. **Whether any real document is non-canonical is unmeasured.** For such a
   document `safe_cut_offsets` returns nothing and an oversized unit fails
   closed. That is deliberate — normalising the corpus is forbidden — but the
   real frequency is unknown.
4. **Chunk-count and runtime impact on the full corpus are unknown.** The
   fallback adds a `decompose` call per oversized unit only, but the overall
   ~30-minute cost is unaddressed; this task did not optimise it.
5. The two defence-in-depth guards noted in §P.6 are not independently verified.
6. **The original failing exception was never captured.** The first real
   `prepare-corpus` invocation ran ~30 minutes and returned exit status 2, but
   the notebook surfaced only `CalledProcessError` — the inner stderr and the
   `Stage1ContractViolation` it carried were lost. The blocking chunker defect is
   therefore established by **three converging lines of evidence**, not by a
   traceback from that run:

   | # | Evidence | Kind |
   |---|---|---|
   | 1 | Real-corpus tokenizer preflight: oversized maximal non-whitespace units exist and exceed 256 at the locked tokenizer revision | measured on real bytes |
   | 2 | Contract inspection of the committed chunker: such a unit, once reached, necessarily fails its fit-alone check | code inspection |
   | 3 | Reproduction against the committed code on a constructed fixture of the same shape (§P.6) | executed locally |

   This is sufficient to justify the repair, and it is **not** the same as having
   observed the original failure directly. A residual possibility cannot be
   excluded from the evidence available: that the ~30-minute run also failed, or
   first failed, somewhere this audit has not identified. The stage banners and
   progress output added in §P.7 exist precisely so the next run does not lose
   that information.

### P.10 Self-audit for Revision 1

| # | Check | Result |
|---|---|---|
| 1 | Audit 029 revised **in place**; no Audit 030 | **yes** |
| 2 | Defect reproduced against committed code before repairing | **yes** — §P.6 |
| 3 | Root cause identified from repository code, not guessed | **yes** — `_segment_syllables` splits on `unit.is_letter` |
| 4 | Existing orthographic API reused; no second syllable parser | **yes** — `safe_cut_offsets` only queries `decompose` |
| 5 | No cut inside a Vietnamese candidate span | **yes** — verified against `decompose` on 4 fixtures |
| 6 | No cut inside a combining sequence | **yes** — candidates are unit boundaries |
| 7 | No blind splitting at bytes, code points, BPE pieces or fixed counts | **yes** |
| 8 | `max_length = 256` unchanged | **yes** |
| 9 | Runtime overflow still `FAIL` | **yes** |
| 10 | No truncation, no document dropped, no text lost | **yes** — exact tiling asserted |
| 11 | No synthetic whitespace; none collapsed | **yes** — byte-exact reconstruction |
| 12 | No corpus normalization / repair / rewrite | **yes** — AST-asserted the chunker calls no `canon`/`replace`/`sub`/… |
| 13 | Preservation invariant strengthened to exact tiling | **yes** — §P.4 |
| 14 | Split-before-chunk and partition inheritance intact | **yes** |
| 15 | Stable chunk ids; order-independent | **yes** |
| 16 | Both paths measured on every **emitted** chunk | **yes** |
| 17 | Fail-closed retained for genuinely atomic regions | **yes**, with full provenance |
| 18 | No scientific hyperparameter changed | **yes** — no change to seeds, grids, `pi_strip`, objective, optimizer, batch size, corpus pin, dev count, TEST policy |
| 19 | UIT-VSFC TEST untouched | **yes** |
| 20 | No raw corpus text in logs or reports | **yes** |
| 21 | Original contract error not swallowed | **yes** — re-raised unchanged |
| 22 | Decisions log updated | **yes** — D-S1B-009 |
| 23 | Proposal: change needed? | **no** — it never stated the whitespace-only rule; recorded explicitly |
| 24 | Compiled PDF | **STALE**, not regenerated |
| 25 | Focused + full suites run | **yes** — 35 / 2 535 passed |
| 26 | Real corpus re-prepared? | **NO** — bytes not available here |
| 27 | Nothing staged; no prohibited git operation | **yes** |
| 28 | Defect claimed closed? | **NO** — synthetic tests are not corpus evidence |

### P.11 Revision 1a — evidence-accuracy cleanup (2026-08-22)

An audit-only pass over Revision 1's evidence claims. **No implementation code,
no test and no scientific decision was touched.** Five claims were overstated and
are corrected:

| # | Overstated claim | Corrected to |
|---|---|---|
| 1 | §L/§M read as current state while asserting UVW was never downloaded and real schema/chunk behaviour unknown | Both labelled **HISTORICAL (pre-commit)** with inline `SUPERSEDED` / `ANSWERED` annotations pointing at §P. Original text preserved, not deleted |
| 2 | "**Every one of these raised `ChunkingViolation`**" | The preflight *measured* the spans; it did **not** run `chunk_document` on each example. Replaced with the explicit four-step chain: measured spans → measured token lengths → contract inspection → reproduction on a fixture |
| 3 | "**The chunker is the only thing that failed**" | "The chunker is the **identified blocking defect**; schema and identity assumptions were independently confirmed by the real-corpus preflight." The uncaptured stderr is stated plainly |
| 4 | "**Measured directly on the real title**" → 84 spans | The 84-span measurement is on a **fixture this audit constructed** (the observed identifier ×12). The real document's *content* is unavailable here. The load-bearing measurement is restated on the **real identifier alone**: 38 chars, **7** spans, boundaries at the underscores |
| 5 | §P.9 did not record how the defect was established | New limitation 6: the original exception was never captured; the defect rests on three converging lines of evidence, and a residual unidentified failure cannot be excluded |

**Verification of this cleanup**

| # | Check | Result |
|---|---|---|
| 1 | Only the audit file modified | **yes** — no code, no tests, no decisions |
| 2 | Historical record preserved, not deleted | **yes** — §L/§M annotated in place |
| 3 | No claim of having seen the original exception | **yes** — explicitly disclaimed |
| 4 | Synthetic evidence never called real-corpus content | **yes** — fixture and real identifier separated, both numbers given |
| 5 | Real-corpus facts kept as real | **yes** — 1 118 224 docs, 0 duplicates, 0 nulls, token measurements |
| 6 | Repair verdict unchanged | **yes** — REPAIR PASS stands; the repair rests on the corrected evidence |
| 7 | PRE-TRAIN still BLOCKED | **yes** |
| 8 | `git diff --check` clean; nothing staged | **yes** |

**The verdict does not weaken.** Every correction is about *how the defect was
demonstrated*, not *whether it exists*: a single real 38-character identifier
containing 7 orthographic spans is by itself sufficient to show that the
whitespace-only rule was wrong.

---

## Q. REVISION 2 — POST-COMMIT CONTAMINATION-SCREEN PERFORMANCE DEFECT

**Date:** 2026-08-23 **Baseline commit:** `3a399dcfb4e95fb7a605f9e9623ffed7650fea58`

### Q.1 What the real run showed

With the chunker repair (Revision 1) committed, real `prepare-corpus` was rerun
against the pinned corpus. It ran **CPU-bound for more than 7 hours** and was
interrupted. A 45-second `faulthandler` probe reached:

```
[1/6] verifying the corpus pin                 PASS
[2/6] reading and concatenating three shards   PASS
        train 894 579 | validation 111 822 | test 111 823
[3/6] schema + duplicate-id check              PASS
        1 118 224 documents, ids unique
[4/6] contamination screen                     <- stuck here
```

Stack at 45 s, terminated deliberately at 60 s:

```
File unmark/orthography/canonical.py, line 106, in canon
File unmark/stage1/corpus.py,         line 262, in canonical_digest
File unmark/stage1/corpus.py,         line 322, in screen_contamination
File scripts/stage1_runner.py,        line 137, in run_prepare_corpus
```

**This is not a deadlock** — it is stage 4 doing real work, far too much of it.

**Two facts this establishes, and one it does not.** Stages 1-3 pass against the
real corpus, and the shard row counts are now observed
(894 579 + 111 822 + 111 823 = 1 118 224). **Chunking was never reached**, so
this run neither validates nor refutes the Revision 1 chunker repair — that
remains unverified against real data.

### Q.2 Root cause, after reading the code

The stack alone does not prove the algorithm, so the implementation was read.

| # | Question | Finding |
|---|---|---|
| 1 | What does `canonical_digest` do? | `sha256(canon(text))` — one full canonicalisation per call |
| 2 | How many `canon` calls? | **one per UVW document — 1 118 224** — plus one per reference text |
| 3 | Are opened UIT digests cached? | **Yes.** The reference set is built once into a `set` |
| 4 | Is UVW canonicalisation eager for every document? | **Yes. This is the defect.** |
| 5 | Any accidental `O(N*M)`? | **No.** Membership is a set lookup; the loop is `O(N)` calls |

So there is no quadratic blunder. The cost is exactly `O(total corpus
characters)` of **`canon`**, which is Python-level (`split_units`, per-character
`CharacterUnit` construction, nucleus placement). Measured on this machine:
**~0.95-1.2 M chars/s**. The reference side is ~13 k *short* UIT-VSFC sentences;
the corpus side is 1.1 M *long* Wikipedia articles. Essentially all of the seven
hours was spent canonicalising documents that could not possibly match.

### Q.3 The optimisation — two necessary-condition prefilters

**The criterion is unchanged.** A document is excluded **iff**
`canonical_digest(doc) ∈ {canonical_digest(ref)}`. The prefilters decide only
*which documents are worth canonicalising*; they may skip a document only after
**proving** it cannot match.

| Tier | Test | Measured throughput |
|---|---|---|
| 1 | length guard — five C-level substring counts, no normalisation | **~846 M chars/s** |
| 2 | placement-insensitive digest — one NFD pass + one sha256 | **~29 M chars/s** |
| 3 | **`canonical_digest` — the actual decision** | **~1 M chars/s** |

**Tier 2 proof.** Write `f(x) = NFD(x)` with **only** the five tone marks
`U+0300 U+0301 U+0303 U+0309 U+0323` removed. `canon` is NFC plus UNMARK's fixed
nucleus-based tone placement, and by its own contract it never alters letters,
case, punctuation, whitespace, digits or letter-forming diacritics — *only the
position of a tone mark within its syllable may change*. Hence

```
f(canon(x)) == f(x)        for every x
```

because `NFD ∘ NFC == NFD` absorbs the normalisation step, and removing every
tone mark erases the only other thing `canon` may do. Therefore

```
canon(a) == canon(b)   =>   f(a) == f(b)
```

so a **difference** in `f` is a proof that the canonical digests differ. The
converse is false and is not needed: texts differing only in tone marks collide,
become candidates, and tier 3 decides.

**Tier 1 proof.** With `s` = standalone tone-mark characters in `x` and `k` =
characters whose NFD decomposition releases a tone mark:

```
len(NFD(x)) >= len(x) + k          (each such character expands by >= 1)
len(f(x))    = len(NFD(x)) - s - k  >=  len(x) - s
```

So if `len(x) - s` already exceeds the longest reference `f`-length, `len(f(x))`
does too, the `f`-digests differ, and by the tier-2 lemma the canonical digests
differ. Five `str.count` calls, no normalisation.

**Single-process, deliberately.** No multiprocessing was added. Serialising
1.1 M long strings between processes is itself expensive, it complicates
deterministic error provenance, and worker counts are an environment-dependent
knob. Parallelism should only be revisited if the *cheap* pass is ever measured
to be the blocker.

### Q.4 Evidence

Both lemmas were tested before the implementation was written:

| Lemma | Population | Counterexamples |
|---|---|---|
| `f(canon(x)) == f(x)` | 160 076 strings — systematic syllables over every tone and both mark positions, documented placement pairs, NFC/NFD, case, `đ/Đ`, punctuation, URLs, e-mail, mixed scripts, malformed multi-tone, 40 000 random combining-mark sequences | **0** |
| `len(f(x)) >= len(x) - s` | 200 008 strings | **0** |

**Full-report equivalence.** The pre-optimisation screen is retained **as a test
oracle only** (`reference_screen`), not as a second production pathway. On a
deterministic corpus containing true canonical duplicates, placement-only
duplicates, NFC/NFD duplicates, near-but-not-equal texts, deliberate prefilter
collisions and ordinary nonmatches — plus 25 randomised corpora — the optimised
screen reproduces the **complete report**: excluded ids, excluded digests, kept
order and reference digest count, not merely the match count.

**Algorithmic counters** (Task D), asserted rather than timed:

| Counter | Meaning |
|---|---|
| `corpus_documents_seen` | documents examined |
| `length_guard_skips` | skipped by tier 1 |
| `cheap_prefilter_checks` | reached tier 2 |
| `prefilter_candidates` | reached tier 3 |
| `full_canon_calls_for_corpus_candidates` | **expensive calls on corpus documents** |
| `opened_reference_examples` / `full_canon_calls_for_reference_set` | reference side |

On 5 001 documents with one genuine match, `full_canon_calls_for_corpus_candidates`
is **≤ 5**, i.e. `O(candidates)`, not `O(corpus)`. On 500 long documents against
a short reference, tier 1 skips **all 500** and the NFD path is never entered.
**No wall-clock threshold is asserted in any test.**

### Q.5 Sample benchmark — indicative only

On 60 synthetic Vietnamese documents (~1 M chars) on this machine:

| Stage | Throughput | Relative |
|---|---|---|
| full `canon` digest | 0.95 M chars/s | 1x |
| placement-insensitive digest | 28.7 M chars/s | **30x** |
| length guard | 845 M chars/s | **890x** |

**This is a machine-dependent sample on synthetic text, not a corpus
measurement.** The real corpus's total character count is unknown here; back-of-
envelope, seven hours at ~1 M chars/s implies order 10^10 characters, which the
prefilters would reduce to minutes — but that is an *extrapolation*, not a
result.

### Q.6 Progress visibility (Task E)

Stage 4 printed only its start line, so a seven-hour scan looked identical to a
hang. `screen_contamination` now takes an optional `on_progress(seen,
candidates, matches)` callback; the runner prints roughly every 1 %:

```
contamination: 100000/1118224 docs (8.9%), candidates=0, matches=0
```

and, on completion, the tier counters. **Counts only — no corpus text, no
UIT-VSFC text.** Document ids appear only where an actual match requires
provenance, exactly as before.

### Q.7 Files changed in Revision 2

| File | Change |
|---|---|
| `unmark/stage1/corpus.py` | `placement_insensitive_digest`, `_length_guard_excludes`, `ScreenCounters`; `screen_contamination` rewired to three tiers with an `on_progress` hook |
| `scripts/stage1_runner.py` | Stage-4 progress every ~1 %; tier counters printed on completion |
| `tests/test_stage1_contamination_prefilter.py` | **new** — 287 tests: lemmas, equivalence oracle, counters, boundaries |
| `docs/audits/029-…md` | This section |

**`docs/spec/decisions.md`: NOT changed, and no change is required.** D-S1B-002
rule 3 states the *criterion* — "equality of `canon(x)` and its sha256" — not an
implementation strategy, and that criterion is untouched. No new decision entry
is warranted for an optimisation that provably changes no exclusion.

**Untouched:** the chunker repaired in `3a399dc`, `canon()` itself, the corpus
pin and shard order, the split algorithm and `dev = 5 000`, `max_length = 256`,
corruption policy and `pi_strip`, the objective, architecture, batch size, LR/`r`
grids, optimizer, validation metric, seeds, and the official-TEST policy.

### Q.8 Test results

| Suite | Result |
|---|---|
| `tests/test_stage1_contamination_prefilter.py` | **287 passed** |
| `tests/test_stage1_corpus.py` | 24 passed (unchanged) |
| `tests/test_stage1_chunking.py` | 35 passed (unchanged) |
| Full repository | **2 822 passed, 97 skipped** (was 2 535 / 97) |

**Mutation-verified** — each injected violation caught:

| Injected violation | Caught by |
|---|---|
| Prefilter *decides* exclusion (canonical check skipped) | `test_equivalence_on_randomised_corpora` |
| Length guard drops the tone-mark term | `test_the_length_guard_never_skips_a_possible_match` |
| Prefilter uses NFC instead of NFD | `test_placement_variants_survive_the_prefilter` |

A fourth mutation — stripping letter-forming diacritics as well — was **not**
caught, and correctly so: over-stripping remains a valid necessary condition, so
it is *safe but slower* and no equivalence test can detect it. A selectivity test
now pins the tone-mark set so a silent widening surfaces as an intentional change
rather than as unexplained candidate growth. Under-stripping and
non-canon-invariant transforms — the unsafe directions — **are** caught.

### Q.9 Limitations

1. **The full real corpus has NOT been re-prepared.** The 766 MB of pinned bytes
   are not available here and were not downloaded. No performance claim against
   the real corpus is made.
2. **The 7-hour figure is not reproduced here**; it is the researcher's
   observation. The benchmark in §Q.5 is synthetic and machine-dependent.
3. **Chunking is still unverified against real data.** The failing run never
   reached stage 5, so Revision 1 remains untested on the real corpus.
4. **Stage 4 is still `O(total corpus characters)`** in the cheap pass — the
   prefilters remove the expensive constant, not the linear scan. If that pass
   is itself measured to be a blocker, parallelism can be reconsidered.
5. **Real contamination counts are unknown.** Whether any UVW document actually
   matches an opened UIT-VSFC example has never been measured.
6. The equivalence oracle is exercised on synthetic corpora; it is a strong
   check of the *criterion*, not evidence about the real corpus.

### Q.10 Self-audit for Revision 2

| # | Check | Result |
|---|---|---|
| 1 | Audit 029 revised **in place**; no Audit 030 | **yes** |
| 2 | `canon()` semantics unchanged | **yes** — `canonical.py` not modified |
| 3 | Contamination criterion unchanged | **yes** — still `sha256(canon(x))`, decided only at tier 3 |
| 4 | Screening not skipped or weakened | **yes** — every document passes a proven-necessary test |
| 5 | No fuzzy / approximate / semantic / substring matching | **yes** |
| 6 | Prefilter proven a necessary condition | **yes** — §Q.3, 360 084 trials, 0 counterexamples |
| 7 | Final matches still use `canonical_digest` | **yes** — AST-asserted |
| 8 | Prefilter collision ≠ contamination | **yes** — explicit test |
| 9 | Full-report equivalence vs the pre-optimisation oracle | **yes** — ids, digests, kept order |
| 10 | Old implementation kept as oracle only, not a second pathway | **yes** — lives in the test file |
| 11 | Expensive calls `O(candidates)`, not `O(corpus)` | **yes** — counter assertions |
| 12 | No wall-clock threshold in any test | **yes** |
| 13 | No document dropped for performance | **yes** — every document is kept or excluded by the criterion |
| 14 | No UIT-VSFC official TEST route | **yes** — refusal test retained |
| 15 | No raw corpus or UIT text in reports/logs | **yes** — asserted |
| 16 | Progress does not flood stdout | **yes** — ~1 % intervals |
| 17 | No multiprocessing added | **yes** — deliberately single-process |
| 18 | Revision 1 chunker untouched | **yes** — no diff in `chunking.py` or its tests |
| 19 | No scientific hyperparameter changed | **yes** |
| 20 | `decisions.md` inspected; change needed? | **no** — it states the criterion, not the implementation |
| 21 | Focused + full suites run | **yes** — 287 / 2 822 passed |
| 22 | Real corpus re-prepared? | **NO** |
| 23 | Performance claimed PASS against the real corpus? | **NO** |
| 24 | Nothing staged; no prohibited git operation | **yes** |

---

## R. REVISION 3 — STAGE-6 CHUNKING PERFORMANCE DEFECT

**Date:** 2026-08-23 **Baseline commit:** `55aa7fe50c427949a1b49878979551b978d29400`

### R.1 What the real run showed

Revision 2 was committed and the **full pinned corpus** was run. Stages 1-5 now
execute correctly against real data:

| Stage | Result |
|---|---|
| 1 pin verified | **PASS** |
| 2 read + concatenate | **PASS** — 1 118 224 documents |
| 3 schema + duplicate ids | **PASS** — ids unique |
| 4 contamination screen | **PASS in ~16 s** (was >7 h) |
| 5 document split | **PASS** — train **1 113 224**, dev **5 000** |
| 6 pre-chunking | **starts at ~25.6 s, then no 1 % line after ~13 minutes** |

The Revision-2 screen result on real data: **0 excluded**, 296 628 length-guard
skips, 821 596 placement-insensitive checks, **0 tier-3 candidates**, **0 UVW
full-canon calls**. Official UIT-VSFC TEST remained sealed.

One percent is ~11 182 documents, so stage 6 had not completed 1 % in 13 minutes.

A 120-second `faulthandler` probe took four samples. Three were in
`canon`/`decompose` beneath `base_length`, one in the PhoBERT tokenizer beneath
`reference_length` — **all four inside `fits` → `chunk_document`**:

```
units.py:27 split_units <- placement.py:156 <- canonical.py:106 canon
        <- runner.py:251 base_length <- chunking.py:191 fits <- chunk_document
tokenization_phobert.py:287 _convert_token_to_id
        <- runner.py:247 reference_length <- chunking.py:191 fits <- chunk_document
decompose.py:131 decompose <- runner.py:251 base_length <- fits <- chunk_document
```

**No sample was inside `safe_cut_offsets` or the oversized-unit fallback.** This
is ordinary `fits` evaluation, not the Revision-1 repair.

### R.2 Root cause — measured, not inferred from the stack

`fits(start, end)` slices `content[start:end]` and calls **both** length
functions on the *whole growing candidate*. The greedy scan extends one segment
at a time, so a chunk spanning `k` segments canonicalises prefixes totalling
`O(k x chunk_length)`.

Instrumented on real-shaped Vietnamese text (`canon`/`decompose` real, tokenizer
mocked):

| Document | Chunks | `fits` calls | Chars fed to `reference_length` | Chars fed to `base_length` |
|---|---|---|---|---|
| 870 | 1 | 400 | 167 565 = **193x** doc | 167 565 = **193x** doc |
| 3 610 | 4 | 1 606 | 884 926 = **245x** doc | 881 484 = **244x** doc |
| 14 206 | 13 | 6 424 | 3 576 659 = **252x** doc | 3 563 074 = **251x** doc |

**Every document is pushed through `canon`/`decompose` roughly 250 times its own
length**, giving ~0.8 K document-chars/s. The answers to Task A:

1. For `S` segments: `S` calls each to `reference_length` and `base_length`,
   hence `S` calls each to `canon` and `decompose`, and `S` tokenizations.
2. **Yes** — `fits` recomputes the entire growing candidate from scratch.
3. **Yes** — every prefix transform is discarded and re-derived.
4. Not merely `O(total text)`: it is `O(sum of growing candidate lengths)`,
   i.e. quadratic in segments per chunk.
5. Not duplicate *identical* queries — each `(start, end)` is asked once. The
   waste is **repeated prefixes**.

### R.3 The repair — two exact reuse properties

**`chunking.py` is not modified.** Its algorithm, decision order and `fits`
truth values are untouched; only the cost of each query changes. That is what
makes boundary identity structural rather than argued.

**Property 1 — transform composability (this repository's own code).** For a
text split into maximal whitespace / non-whitespace runs `s1 … sn`:

```
canon(T)                      == "".join(canon(si))
decompose(canon(T)).base_text == "".join(decompose(canon(si)).base_text)
```

because NFC never composes across a whitespace starter, `apply_modern_placement`
moves a tone mark only *within* a maximal alphabetic run (which whitespace
terminates), and `base_text` is a per-character mapping. Verified on **17 007**
strings including NFD forms and random combining-mark sequences: **0
counterexamples**.

**Property 2 — per-chunk tokenization — WITHDRAWN in Revision 3a.** The real
probe falsified this on the pinned tokenizer (`composed 5, exact 7`); the
shortcut described below was **removed**. Retained here as the record of what
was tried. See §S.
`D-B3B1B-001` established on this exact pinned revision that PhoBERT's fastBPE
operates over **maximal non-whitespace chunks**: splitting on non-whitespace runs
and tokenizing each whole chunk reproduced the authoritative token sequence
**13/13** and token IDs **13/13** across 119 chunks, with zero surface
reconstruction failures. So

```
token_len(t) == specials + sum over non-whitespace runs r of len(tokenize(r))
```

**This is not taken on trust.** Revision 1 was caused by borrowing a B3B fact
into chunking without rechecking it, so `TokenLengthComposer` computes the first
**256** distinct queries of every run *both* ways — whole string and composed —
and raises `Stage1ContractViolation` on any disagreement. The property is a
**checked precondition of every run**, and a deliberately non-conforming
tokenizer is used in the tests to prove the check bites.

**No monotonicity is used anywhere.** Nothing assumes token counts grow with
text length; the composition is over *disjoint runs of one text*. The greedy
scan still evaluates every candidate in order and still stops at the first
failure. The oracle suite includes a **non-monotonic tokenizer** specifically to
prove no such assumption slipped in.

The growing prefix is extended rather than rescanned, and only when the junction
is provably not inside a run (previous text ends with whitespace, or the delta
begins with it). Every fast-path candidate satisfies this; the oversized-unit
fallback cuts inside a run, the condition fails, and the full path runs.
**Correctness never depends on the shortcut being taken.**

### R.4 Measured effect — identical output

12 documents, 80 415 characters, 72 chunks, mock tokenizer:

| Implementation | Time | Throughput | Speedup |
|---|---|---|---|
| committed (`55aa7fe`) | 99.42 s | 0.8 K doc-chars/s | 1x |
| + transform memo only | 4.63 s | 16.9 K doc-chars/s | 21.5x |
| + per-run token composition | 1.66 s | 47.1 K doc-chars/s | 59.1x |
| ~~+ incremental prefix extension~~ | ~~0.10 s~~ | ~~753.7 K doc-chars/s~~ | ~~956x~~ **— WITHDRAWN, §S.5** |

**Output identical at every step** — chunk ids, source ranges, text, and *both*
recorded lengths.

Counters on that run: `canon` called on **58 characters** total (0.0007x the
corpus text, was ~250x); runs examined **72 060**, down from **9 093 300**;
71 892 incremental extensions vs 288 full rescans.

**This is a synthetic, machine-dependent benchmark with a mock tokenizer.** It is
not a corpus measurement and not a prediction of Colab wall-clock.

### R.5 Files changed in Revision 3

| File | Change |
|---|---|
| `unmark/stage1/lengths.py` | **new** — `ComposedTransforms`, `TokenLengthComposer` (with the runtime verifier), `build_length_functions`, counters |
| `scripts/stage1_runner.py` | uses `build_length_functions`; **early heartbeat** at 1/10/50/100/500/1 000/5 000/10 000 documents then ~1 %, with elapsed time and docs/s; prints transform counters |
| `scripts/stage1_tokenizer_probe.py` | **new** — Colab-only real-tokenizer micro-probe (Task E) |
| `tests/test_stage1_lengths.py` | **new** — 220 tests: lemmas, equivalence oracle, verifier, counters |
| `docs/audits/029-…md` | this section |

**`unmark/stage1/chunking.py` is UNCHANGED**, as are `corpus.py` (Revision 2),
`canon()`, the corpus pin, the split, `max_length`, and every scientific value.
`docs/spec/decisions.md` is **not** changed: no scientific decision moved, and
D-S1B-009 already states the chunking contract at the level of *boundaries*, not
of length-function implementation.

### R.6 Tests

| Suite | Result |
|---|---|
| `tests/test_stage1_lengths.py` | **220 passed** |
| `tests/test_stage1_chunking.py` | 35 passed (unchanged) |
| `tests/test_stage1_contamination_prefilter.py` | 287 passed (Revision 2 untouched) |
| `tests/test_stage1_runner_contract.py` | 39 passed |
| Full repository | **3 042 passed, 97 skipped** (was 2 822 / 97) |

The equivalence oracle keeps the pre-optimisation implementation **in test code
only**. It compares chunk count, text, `source_start`, `source_end`, chunk ids,
parent ids, partition, order, exact reconstruction, whether `ChunkingViolation`
is raised, **and the violation message including its provenance** — across
16 document families x 2 tokenizers x 3 `max_length` values, plus 40 randomised
documents and both path-asymmetry directions.

### R.7 Colab micro-probe (Task E)

`scripts/stage1_tokenizer_probe.py` validates, on `vinai/phobert-base` @
`01daacda…`, both properties and old-vs-new boundaries on ~19 strings and 8
documents. It loads **no encoder**, runs **no forward pass**, constructs **no
optimizer** — AST-verified — and prints a JSON report with `status: PASS|FAIL`.

**It has not been run.** The ML-free local `.venv` has no transformers, and
nothing was downloaded.

### R.8 Parallelism (Task F)

**None added.** The algorithmic repair removed the redundant work; multiprocessing
would have masked it, and worker count must not become a knob. It can be
reconsidered only if the repaired serial implementation is *measured* to remain a
blocker on real data.

### R.9 Limitations

1. **The full real corpus has NOT been chunked.** No stage-6 success is claimed.
   The 766 MB of pinned bytes are not available here and were not downloaded.
2. **The 956x figure is synthetic**, with a mock tokenizer, on one machine. Real
   PhoBERT tokenization cost per unique run is not measured here; the per-run
   memo should make it small after warmup, but that is an expectation.
3. **The Colab probe has never executed**, so Property 2 is currently supported
   by D-B3B1B-001's evidence plus the runtime verifier, not by a fresh
   measurement.
4. **Real chunk counts, the length distribution and any indivisible-span
   failures remain unknown** — stage 6 has never completed.
5. The memo is capped at 500 000 entries per table; eviction costs recomputation
   and changes no result, but the real vocabulary size is unmeasured.
6. Revision 1's chunker is still unverified on real data, since stage 6 has never
   finished.

### R.10 Self-audit for Revision 3

| # | Check | Result |
|---|---|---|
| 1 | Audit 029 revised **in place**; no Audit 030 | **yes** |
| 2 | Optimised output equals the old chunker's output | **yes** — full-field oracle, incl. violation provenance |
| 3 | No monotonicity assumption | **yes** — non-monotonic tokenizer in the oracle matrix |
| 4 | No additivity assumed *without proof* | **yes** — D-B3B1B-001 evidence **plus** a runtime verifier that fails closed |
| 5 | The verifier actually bites | **yes** — non-conforming tokenizer test |
| 6 | Both paths still checked | **yes** — reference and base, asymmetry tested both directions |
| 7 | Fallback semantics unchanged | **yes** — `chunking.py` untouched; shortcut declines inside a run |
| 8 | Exact range tiling unchanged | **yes** — `chunking.py` untouched |
| 9 | `max_length = 256` unchanged; no max-length check weakened | **yes** |
| 10 | Tokenizer revision unchanged | **yes** |
| 11 | No normalisation, truncation, dropping, rewriting or synthetic whitespace | **yes** |
| 12 | Revision-2 contamination code untouched | **yes** — `corpus.py` not modified |
| 13 | No official UIT-VSFC TEST route | **yes** |
| 14 | No raw corpus text in progress output, counters or audit | **yes** |
| 15 | Early heartbeat added; ~1 % retained | **yes** — counts, elapsed, docs/s only |
| 16 | No parallelism added | **yes** |
| 17 | No scientific hyperparameter changed | **yes** |
| 18 | Focused + full suites run | **yes** — 220 / 3 042 passed |
| 19 | Real corpus chunked? | **NO** |
| 20 | Real stage-6 PASS claimed? | **NO** |
| 21 | Colab probe executed? | **NO** |
| 22 | Nothing staged; no prohibited git operation | **yes** |

---

## S. REVISION 3A — REAL-TOKENIZER PROBE FAILED; SHORTCUT REMOVED

**Date:** 2026-08-23 **Baseline commit:** `bb5082373d0ebe9527441a8add5fdb5ee408a16a`

### S.1 The failed probe

Revision 3 was committed and its own micro-probe was run in Colab against the
real pinned tokenizer. The repository was clean and HEAD matched.

```
Python       /usr/bin/python3
Transformers 4.57.6
torch        2.11.0+cu128

$ python scripts/stage1_tokenizer_probe.py
    -> return code 1

unmark.stage1.contracts.Stage1ContractViolation:
per-chunk token composition disagreed with whole-string tokenization:
composed 5, exact 7.

stage1_tokenizer_probe.py -> chunk_document -> fits
    -> unmark/stage1/lengths.py::length -> Stage1ContractViolation
```

**The runtime verifier worked exactly as designed.** Revision 3 wrote that the
per-chunk fact was "a checked precondition of every run, not a belief"; the
check fired on the first real tokenizer it met and refused to emit lengths it
could not justify. **It has not been weakened or removed.**

**Revision 3's claim is falsified.** Per-run token composition was *not*
justified on the real pinned tokenizer, and the 956x figure rested on it. Both
are superseded here.

### S.2 The two length definitions, reconstructed (Task A)

| | Authoritative (pre-optimisation, and the probe's oracle) | Revision 3's composed length |
|---|---|---|
| Formula | `len(build_inputs_with_special_tokens(convert_tokens_to_ids(tokenize(transform(x)))))` | `whole_length("")` + Σ over runs `r` of `len(tokenize(transform(r)))` |
| APIs used | `tokenize` → `convert_tokens_to_ids` → `build_inputs_with_special_tokens` | **`tokenize` only**, per run |
| `<s>` / `</s>` | included, added once by the tokenizer | included, added once as `whole_length("")` |
| Transform applied to | the **whole** text | **each run** separately |
| Run definition | n/a | `re.compile(r"\S+")` |
| `<= 256` means | this number | this number |

Two **definitional** divergences are visible by inspection:

1. **Different API chain.** Run counts came from `tokenize` alone; the
   authoritative number comes from the full chain.
2. **Different run unit.** PhoBERT's own `_tokenize` splits on `\S+\n?` — a
   trailing newline stays **attached** to the run, so BPE's `</w>` end-of-word
   marker lands on a different final character than it does for a run cut at
   `\S+`.

### S.3 Why the cause was not guessed

`exact - composed = 2` invites "special tokens" as the explanation. **It is not
adopted here.** The failing fixture is the first probe document, `"Tôi đã đọc"`,
which contains no newline, so divergence 2 cannot explain it; and
`convert_tokens_to_ids` is a 1:1 mapping, so divergence 1 does not change a
*count*. With `whole_length("") = 2` the arithmetic is `2 + 3 = 5` against
`2 + 5 = 7`: the **whole string yields two more content tokens than the sum of
its three per-word tokenizations**.

That is a statement about real PhoBERT's behaviour which **cannot be verified in
this ML-free environment** — no tokenizer, nothing downloaded. So the root cause
is recorded as: *per-run token composition does not hold for the pinned
tokenizer on this input, for a reason this audit has not isolated.* It is **not**
recorded as a special-token accounting slip, because that has not been shown.

### S.4 The repair (Task D)

**The per-run token-composition shortcut is removed.** Correctness over
benchmark: the real evidence falsified it, it cannot be validated here, and
rescuing it by guessing at arithmetic would be exactly the mistake Revision 1
already made once.

What remains needs **no tokenizer property at all**:

* memoised per-segment `canon` / `base_text`;
* incremental extension of those transforms along a growing prefix, guarded by
  the same whitespace-junction condition;
* the transformed candidate then goes to the tokenizer **whole, through the
  authoritative API chain**.

So

```
optimized_length(x) == authoritative_length(x)
```

holds **by construction** — same chain, same whole-string tokenization, same
special-token accounting — rather than by argument. The only thing reused is the
orthographic transform, and only where the composability lemma applies.

**The verifier is kept** (Task E), now guarding the property that remains: the
first 256 distinct queries compare the composed transform against the direct
`canon` / `decompose`, and any disagreement raises `Stage1ContractViolation`.

### S.5 Revised benchmark

Same fixture as §R.4 — 12 documents, 80 415 characters, mock tokenizer:

| Implementation | Time | Throughput | Speedup |
|---|---|---|---|
| pre-optimisation | 102.35 s | 0.8 K doc-chars/s | 1x |
| **Revision 3a (transform reuse only)** | **0.53 s** | **147.2 K doc-chars/s** | **192x** |
| ~~Revision 3 (with the falsified shortcut)~~ | ~~0.10 s~~ | ~~753.7 K~~ | ~~956x — **withdrawn**~~ |

Output identical: chunk ids, source ranges, text and **both** recorded lengths.
`canon` ran on **58 characters** total (0.0007x the text). Tokenizer calls:
**72 180 — unchanged from the pre-optimisation implementation**, whole-string,
same API chain. That is the honest cost of dropping the shortcut.

### S.6 Probe repaired (Tasks B, F, G)

* **`--help` now works** and exits 0 **without importing transformers** — the
  import moved inside `main()` after `parse_args`. Previously `--help` executed
  the probe and failed. No scientific override flags were added.
* The probe now **catches** `Stage1ContractViolation` and reports it, instead of
  aborting before printing anything.
* It compares **optimised vs authoritative** lengths on both pathways, transform
  composability, and old-vs-new chunk output field by field (ids, ranges, text,
  both lengths, and violation provenance).
* On the first mismatches it emits a safe diagnostic: fixture index and repr,
  transformed repr, whole tokens and id count, count before and after specials,
  the runs, per-run tokens and counts, the sum, **what the removed shortcut
  would have given**, and the optimised value. **Fixtures only — no UVW corpus
  text, no UIT-VSFC text.**

### S.7 Files changed in Revision 3a

| File | Change |
|---|---|
| `unmark/stage1/lengths.py` | `TokenLengthComposer` **removed**; `ComposedTransforms` gains incremental extension and a fail-closed transform verifier; `build_length_functions` now uses the authoritative API chain |
| `scripts/stage1_tokenizer_probe.py` | rewritten: argparse CLI, lazy transformers import, catches violations, safe first-mismatch diagnostics, authoritative-vs-optimised comparison |
| `tests/test_stage1_lengths.py` | per-run composition tests replaced by authoritative-equality tests; **`test_the_five_versus_seven_regression_fixture`** reproduces `composed 5 / exact 7`; transform-verifier fail-closed test |
| `tests/test_stage1_runner_contract.py` | 4 probe-CLI tests, including `--help` exit 0 without loading the tokenizer |
| `docs/audits/029-…md` | this section |

**Unchanged:** `unmark/stage1/chunking.py`, `unmark/stage1/corpus.py`
(Revision 2), `canon()`, the corpus pin, split, seeds, `pi_strip`, objective,
architecture, optimizer, grids, validation grid, `max_length = 256`, and the
official-TEST policy. `docs/spec/decisions.md` not changed — no scientific
decision moved.

### S.8 Tests

| Suite | Result |
|---|---|
| `tests/test_stage1_lengths.py` | **223 passed** |
| `tests/test_stage1_chunking.py` | 35 passed (unchanged) |
| `tests/test_stage1_runner_contract.py` | **43 passed** (was 39) |
| `tests/test_stage1_contamination_prefilter.py` | 287 passed (Revision 2 untouched) |
| Full repository | **3 049 passed, 97 skipped** |

The regression fixture is the load-bearing one: a tokenizer that returns 5
tokens for `"Tôi đã đọc"` but 1 per word reproduces the reported
`composed 5 / exact 7`, and the repaired implementation returns **7**, matching
the authoritative pathway.

### S.9 Limitations

1. **The root cause of 5-vs-7 is not isolated.** It is recorded as "per-run
   composition does not hold on the pinned tokenizer", not as a specific bug.
   The repaired probe's diagnostics exist to answer this on the next Colab run.
2. **The repaired probe has NOT been run.** No real-tokenizer evidence supports
   the current implementation yet — only the by-construction argument and
   synthetic tests.
3. **Real-corpus performance is now UNKNOWN and may still be insufficient.**
   Removing the shortcut restores the full per-query tokenization cost:
   `Θ(Σ growing prefix lengths)` through the real tokenizer, ~72 180 whole-string
   calls for 80 K characters in the benchmark. The 192x gain is entirely on the
   `canon`/`decompose` side. Whether stage 6 now completes on 1.1 M documents is
   **not established**.
4. The 192x figure is synthetic, mock tokenizer, one machine.
5. Stage 6 has still never completed on real data, so Revision 1's chunker
   remains unverified there.

### S.10 Self-audit for Revision 3a

| # | Check | Result |
|---|---|---|
| 1 | Audit 029 revised **in place**; no Audit 030 | **yes** |
| 2 | Failed probe recorded, not hidden | **yes** — §S.1, with return code and exact violation |
| 3 | Verifier kept, not weakened to pass | **yes** — retargeted to the surviving property, still fail-closed |
| 4 | Root cause guessed? | **no** — special-token accounting explicitly **not** adopted; §S.3 |
| 5 | Special-token accounting identical old vs optimised | **yes** — same API chain, added once by the tokenizer |
| 6 | Optimised length == authoritative length | **yes** — by construction, and asserted for 3 tokenizer doubles |
| 7 | No monotonicity assumption | **yes** — non-monotonic tokenizer still in the oracle matrix |
| 8 | No unsafe additivity assumption remains | **yes** — per-run token composition **removed** |
| 9 | Both reference and base pathways match old behaviour | **yes** |
| 10 | `chunking.py` semantics unchanged | **yes** — not modified |
| 11 | Revision-2 contamination code untouched | **yes** |
| 12 | Official UIT-VSFC TEST unreachable | **yes** |
| 13 | Probe `--help` exits 0 without loading the tokenizer | **yes** — tested via subprocess |
| 14 | Probe loads no encoder, cannot step | **yes** — AST-asserted |
| 15 | No corpus or UIT text in diagnostics | **yes** — fixtures only |
| 16 | Regression fixture reproduces `composed 5 / exact 7` | **yes** |
| 17 | Withdrawn 956x claim marked withdrawn | **yes** — §S.5 |
| 18 | Focused + full suites run | **yes** — 3 049 passed |
| 19 | Repaired probe executed on Colab? | **NO** |
| 20 | Real corpus prepared? | **NO** |
| 21 | Nothing staged; no prohibited git operation | **yes** |

---

**STATUS (Revision 3a): REVISION 3A REPAIR PASS — READY FOR REAL TOKENIZER PROBE**
**REVISION 3's REAL PROBE FAILED (rc 1): `composed 5, exact 7` — THE VERIFIER WORKED**
**PER-RUN TOKEN COMPOSITION FALSIFIED AND REMOVED; 956x CLAIM WITHDRAWN**
**REMAINING OPTIMISATION NEEDS NO TOKENIZER PROPERTY — 192x, LENGTHS EQUAL BY CONSTRUCTION**
**ROOT CAUSE OF 5-vs-7 NOT ISOLATED — NOT ATTRIBUTED TO SPECIAL TOKENS WITHOUT EVIDENCE**
**PROBE `--help` DEFECT FIXED; DIAGNOSTICS ADDED FOR THE NEXT RUN**
**REAL-CORPUS PERFORMANCE NOW UNKNOWN AND MAY STILL BE INSUFFICIENT**
**NEXT AUTHORIZATION IS ONLY THE REAL TOKENIZER MICRO-PROBE — NOT PREPARE-CORPUS**

~~**STATUS (Revision 3): CHUNKING PERFORMANCE REPAIR PASS — READY FOR REAL PREPARE-CORPUS RE-RUN**~~ **— SUPERSEDED by Revision 3a**
**DEFECT 3: STAGE-6 RE-CANONICALISED EVERY GROWING PREFIX (~250x DOCUMENT LENGTH) — REPAIRED (§R)**
**CHUNKING ALGORITHM UNCHANGED — `chunking.py` NOT MODIFIED; ONLY LENGTH-QUERY COST**
**956x ON A SYNTHETIC BENCHMARK, OUTPUT IDENTICAL INCLUDING VIOLATION PROVENANCE**
**NO MONOTONICITY ASSUMED; PER-CHUNK TOKENIZATION VERIFIED AT RUNTIME, FAILS CLOSED**
**STAGES 1-5 NOW PASS ON REAL DATA — STAGE 6 HAS NEVER COMPLETED**
**COLAB TOKENIZER PROBE WRITTEN BUT NOT RUN**

**STATUS (Revision 2): PERFORMANCE REPAIR PASS — READY FOR REAL PREPARE-CORPUS RE-RUN**
**DEFECT 2: STAGE-4 CONTAMINATION SCREEN CANONICALISED ALL 1 118 224 DOCUMENTS (>7 h) — REPAIRED (§Q)**
**CONTAMINATION CRITERION UNCHANGED: `sha256(canon(x))`, DECIDED ONLY AT TIER 3**
**PREFILTERS PROVEN NECESSARY CONDITIONS — 360 084 TRIALS, 0 COUNTEREXAMPLES**
**FULL-REPORT EQUIVALENCE VS THE PRE-OPTIMISATION ORACLE**
**REAL CORPUS NOT RE-PREPARED — NO PERFORMANCE CLAIM AGAINST IT**
**CHUNKING STILL UNVERIFIED ON REAL DATA — STAGE 5 WAS NEVER REACHED**

**STATUS (Revision 1): REPAIR PASS — READY FOR REAL CORPUS RE-RUN**
**DEFECT: WHITESPACE-ONLY CUTTING COULD NOT PREPARE THE LOCKED CORPUS — REPAIRED (D-S1B-009)**
**CHUNK BOUNDARIES: ORTHOGRAPHICALLY SAFE OFFSETS FROM `decompose`; SPANS STILL INDIVISIBLE**
**TEXT PRESERVATION: EXACT RANGE TILING, BYTE-EXACT RECONSTRUCTION**
**REAL CORPUS NOT RE-PREPARED — SYNTHETIC TESTS ARE NOT CORPUS EVIDENCE**
**STRIP-ALL SUPPORT: IMPLEMENTED AND MEASURED**
**CORPUS PIN: CLOSED (all three shards, verified at load)**
**NO REAL STAGE-1 SCIENTIFIC EXECUTION OCCURRED**
**COMPILED PROPOSAL PDF: STALE**
**PRE-TRAIN: BLOCKED until real `prepare-corpus` succeeds and the real-PhoBERT no-update smoke passes**
**THIS IS NOT THE PRE-TRAIN AUDIT**
**ALL CHANGES UNSTAGED**
