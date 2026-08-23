# Audit 029 — Stage-1 runner implementation

| | |
|---|---|
| **Audit id** | 029 |
| **Created (UTC)** | 2026-08-22 |
| **Last revised (UTC)** | 2026-08-23 — runner-wiring repair (§W), then this consistency cleanup (§X) |
| **Original baseline HEAD** | `5b07430` (`docs: lock Stage-1 scientific configuration`) — the state at creation |
| **Current baseline HEAD** | `f9c23fe` (`feat: add resumable Stage-1 corpus preparation`) |
| **Predecessor** | [028](028-stage1-scientific-config-review.md) Revision 2 — the authoritative config lock |
| **Scope — as created** | *(historical, superseded)* "Implement the complete pre-training Stage-1 execution stack from the locked Audit 028 configuration. **Execute none of it.**" That was true on 2026-08-22 and is **no longer** the state: the stack has since been run repeatedly against the real pinned corpus. |
| **Scope — CURRENT** | The Stage-1 **corpus-preparation** stack and its repair history against real data. Stages 1-5 execute and pass on the real corpus; Stage 6 is implemented, streamed and resumable but **has never completed and has never committed one checkpoint interval**. Stage-1 *training* is implemented but unexecuted and out of scope here. |
| **Type** | Implementation + tests + **post-commit real-data defect repair**. Revisions §P–§W each record a defect that appeared only against the real corpus or the real pinned tokenizer, not in the local suite. |
| **What HAS run on real data** | Real pinned-tokenizer probe **PASS** (`vinai/phobert-base` @ `01daacda…`, Transformers 4.57.6). Real UVW-2026 downloaded and inspected **in Colab, never in this environment**. **Stages 1-5 PASS on all 1 118 224 documents** — 0 contaminated, 296 628 length-guard skips, 821 596 prefilter checks, 0 candidates, split **1 113 224 / 5 000**. Stage 6 has chunked at most **5 000 documents** in a pre-checkpoint timing run at `4c72639` (§U.1), and at `f9c23fe` **crashed at document 0** on runner wiring (§W). |
| **What has NOT run** | **No completed Stage-6 prepare. No Stage-6 checkpoint interval ever committed. No real Drive resume. No encoder load, no forward pass, no optimizer, no training. No downstream task.** Official UIT-VSFC TEST **SEALED** and structurally unreachable. Local `.venv` remains **ML-free**; nothing was downloaded here. Compiled proposal PDF **STALE**. |
| **Next step** | The real Drive **START → checkpoint → kill → RESUME** probe. Not the full corpus run, and not training — **neither is authorised by this audit**. |
| **NOT** | **This is not the PRE-TRAIN audit.** That happens after a completed real prepare, a demonstrated real Drive resume, the proposal/PDF synchronisation, and a no-update real-model smoke available for review |
| **Revision 3c runner-wiring repair** | **2026-08-23** — the real Colab probe crashed at `[6/6]`: `prepare-corpus` read `args.repository_head`, a flag it never defined. HEAD is now derived from the executing tree; a second latent crash (`RAW_BASE_POLICY` unimported) was found by the new real-parser end-to-end test. See §W. |
| **Revision 3c hardening** | **2026-08-23** — pre-commit review: the direct-BPE fast path **bypassed the wrapper's added-token split** and would have miscounted any run containing e.g. `<mask>`. Removed. Composition additionally gated on the tokenizer's own added tokens. Verdict/metadata consistency repaired. See §V. |
| **Revision 3c** | **2026-08-23** — durable cross-runtime Stage-6 resume (append-only shards, document-boundary commits, failure-atomic state, identity-bound), plus a **second blocker found in inspection**: the pre-3c writer accumulated ~29.9 GB of chunks in RAM. Streaming removes it; output byte-identical. See §U. |
| **Revision 3b** | **2026-08-23** — forensics: the historical `composed 5, exact 7` was **the wrong run unit** (`\S+` instead of the tokenizer's `\S+\n?`), reproduced exactly. Revision 3a's "composition falsified" reading is withdrawn. Exact run composition restored; BPE work now scales with distinct runs. See §T. |
| **Revision 3a** | **2026-08-23** — **Revision 3's own micro-probe FAILED on the real pinned tokenizer** (`composed 5, exact 7`, rc 1). The runtime verifier worked and is kept. Per-run token composition removed; the 956x claim is withdrawn; 192x remains and is exact by construction. See §S. |
| **Revision 3** | **2026-08-23** — **third post-commit real-corpus defect**: stage-6 pre-chunking re-canonicalised every growing prefix (~250x document length) and produced no 1 % line in 13 minutes. Repaired by memoised, composable transforms and per-run token composition; `chunking.py` untouched. See §R. |
| **Revision 2** | **2026-08-23** — **second post-commit real-corpus defect**: the stage-4 contamination screen applied full `canon()` to all 1 118 224 documents and ran >7 h. Repaired with two proven necessary-condition prefilters; the criterion is unchanged. See §Q. |
| **Revision 1a** | **2026-08-22** — evidence-accuracy cleanup. Audit-only: §L/§M labelled historical, three overstated §P claims corrected, and the uncaptured original exception recorded. No code, test or decision changed. See §P.11. |
| **Revision 1** | **2026-08-22** — **post-commit real-corpus defect and repair**, revised in place. The chunker could not prepare the locked corpus: "never split a syllable" had been implemented as "cut only at whitespace", and real UVW-2026 contains oversized **non-whitespace** units. See §P. |

---

## A. VERDICT — CURRENT

**REVISION 3C (2026-08-23): RUNNER-WIRING REPAIR PASS — READY FOR REAL DRIVE RESUME PROBE**

Stage 6 is streamed and durably resumable; the direct-BPE fast path is
**removed** as unsafe (§V) and the real tokenizer probe confirms it; the
Stage-6 runner-wiring crash found by the real probe is repaired (§W). Every
prior verdict below is **HISTORICAL** and superseded.

**What has actually happened on real data:** the corpus pin, schema,
contamination screen and document split all pass on all **1 118 224** documents,
and the pinned-tokenizer probe passes. Stage 6 has been **entered twice**, and
the two events must not be conflated:

* at `4c72639`, **before** checkpointing existed, a timing run chunked **5 000
  documents** at ~29.5 docs/s (§U.1) — a measurement, not a prepared artifact;
* at `f9c23fe`, **with** checkpointing, it **crashed at document 0** on runner
  wiring before any heartbeat (§W).

So **no checkpoint interval has ever been committed**, and Stage 6 has **never
completed**.

**What remains true:**

* **no successful full Stage-6 prepare** — and this audit does not authorise one;
* **no committed Stage-6 checkpoint, and no real Drive resume demonstrated**;
* **no encoder training, no optimizer step, no forward pass**;
* **no downstream scientific Stage-1 run**;
* **official UIT-VSFC TEST sealed** and structurally unreachable;
* **compiled proposal PDF STALE**.

**The next step is the real Drive START → checkpoint → kill → RESUME probe** —
a bounded experiment that must reach at least one committed 5 000-document
checkpoint, survive a deliberate runtime kill, and resume at the committed
prefix. It is **not** the full 1 118 224-document run, and it is **not**
training. Neither is authorised here.

---

## A1. VERDICT — HISTORICAL (Revision 1)

> **SUPERSEDED.** Kept as the record of what was concluded then.

**REVISION 1 (2026-08-22): REPAIR PASS — READY FOR REAL CORPUS RE-RUN**

The implementation below passed every local and torch-gated test, was committed
as `0a34083`, and then **failed against the real corpus**: the chunker treated a
maximal non-whitespace unit as orthographically atomic, and UVW-2026 contains
such units up to **1 707 tokens**. Repaired in §P. **PRE-TRAIN remains BLOCKED**
until real `prepare-corpus` succeeds in Colab and the real-PhoBERT no-update
smoke passes.

---

## A2. VERDICT — HISTORICAL (as first written, pre-commit)

> **SUPERSEDED.** Written before commit `0a34083`, when nothing had executed.

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

### S.3 Why the cause was not guessed *(CORRECTED — see §T.2)*

> **SUPERSEDED.** The cause **was** subsequently isolated: bb50823 composed over
> `\S+` while the pinned tokenizer decomposes with `\S+\n?`. The reasoning below
> was correct not to guess, but its conclusion — that per-run composition is
> falsified — is **withdrawn**. §T.2 reproduces `composed 5, exact 7` exactly.


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

**The per-run token-composition shortcut is removed.** *(Revision 3b restored
it over the correct run unit — see §T.3.)* Correctness over benchmark: the real
evidence appeared to falsify it, it cannot be validated here, and
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
official-TEST policy.

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

## T. REVISION 3B — THE 5-vs-7 FORENSICS, AND EXACT RUN COMPOSITION

**Date:** 2026-08-23 **Baseline commit:** `2e12967a3308a9f686c70d841b9c4278d99aee5a`

### T.1 New real evidence

Revision 3a's probe **passed** on the real pinned tokenizer, and the full corpus
was attempted again.

```
Transformers 4.57.6   PhobertTokenizer
vinai/phobert-base @ 01daacda68afe13d83023d16ec647239e344a1e6
special tokens = 2
```

**Stages 1-5 still PASS.** **Stage 6 is a confirmed blocker:**

| Progress | Elapsed | Chunks |
|---|---|---|
| document 1 | 0.4 s | — |
| document 10 | 24.6 s | 280 |
| document 50 | 93.2 s | 1 047 |
| document 100 | 219.7 s | 2 489 |

≈ **0.45 documents/s**, i.e. **weeks** for 1 118 224 documents. Revision 3a
removed the composition and therefore still ran the authoritative whole-string
tokenizer on every growing prefix.

**The pinned tokenizer's own decomposition**, from `PhobertTokenizer._tokenize`:
it collects `re.findall(r"\S+\n?", text)` and calls `bpe` on each run
independently. **The trailing newline is part of the run.**

Real UVW sample — 64 documents, 1 920 reference/base slice cases, no corpus text
printed:

| Composition | Failures |
|---|---|
| wrapper `tokenize` vs `_tokenize` | **0** |
| naive `\S+` | **1 708** |
| exact `\S+\n?` | **0** |

Real-sample timing over 400 cases: authoritative wrapper 1.709 s; whole
`_tokenize` 0.310 s (5.52x); naive `\S+\n?` recompose 0.794 s (2.15x). **That
last benchmark recomputed runs per query and is *not* the Revision-3b
algorithm.**

### T.2 Forensics — what `composed 5, exact 7` actually was (Task A)

Revision 3a concluded the pinned tokenizer had *falsified* per-run composition,
and could not isolate a cause. **That conclusion was wrong, and is corrected
here.**

Read-only inspection of `bb50823:unmark/stage1/lengths.py` shows its run unit:

```python
_NON_WHITESPACE = re.compile(r"\S+")        # bb50823
```

against the tokenizer's own `\S+\n?`. The historical probe's document list
begins `"short"`, `"vietnamese"`, `"whitespace"`, and the third one is
`'alpha  beta\tgamma\n\ndelta   '`. Replaying every prefix the chunker queries,
with a tokenizer double faithful to `_tokenize` (runs `\S+\n?`, BPE end-of-word
marker on the run's last character), the first mismatch is:

```
piece            'alpha  beta\tgamma\n\n'      pathway: reference
bb50823 runs     ['alpha', 'beta', 'gamma']     -> composed 5
PhoBERT runs     ['alpha', 'beta', 'gamma\n']   -> exact    7
exact \S+\n?    composition                    -> 7  (matches)
```

**`composed 5, exact 7` is reproduced exactly.** The cause is neither
special-token accounting, nor a transform difference, nor stale cache, nor
reference/base confusion: it is the **wrong run unit**. `"gamma\n"` does not BPE
like `"gamma"`, because the end-of-word marker lands on the newline.

This also explains why the new diagnostic reports a clean length 5 for
`"Tôi đã đọc"`: that string contains no newline, so `\S+` and `\S+\n?` agree on
it. **It was never the failing fixture** — the historical traceback showed only
`chunk_document`, and Revision 3a inferred the first document. That inference is
withdrawn.

**Honest scope of this reproduction.** The mechanism is confirmed by the real
sample (1 708 `\S+` failures vs 0 for `\S+\n?`). The specific 5/7 arithmetic is
reproduced with a faithful-shaped double, because the real tokenizer is not
available in this ML-free environment. The failed probe itself is preserved in
§S as historical evidence; only its *interpretation* is corrected.

### T.3 The Revision-3b algorithm (Task B)

`chunking.py` remains **untouched**: same greedy order, same `fits` truth values,
same boundaries.

| Layer | Behaviour |
|---|---|
| Transforms | Revision-3a memoised, incrementally extended `canon` / `base_text` — unchanged |
| Run unit | **`PHOBERT_RUN = re.compile(r"\S+\n?")`** — the tokenizer's own, never `\S+` |
| Per-run cost | `len(tokenize(run))` memoised per **exact run string**, including any trailing newline |
| Specials | `authoritative_length("")` through the tokenizer API — **never a hard-coded `+2`** |
| Incremental | only the **final** run can change when text is appended (extended, or gaining `\n`), so the total is kept alongside the last run's start offset and only the tail is recomputed |
| Fallback | any query that does not extend the previous one recomputes fully; the oversized-unit path is unaffected |
| Verification | first 256 distinct queries per composer also computed through the authoritative chain; mismatch raises `Stage1ContractViolation` |

Correctness rests on a property of the **tokenizer's own implementation** —
it *is* a per-run composition — not on an assumption about BPE. No monotonicity
is used: the composition is over disjoint runs of one string.

### T.4 Measured effect (Tasks G, H)

12 documents, 78 351 characters, PhoBERT-shaped double:

| Implementation | Time | Throughput | Speedup |
|---|---|---|---|
| old authoritative | 94.74 s | 0.8 K doc-chars/s | 1.0x |
| Revision 3a | 4.71 s | 16.6 K | 20.1x |
| **Revision 3b** | **0.39 s** | **202.5 K** | **244.9x** |

**Output identical to the old implementation for both 3a and 3b** — ids, source
ranges, text, and both recorded lengths.

The algorithmic claim, which matters more than the wall-clock number:

| Counter | Value |
|---|---|
| length queries | 72 216 |
| **authoritative whole-string calls** | **514** — bounded by the verification window |
| **BPE run evaluations** | **34** |
| run-cache hits / misses | 215 798 / 34 |
| incremental appends / full fallbacks | 71 880 / 336 |
| characters canonicalised | 58 of 78 351 |

Expensive tokenizer work now scales with **distinct runs**, not with the sum of
growing prefix lengths.

### T.5 Tests (Tasks C, D, E, F, L)

| Suite | Result |
|---|---|
| `tests/test_stage1_lengths.py` | **299 passed** (was 223) |
| `tests/test_stage1_chunking.py` | 35 passed (unchanged) |
| `tests/test_stage1_runner_contract.py` | 43 passed |
| `tests/test_stage1_contamination_prefilter.py` | 287 passed (Revision 2 untouched) |
| Full repository | **3 125 passed, 97 skipped** |

New in Revision 3b:

* **Newline attachment** — 16 fixtures (`"Tôi\nđã\nđọc"`, `"Tôi đã đọc\n"`,
  `"Tôi đã\nđọc"`, tabs, multiple spaces, CR/LF, leading/trailing, empty, bare
  newlines) asserting **token LIST** equality, not merely counts, and a test that
  naive `\S+` **demonstrably fails** on them.
* **The historical fixture** — `test_the_historical_five_versus_seven_is_reproduced_and_repaired`
  asserts `(naive, exact) == (5, 7)` on `'alpha  beta\tgamma\n\n'` and that the
  repaired composer returns **7**.
* **Authoritative equality** on both pathways across the fixture set.
* **Specials via the API** — a tokenizer with *three* special tokens still
  matches, so no `+2` is baked in.
* **A run gaining a newline mid-growth** — every prefix of the historical
  document checked one character at a time.
* **Fail-closed** — a deliberately corrupted run counter raises; a
  non-conforming tokenizer raises.
* Three Revision-3a tests that asserted the *absence* of composition were
  **replaced**, not loosened: with 3b a non-conforming tokenizer must now fail
  closed rather than be harmless.

The chunk-output oracle keeps the pre-optimisation implementation in test code
only and still compares every field including violation provenance, across
non-monotonic and newline-sensitive tokenizers, path asymmetry, oversized
fallback, safe interior cuts, whitespace forms and randomised documents.

### T.6 Probe (Task I)

`scripts/stage1_tokenizer_probe.py` now also checks `tokenize == _tokenize`,
exact `\S+\n?` composition, and counts how many naive `\S+` cases fail (expected
non-zero — recorded as evidence, not as a failure). `--help` remains
side-effect free and exits 0 without importing transformers. No scientific
override flags. No encoder, forward pass, optimizer or training.

### T.7 Limitations

1. **Revision 3b has not been run on the real tokenizer.** Every number here is
   from a faithful-shaped double.
2. **The real corpus has not been chunked.** Stage 6 has still never completed;
   no real Stage-6 PASS is claimed.
3. The 244.9x figure is synthetic and machine-specific. **The algorithmic
   counters are the claim**; the wall-clock is illustrative.
4. The 5/7 reproduction uses a double, not the real tokenizer (§T.2).
5. Real BPE cost per distinct run, and the real run vocabulary size against the
   500 000-entry memo ceiling, are unmeasured.
6. Whether ~0.45 docs/s becomes acceptable on real data is **unknown** until
   Colab runs it.

### T.8 Self-audit for Revision 3b

| # | Check | Result |
|---|---|---|
| 1 | Historical 5-vs-7 cause identified, not guessed | **yes** — reproduced exactly, §T.2 |
| 2 | `\S+` nowhere used as the composition contract | **yes** — `PHOBERT_RUN` is `\S+\n?`; naive form appears only in tests as a counter-example |
| 3 | Exact newline behaviour covered | **yes** — 16 fixtures, token-list equality |
| 4 | Special tokens from the authoritative API | **yes** — three-special tokenizer test |
| 5 | Reference path exact | **yes** |
| 6 | RAW_BASE path exact | **yes** |
| 7 | No monotonicity assumption | **yes** — non-monotonic tokenizer still passes |
| 8 | Runtime verifier fails closed | **yes** — corrupted counter and non-conforming tokenizer both raise |
| 9 | Old-vs-new chunk oracle passes | **yes** — all fields incl. provenance |
| 10 | Fallback semantics unchanged | **yes** |
| 11 | `chunking.py` semantics unchanged | **yes** — not modified |
| 12 | Revision-2 contamination code untouched | **yes** |
| 13 | Official UIT-VSFC TEST sealed | **yes** |
| 14 | No training, no optimizer step | **yes** |
| 15 | Focused + full suites pass | **yes** — 3 125 passed |
| 16 | Revision 3a interpretation corrected at its point of origin | **yes** — §S annotated |
| 17 | Failed probe preserved as historical evidence | **yes** — §S intact |
| 18 | Real Stage 6 claimed PASS? | **NO** |
| 19 | PRE-TRAIN claimed ready? | **NO** |
| 20 | Nothing staged; no prohibited git operation | **yes** |

---

## U. REVISION 3C — DURABLE STAGE-6 RESUME, AND A 30 GB MEMORY BLOCKER

**Date:** 2026-08-23 **Baseline commit:** `4c72639a3215c0c5c73b0408f2088cf11a110287`

### U.1 Where Revision 3b left it

The real pinned-tokenizer probe **PASSed**: `status PASS`, `failures []`,
`run_unit "\S+\n?"`, `encoder_loaded false`, `forward_passes 0`,
`optimizer_steps 0`. Stages 1-5 continue to pass on all 1 118 224 documents
(contamination: 0 excluded, 296 628 length-guard skips, 821 596 prefilter
checks, 0 candidates, 0 corpus canon calls; split 1 113 224 / 5 000; official
UIT-VSFC TEST **SEALED**).

Real 5 k Stage-6 timing:

| Documents | Elapsed |
|---|---|
| 1 | 0.5 s |
| 100 | 9.3 s |
| 1 000 | 53.1 s |
| 5 000 | 188.9 s |

**26.47 docs/s overall, 29.46 docs/s warm (1 000 → 5 000)**, projecting
**~10.55 hours** for the full corpus — a **~65.5x** improvement on Revision 3a.
Feasible, but a Colab runtime death would restart it at document 0.

### U.2 Task A — what the pre-3c writer actually did

| # | Question | Finding |
|---|---|---|
| 1 | Where are the 1 118 224 documents held? | Fully materialised in RAM: `documents`, then `kept` |
| 2 | Are all `PreparedChunk`s accumulated? | **Yes** — `chunks = chunk_corpus(...)` builds one list of every chunk |
| 3 | When is the payload written? | **Only at the very end**, after all chunking |
| 4 | Format | JSONL, one object per line: `chunk_id, document_id, partition, chunk_index, text, source_start, source_end, source_shard`, `ensure_ascii=False` |
| 5 | Manifest | `build_manifest(...)` → `manifest.json`, `indent=2, ensure_ascii=False, sort_keys=True` |
| 6 | Does any output hash depend on serialization order? | **No.** `chunk_membership_digest` **sorts** its keys, and every count is order-free. The JSONL line order is document order |
| 7 | Needed after chunking | every chunk (counts, digest, payload), plus `source`, `contamination`, `partition` |
| 8 | Cross-document scientific state in Stage 6 | **None.** `chunk_document` is document-local; the partition is decided before chunking |
| 9 | `ComposedTransforms` / `RunLengthComposer` caches | **Performance-only** — memoisation; discarding them changes no output |
| 10 | Safe to discard at a runtime boundary | all caches, all already-committed chunks, and `documents`/`kept` (Stages 1-5 rebuild them in seconds) |

**And the finding that changed the scope of this revision.** Point 2 is not just
an inefficiency. From the real run, 100 documents produced 2 489 chunks — 24.89
chunks/document. For 1 118 224 documents that is **≈ 27.8 million chunks**, and
a `PreparedChunk` with an ~800-character text costs ~1 075 bytes:

```
27 832 595 chunks  x  ~1 075 B  =  ~29.9 GB
```

held in Python RAM *before the first byte is written*, on top of the corpus
itself. **The pre-3c writer would very likely have exhausted Colab memory hours
into the 10.55-hour run**, having written nothing. This is a second, independent
blocker, and it makes streaming mandatory rather than an optimisation.

### U.3 The repair — streaming, append-only, failure-atomic

**No scientific value changes.** Chunk boundaries, ids, ranges, both lengths,
text, ordering, partitioning and every manifest scientific field are identical;
`chunking.py`, `corpus.py`, `canon()` and the protocol are untouched.

*Streaming.* Chunks are serialised to a shard buffer as each document completes,
so peak memory is bounded by **one shard**, not by the corpus. Measured with a
fixed interval while growing the corpus 9x:

| Documents | Chunks | RSS delta |
|---|---|---|
| 1 000 | 3 102 | 2.0 MB |
| 3 000 | 9 349 | 1.2 MB |
| 9 000 | 27 956 | 0.5 MB |

Flat, as required. At the 5 000-document interval the buffer is ~100 MB against
the pre-3c ~29.9 GB.

*Streaming manifest.* `build_manifest_from_counts` assembles the identical
manifest from counts accumulated during the stream; `build_manifest` is retained
unchanged and the two are asserted equal. The membership digest sorts its keys,
so it is computed by an **external merge sort** (blocks sorted, spilled, merged)
— bounded memory, and verified to produce the *same digest* as the in-memory
version even with a 7-line block size forcing many spills.

*Append-only immutable shards.* Each shard covers one contiguous range of source
document indices; a committed shard is never rewritten, so checkpoint cost is
O(new work), never O(progress).

*Commit protocol.* Payload: temp → flush → `fsync` → close → sha256 → `replace`
→ **re-read and re-verify size and digest from disk** → only then state. State:
temp → flush → `fsync` → `replace`. Drive's FUSE layer is not assumed POSIX, so
nothing is trusted on the strength of a successful `write()`. A death between
payload and state leaves an unreferenced shard that resume simply ignores.

*Document boundaries only.* A commit means every document below
`next_document_index` is **completely** processed; a document is never split
across a commit. The interval is **5 000 documents** — at 29.46 docs/s that
bounds lost work at ~3 minutes. **Operational, not scientific**: any interval
produces the same artifacts.

### U.4 Identity binding and the state machine

The checkpoint binds: schema version, repository HEAD, protocol and chunk-schema
versions, corpus dataset/revision and all three pinned file identities,
tokenizer checkpoint/revision, **Transformers version**, `max_length`,
`RAW_BASE` policy, split seed, dev count, contamination method and excluded
count, the **ordered document-sequence digest**, the **partition-assignment
digest**, `next_document_index`, completed-document and chunk counts (total,
train, dev), the shard list with per-shard byte size and sha256, and the
**last completed document id** so an off-by-one resume is detectable.

**No raw corpus text and no UIT-VSFC text** appear in checkpoint metadata —
asserted by test.

| State | Trigger |
|---|---|
| **START** | no `state.json` |
| **RESUME** | valid `state.json`: Stages 1-5 are re-run from the pinned inputs, every identity field must match, every committed shard's size and digest must still verify, the shard ranges must form a contiguous prefix, and only then does work continue at `next_document_index` |
| **ALREADY_COMPLETE** | `COMPLETE.json` validates **and** every artifact it names re-hashes correctly |

Stages 1-5 are deliberately **not** checkpointed — they take seconds and are
scientifically load-bearing. They are re-derived and compared; any difference in
corpus order, contamination, split or protocol **fails closed** rather than
resuming against a different stream.

`COMPLETE.json` is written **last**, after every artifact is on disk and
verified, and binds their hashes plus the run identity. A directory existing
proves nothing.

### U.5 Interruption evidence (Task G)

`tests/test_stage1_checkpoint.py` — **41 tests**. Death is simulated at eight
document positions spanning every structural point (before the first commit,
exactly on a commit, mid-uncommitted-shard, several commits in, and at the last
document); each is resumed and compared against an uninterrupted oracle.

| Property | Result |
|---|---|
| Resumed payload **byte-identical** to uninterrupted | pass, all positions |
| Resumed manifest byte-identical | pass |
| Streamed payload == the in-memory writer's bytes | pass |
| Streamed manifest == `build_manifest`'s dict | pass |
| Resume starts exactly at the committed prefix | pass |
| No duplicate / missing document; order preserved | pass |
| No duplicate chunk id | pass |
| Mid-shard death loses only the uncommitted shard | pass |
| Orphan `.tmp` (shard **and** state) ignored and removed | pass |
| Unreferenced shard on disk never used | pass |
| Tampered shard → fails closed | pass |
| Malformed state → refused, not guessed | pass |
| Out-of-order document → refused | pass |
| **13 identity fields**, each mutated → fails closed | pass |
| Directory alone ≠ ALREADY_COMPLETE | pass |
| Completion refused if an artifact changed, or identity is foreign | pass |
| Finalisation idempotent | pass |
| Checkpoint metadata carries no corpus text | pass |

### U.6 Performance work considered (Task I)

**~~Accepted~~ REMOVED — direct pinned BPE (I.1).** This was accepted in the
first pass of Revision 3c and is **withdrawn by the 3c hardening**: the wrapper
overhead it skipped *is* the added-token split, and skipping that is not an
optimisation but a defect. See **§V**. `RunLengthComposer` calls
`tokenizer.tokenize(run)`, exactly as Revision 3b did.

**Accepted — cache instrumentation (I.2).** `run_cache_evictions`,
`run_cache_entries`, `run_cache_max_entries` added. On a 240-document
real-shaped workload: **34 BPE evaluations, 3 452 846 cache hits, 0 evictions**.
The 500 000-entry ceiling is **not changed** — there is no evidence it is
costing work, and a cap change without that evidence would be guessing. Caches
live in **CPU** memory; the 90 GB of GPU VRAM is irrelevant to them.

**Rejected for now — document-level multiprocessing (I.3).** Benchmarked
honestly on a real-shaped workload, output verified identical:

| Workers | Throughput | Speedup | Identical | BPE evals |
|---|---|---|---|---|
| 1 | 52.0 docs/s | 1.00x | — | 34 |
| 2 | 92.3 docs/s | **1.78x** | **yes** | 68 (2.0x) |
| 4 | 188.3 docs/s | **3.62x** | **yes** | 136 (4.0x) |

It works and is exact, and BPE-evaluation duplication scales with workers
exactly as warned — negligible here only because the hit rate is ~100%. It is
**not adopted in this revision** because: the catastrophic risk was runtime
death, which checkpointing now bounds at ~3 minutes; the actual hard blocker was
the ~30 GB accumulation, now fixed; multiprocessing complicates the
contiguous-committed-prefix invariant the durability guarantee rests on; and it
cannot be validated against the real tokenizer here. This machine has 16 CPUs
while Colab standard runtimes typically expose 2, where the measured gain is
1.78x — material, but not a qualitative change to a now-survivable 10.55-hour
run. The measurement is recorded so the decision can be revisited with evidence
rather than repeated from scratch.

**Rejected outright.** GPU/CUDA tokenization, a fast tokenizer, approximate
counts, any change to `max_length`, truncation, dropping, normalisation or
`RAW_BASE`. Stage 6 is CPU tokenizer and Python work; **GPU VRAM does not
accelerate it**, and this audit does not pretend otherwise.

**Checkpoint overhead (Task J).** Measured on 3 000 documents / 9 349 chunks
with a deliberately aggressive 500-document interval: **6 commits, 3.3 MB,
0.02 s = 1.65 % of Stage-6 time**; finalisation 0.04 s. At the real
5 000-document interval commits are ~10x rarer. Shards accumulate on **local
staging** and only completed, verified shards are copied to the checkpoint
directory, so a Drive mount never sees one write per chunk.

### U.7 Operational interface (Tasks K, L)

`prepare-corpus` gains exactly one operational flag, `--checkpoint-dir`
(default `<output-dir>/_checkpoint`). Resume is **automatic**: a valid
checkpoint continues, a verified completion marker short-circuits. **No
scientific override was added** — no `max_length`, seed, split, dev count,
`pi_strip`, corruption, tokenizer or corpus revision flag exists.

The immutable-output guard now permits reuse **only** when a checkpoint state or
completion marker is present, so a stale directory still cannot be silently
overwritten. Before the long run the runner prints local and checkpoint free
disk and process RSS, and reports checkpoint commits, bytes, time and share of
Stage-6, plus the length/BPE/cache/fallback counters.

### U.8 Tests

| Suite | Result |
|---|---|
| `tests/test_stage1_checkpoint.py` | **41 passed** (new) |
| `tests/test_stage1_lengths.py` | 299 passed |
| `tests/test_stage1_chunking.py` | 35 passed (unchanged) |
| `tests/test_stage1_runner_contract.py` | 43 passed |
| `tests/test_stage1_contamination_prefilter.py` | 287 passed (Revision 2 untouched) |
| Full repository | **3 166 passed, 97 skipped** (was 3 125 / 97) |

### U.9 Limitations

1. **The full 1.118 M-document run has NOT been performed.** No Stage-6 PASS is
   claimed, and this audit does not authorise the full run.
2. **Checkpointing has never been exercised on the real corpus or a real
   Drive mount.** Every durability test uses a local filesystem; Drive's FUSE
   behaviour under a real runtime death is untested here.
3. **The direct-BPE path has not run against the real tokenizer.** It
   self-verifies at startup and degrades to the wrapper, but its actual benefit
   is unmeasured.
4. The ~30 GB figure is an extrapolation from the real 24.89 chunks/document and
   a measured per-object cost, not an observed OOM.
5. Multiprocessing is measured but unadopted; the 1.78x at two cores is from a
   double, not the real tokenizer.
6. The 5 000-document interval is justified from the *reported* throughput; real
   checkpoint overhead at that interval on Drive is unmeasured.
7. Stage 6 has still never completed, so Revision 1's chunker remains unverified
   on real data end to end.

### U.10 Self-audit for Revision 3c

| # | Check | Result |
|---|---|---|
| 1 | Audit 029 revised in place; no Audit 030 | **yes** |
| 2 | Scientific outputs unchanged | **yes** — payload bytes and manifest asserted identical to the in-memory writer |
| 3 | Resume starts only at document boundaries | **yes** — tested |
| 4 | Every commit is a contiguous prefix | **yes** — enforced and tested |
| 5 | Shards immutable once committed | **yes** — never reopened |
| 6 | Shard hashes verified on resume | **yes** — tampering fails closed |
| 7 | State write failure-atomic | **yes** — temp/fsync/replace, payload before state |
| 8 | Orphan temps cannot be accepted | **yes** — removed on begin, tested |
| 9-14 | HEAD / corpus pin / tokenizer revision / Transformers / protocol / split-order mismatch all fail | **yes** — 13 parametrised identity cases |
| 15-17 | No skipped, duplicated document or chunk | **yes** — tested after resume |
| 18 | Resumed output == uninterrupted output | **yes** — byte-identical, 8 death positions |
| 19 | Partial directory cannot masquerade as COMPLETE | **yes** |
| 20 | COMPLETE marker written last | **yes** — after artifacts are verified |
| 21 | Finalisation idempotent | **yes** — tested |
| 22 | BPE caches performance-only | **yes** — never serialised, cold resume is correct |
| 23 | GPU not claimed to accelerate CPU tokenization | **yes** — stated explicitly |
| 24 | Direct-BPE path exact and verified | **NO — the 3c hardening found it unsafe and REMOVED it (§V)** |
| 25-26 | Multiprocessing exactness / failure propagation | **measured identical, but NOT adopted** (§U.6) |
| 27 | Checkpoint overhead measured | **yes** — 1.65 % at an aggressive interval |
| 28 | Memory/disk bounded | **yes** — flat RSS across a 9x corpus growth |
| 29 | `chunking.py` semantics unchanged | **yes** — not modified |
| 30 | Contamination semantics unchanged | **yes** — `corpus.py` not modified |
| 31 | Official UIT-VSFC TEST sealed | **yes** |
| 32-35 | No encoder, forward, optimizer, training | **yes** |
| 36-37 | Focused and full suites pass | **yes** — 3 166 passed |
| 38-39 | `git diff --check` clean; nothing staged | **yes** |
| 40 | Full corpus claimed PASS? PRE-TRAIN ready? | **NO** to both |

---

## V. REVISION 3C HARDENING — THE DIRECT-BPE PATH WAS UNSAFE

**Date:** 2026-08-23 **Reviewing:** the unstaged Revision-3c working tree over
baseline `4c72639a3215c0c5c73b0408f2088cf11a110287`

Revision 3c's architecture is unchanged and accepted. One thing in it was
wrong, and this section records and removes it.

### V.1 Task A — what the wrapper actually does

`PreTrainedTokenizer.tokenize(text)` does **not** simply call `_tokenize`. It

1. runs `prepare_for_tokenization`, then
2. **splits the text on the added/special tokens** (`<s>`, `</s>`, `<unk>`,
   `<pad>`, `<mask>` and anything else in the added vocabulary), emitting those
   as whole tokens, and only then
3. passes the remaining segments to `_tokenize`, which applies
   `re.findall(r"\S+\n?", ...)` and `bpe` per run.

Step 2 is invisible to `tokenizer.bpe(run)`. So for any run **containing** an
added token as a substring, the two disagree — and PhoBERT's `_tokenize`
calling `bpe` internally proves nothing about the *public* pathway, which is
authoritative.

### V.2 The defect, demonstrated

The 17-run startup probe Revision 3c used to license the fast path contains **no
added token**, so it enabled direct BPE. On a tokenizer double faithful to the
wrapper's trie split:

| Run | direct `bpe` | wrapper `tokenize` | |
|---|---|---|---|
| `abc<mask>def` | 4 | **3** | wrong |
| `<mask>` | 2 | **1** | wrong |
| `x</s>y` | 2 | **3** | wrong |

A single such run — at document 900 000, long past the 256-query verification
window — would have changed `fits()` and therefore **chunk boundaries**, silently.
The user's review caught this before any real run.

### V.3 Task B — removed, not patched

A safe predicate would have to scan every run against every added token, which
costs about what the wrapper call it replaces costs. And the benefit was never
measured: the run-count memo means the tokenizer is consulted **once per distinct
run**, not per query (34 evaluations against 3 452 846 cache hits in the §U.6
benchmark), so the ceiling on the saving is small and entirely unquantified on
the real tokenizer. Revision 3b already delivered the feasible ~10.55 h **without
it**.

**The optimisation is removed.** `lengths.py` contains no call to
`tokenizer.bpe` — AST-asserted in test.

### V.4 A second, pre-existing hazard this exposed

The same inspection revealed a latent risk in Revision 3b's *composition itself*,
independent of direct BPE. The wrapper matches added tokens as literal
substrings **anywhere**, including across whitespace. An added token containing
whitespace would therefore be lifted out whole by `tokenize`, while per-run
composition would split it in two:

| Text | wrapper | composed | |
|---|---|---|---|
| `abc [NEW LINE] def` | 3 | 6 | **mismatch** |
| `[NEW LINE]` | 1 | 4 | **mismatch** |

No PhoBERT special token contains whitespace, so this was never live — but it
was never *checked*. `RunLengthComposer` now reads the tokenizer's **own**
added-token collection (`get_added_vocab`, `all_special_tokens`,
`added_tokens_encoder` — never hard-coded strings) and **disables composition
entirely**, falling back to the authoritative whole-string chain, if any token
contains whitespace **or if the collection cannot be read at all**. False
negatives only cost speed; a false positive would change scientific output, so
unknown is treated as unsafe.

Composition remains enabled for a PhoBERT-like special-token set, and runs
*containing* those tokens stay exact — because each run is counted with the
**wrapper**, so the trie split happens inside the run.

### V.5 Tests (Tasks C, D, E)

Added to `tests/test_stage1_lengths.py` (**325 passed**, was 299):

| # | Case | Result |
|---|---|---|
| 1 | Ordinary run stays exact | pass |
| 2 | Exact added token (`<mask>`, `<s>`, …) | exact via wrapper |
| 3 | Added token embedded in a larger run (`abc<mask>def`) | exact |
| 4 | Adjacent / multiple added tokens | exact |
| 5 | Newline-sensitive runs (Revision 3b) | unchanged |
| 6 | Direct `bpe` disagreement is *demonstrated*, and `lengths.py` proven not to call `bpe` | pass |
| 7 | Tokenizer that cannot report added tokens | composition **disabled** |
| 8 | **Wrapper-sensitive run after 600 queries**, past the 256-query window | exact |
| 9 | Reference **and** RAW_BASE pathways | both exact |
| 10 | Chunk output vs authoritative oracle under a wrapper tokenizer | identical |

20 added-token fixtures cover the token alone, prefixed, suffixed, both,
adjacent, and with newlines/tabs/spaces around it, on both pathways.

The Colab probe now builds its fixtures from the **tokenizer's own** added
tokens and reports `direct_bpe_enabled` (false), `composition_enabled`,
`added_tokens`, `direct_bpe_safe_cases`, `direct_bpe_wrapper_fallback_cases`
and `direct_bpe_mismatches`. `--help` remains side-effect free; no encoder,
forward, optimizer or training — AST-asserted.

### V.6 Task G — `protocol.py` claim corrected

Revision 3c's §U.3 said the protocol was untouched. Precisely:

* **`unmark/stage1/protocol.py` is modified: `+6 / −0`**, adding
  `RAW_BASE_POLICY = "RAW_BASE"` so the checkpoint can bind the base-pathway
  identity and refuse to resume a stream prepared under a different one. Purely
  additive; no existing line changed.
* **Scientific constants are unchanged** — **21** of them value-compared
  against **`4c72639`**: `MAX_LENGTH`, `PI_STRIP`, `DEV_DOCUMENTS`, `SPLIT_SEED`,
  `SELECTION_SEED`, `TRAIN_SEEDS`, `CORRUPTION_SEED`,
  `VALIDATION_CORRUPTION_SEED`, `LR_PILOT_GRID`, `R_PHASE1_GRID`, `BATCH_SIZE`,
  `CORPUS_REVISION`, `ENCODER_REVISION`, `ON_OVERFLOW`, `PRECISION`,
  `VALIDATION_CONDITIONS`, `ADAPTER_TRAINABLE_PARAMETERS`,
  `CONTAMINATION_METHOD`, `CORPUS_SHARD_ORDER`, `STAGE1_PROTOCOL_VERSION`,
  `CHUNK_SCHEMA_VERSION` — **NONE changed**. The group labels used in the first
  draft of this line ("all seeds", "both grids") are expanded above so the count
  is checkable; it is the **same set**, and it is **21** names, not 22.
* **`RAW_BASE_POLICY` is deliberately NOT in that 21.** It is the constant this
  revision **added**, so it has no `4c72639` value to be unchanged against.
  Counting it would turn a newly-introduced identity constant into evidence of
  stability, which is the opposite of what the comparison is for. See §X.1.

"scientific protocol unchanged" and "`protocol.py` byte-untouched" are different
claims; only the first is true.

### V.7 Task I — checkpoint code untouched

No correctness bug was found in the checkpoint architecture, so **none of it was
modified**. All 41 checkpoint/resume tests still pass unchanged.

### V.8 Limitations

1. The wrapper's behaviour is modelled here with a faithful double; **the real
   Transformers 4.57.6 `tokenize` was not executed** in this environment. The
   probe exercises it on the next Colab run.
2. `prepare_for_tokenization` is not separately proven to be identity for the
   pinned tokenizer. It is not relied on: every run goes through the public
   wrapper, which applies it.
3. The removed optimisation's real benefit was never measured, so "negligible"
   is an argument from the memo-hit counters, not a timing.
4. Everything in §U.9 still stands: the full 1.118 M run has not been performed,
   and checkpointing has not been exercised on a real Drive mount.

### V.9 Self-audit for the 3c hardening

| # | Check | Result |
|---|---|---|
| 1 | Direct BPE cannot bypass added/special-token semantics | **yes — removed entirely** |
| 2 | Wrapper-sensitive run after query 256 still exact | **yes** — 600-query test |
| 3 | Safe cases match the authoritative wrapper | **yes** — 20 fixtures x 2 pathways |
| 4 | Unsafe cases deterministically fall back | **yes** — composition disabled |
| 5 | Exact safety unprovable → optimisation removed | **yes** |
| 6 | Reference path exact | **yes** |
| 7 | RAW_BASE path exact | **yes** |
| 8 | Chunk-output oracle unchanged | **yes** |
| 9 | Revision-3c checkpoint/resume tests pass | **yes** — 41 |
| 10 | Streaming memory bound still holds | **yes** |
| 11 | Interrupted/resumed output byte-identical | **yes** — 8 death positions |
| 12 | No checkpoint scientific identity weakened | **yes** — 13 fields unchanged |
| 13 | Top-level verdict is current | **yes** — §A |
| 14 | Historical verdicts marked | **yes** — §A1, §A2 |
| 15 | `protocol.py` claims factually accurate | **yes** — §V.6 |
| 16 | Scientific constants unchanged | **yes** — **21** compared against `4c72639`, none changed; `RAW_BASE_POLICY` excluded as newly added |
| 17 | Decision log updated | **yes** — D-S1B-010 |
| 18 | Proposal update necessity assessed | **yes** — not required (§V.6, D-S1B-010) |
| 19-21 | `chunking.py`, contamination semantics, sealed TEST | **yes** |
| 22-25 | No encoder, forward, optimizer, training | **yes** |
| 26-27 | Focused and full suites pass | **yes** — 3 192 passed, 97 skipped |
| 28-29 | `git diff --check` clean; nothing staged | **yes** |

---

## W. REVISION 3C RUNNER-WIRING REPAIR — THE REAL PROBE CRASHED AT STAGE 6

**Date:** 2026-08-23 **Baseline commit:** `f9c23fedd4b5dd85b206454886a3e5bade3cfa86`

### W.1 What the real probe showed

Environment: Python 3.13.15, Transformers 4.57.6, HEAD `f9c23fe`.

**The hardened tokenizer probe PASSED** — `direct_bpe_enabled: false`,
`composition_enabled: true`, `added_tokens: 5`, `wrapper_fixtures: 58`,
`direct_bpe_mismatches: 0`, `failures: []`, `status: PASS`. No encoder, forward
pass, optimizer or training. §V's removal of the direct-BPE path is confirmed on
the real tokenizer.

**Stages 1-5 PASSED on the full pinned corpus**: 1 118 224 documents,
0 contaminated, 296 628 length-guard skips, 821 596 prefilter checks,
0 candidates, 0 corpus canon calls, split 1 113 224 / 5 000, official UIT-VSFC
TEST **SEALED**.

**Stage 6 then crashed immediately**, before any progress heartbeat and before
any checkpoint interval:

```
File "scripts/stage1_runner.py", line 189, in run_prepare_corpus
    repository_head=args.repository_head,
AttributeError: 'Namespace' object has no attribute 'repository_head'
```

Wrapper state at the time: `mode START`, `next_document_index 0`. **No
5 000-document checkpoint was reached, so nothing about Stage-6 throughput or
resume was measured.**

### W.2 Root cause, from the code

| # | Question | Finding |
|---|---|---|
| 1 | Why does `run_prepare_corpus` read `args.repository_head`? | Revision 3c added the `CheckpointIdentity` construction and copied the field from the pattern used by the *other* subcommands |
| 2 | Which subcommands define it? | `lr-pilot`, `r-phase1`, `final-main` (via `_corpus_consumer`) and `smoke`. **`prepare-corpus` has its own argument block and never defined it** |
| 3 | Is there an existing repository-identity helper? | **No.** `unmark/alignment/contracts.py` has an `observed_revision` *field* documented as "`git rev-parse HEAD` of the provisioned checkout", but it is a dataclass field, not a resolver |
| 4 | How do the other commands obtain HEAD? | From `--repository-head`, default `None` — **provenance the caller claims** |
| 5 | Why did the tests miss it? | Every Stage-1 runner test is **AST-only** (`build_parser` inspected as a tree), and the checkpoint tests construct `CheckpointIdentity` **objects directly**. **No test ever parsed a real `prepare-corpus` command line and entered `run_prepare_corpus`** |

Point 5 is the important one: a regression written as
`argparse.Namespace(repository_head=...)` would have *supplied the very attribute
whose absence was the bug*, reproducing the blind spot rather than closing it.

### W.3 The repair

`prepare-corpus` now derives the **actual** HEAD of the executing source tree:

```python
resolve_repository_head()   # git -C <repo root> rev-parse HEAD
```

* returns the **full 40-character** SHA, shape-validated;
* **fails closed** — a missing `git`, a non-zero exit, or a non-SHA result all
  raise `CheckpointViolation`. Never `"unknown"`, never a branch name, never a
  default, never a hard-coded commit;
* reads **no environment variable** — a caller-supplied value is not an identity;
* **no `--repository-head` flag was added** to `prepare-corpus`. The caller must
  not be able to claim a HEAD for checkpoint identity.

The flag on the three consuming commands and `smoke` is left as it was: this
repair fixes the crash, it does not open a new policy question about those.

**HEAD remains a mandatory fail-closed checkpoint identity field.** A checkpoint
written by commit A still cannot resume under commit B — proven by test.

### W.4 A second latent crash on the same line of code

The new end-to-end test immediately exposed a further defect the AST tests could
not see: `RAW_BASE_POLICY` was **used in the identity construction but never
imported**. It would have raised `NameError` on the very next real run, at the
same point, after the first fix. It is now imported, and a static check confirms
every name `run_prepare_corpus` loads is bound.

### W.5 Failed-probe artifact safety

Inspected what `run_prepare_corpus` creates before the crash line: the
`CheckpointIdentity` is constructed at line 190, while the first `mkdir` is at
line 273 and `checkpoint.begin()` at line 222 — both **after** it.

**The real `f9c23fe` failure therefore left nothing**: no output directory, no
checkpoint directory, no `state.json`, no shards, no `COMPLETE.json`. Asserted by
test.

**What the next Colab run should do with the `f9c23fe` probe directories:**
nothing needs deleting, because nothing was created. If a directory *does* exist
from some other attempt, the unchanged immutable-output contract applies — a
directory holding neither `state.json` nor `COMPLETE.json` is **refused**, and
**no user data is deleted automatically**. Both behaviours are now tested.

### W.6 The regression test that closes the blind spot

`tests/test_stage1_prepare_cli.py` — **17 tests**, ML-free. They go through
`build_parser()` (the same path `main()` uses), parse a real `prepare-corpus`
argv, and run the whole pipeline with only the corpus pin, the shard reader and
the tokenizer injected — chunking, checkpointing, finalisation and the
completion marker all execute for real.

| Case | Result |
|---|---|
| Parsed Namespace has **no** `repository_head` attribute | pass |
| START runs end to end, no `AttributeError`, document order preserved | pass |
| Checkpoint records the **real** resolved HEAD, 40 hex | pass |
| ALREADY_COMPLETE short-circuits and does not alter artifacts | pass |
| RESUME after simulated death, no skip or repeat | pass |
| A checkpoint from **another HEAD** cannot resume | pass |
| Pre-checkpoint failure leaves nothing resembling progress | pass |
| Stale directory without state/COMPLETE refused; user data untouched | pass |
| `--help` side-effect free; no `--repository-head` in help | pass |
| HEAD is a full 40-char SHA matching `git rev-parse` | pass |
| Non-SHA results (`main`, abbreviated, empty, non-hex) rejected | pass, 4 cases |
| `git` exit 128, and a missing `git` binary, both fail closed | pass |
| No `environ`/`getenv`/`"unknown"` in the resolver | pass, AST over the body with the docstring stripped |

The corpus fixture uses **5 200 documents** because `DEV_DOCUMENTS = 5000` is a
locked scientific constant — the fixture was enlarged rather than the constant
lowered.

### W.7 What did NOT change

`unmark/stage1/checkpoint.py`'s architecture (only the new resolver was added),
the streaming writer, immutable shards, document-boundary commits, state
atomicity, COMPLETE semantics, `lengths.py` composition and wrapper-only run
counting, **direct BPE remains removed**, `chunking.py`, `corpus.py`, the
contamination criterion, manifest scientific semantics, `RAW_BASE`,
`max_length`, seeds, dev count, `pi_strip`. The **same 21** scientific
constants enumerated in §V.6 were re-compared, here against **`f9c23fe`**:
**NONE changed**. `unmark/stage1/protocol.py` additionally has **zero diff**
against `f9c23fe` — this repair did not touch it at all, so `RAW_BASE_POLICY`
(added at `f9c23fe`) is unchanged too, but it is still excluded from the 21 for
the reason given in §V.6.

### W.8 Tests

| Suite | Result |
|---|---|
| `tests/test_stage1_prepare_cli.py` | **17 passed** (new) |
| `tests/test_stage1_checkpoint.py` | 41 passed |
| `tests/test_stage1_runner_contract.py` | 43 passed |
| `tests/test_stage1_lengths.py` | 325 passed |
| `tests/test_stage1_chunking.py` | 35 passed |
| `tests/test_stage1_contamination_prefilter.py` | 287 passed |
| Full repository | **3 209 passed, 97 skipped** |

### W.9 Limitations

1. **The full prepare is still not complete.** Stage 6 has **never committed a
   single checkpoint interval** on the real corpus, so nothing about real
   checkpoint overhead, shard sizes or sustained Stage-6 throughput is measured.
   The 5 000 documents chunked at `4c72639` (§U.1) were a **pre-checkpoint timing
   run**: documents were processed, but no shard was ever committed and no state
   was ever written.
2. **Real Drive resume has not been demonstrated.** All durability evidence is
   local-filesystem and stubbed-tokenizer.
3. The end-to-end regression uses a stub tokenizer; the real tokenizer is
   exercised only by the separate probe.
4. No Stage-6 resume PASS is claimed.
5. Training remains unauthorised, and this audit does not authorise the full run.

### W.10 Self-audit

| # | Check | Result |
|---|---|---|
| 1 | Root cause established from code, not the traceback | **yes** — §W.2, five questions answered |
| 2 | `prepare-corpus` no longer reads a nonexistent field | **yes** — AST-asserted |
| 3 | Actual repository HEAD derived internally | **yes** — `git rev-parse HEAD` |
| 4 | No caller-controlled HEAD override added | **yes** — no flag, no env var |
| 5 | Full 40-char SHA required | **yes** — 4 rejection cases |
| 6 | Resolution failure fails closed | **yes** — exit 128 and missing binary |
| 7 | HEAD remains checkpoint identity | **yes** |
| 8 | A different HEAD cannot resume | **yes** — explicit test |
| 9-11 | START / RESUME / ALREADY_COMPLETE regressions at the real parser boundary | **yes** |
| 12 | Failed pre-checkpoint state understood and safe | **yes** — nothing is created before the crash line |
| 13 | Checkpoint architecture unchanged | **yes** — resolver added only |
| 14 | Direct BPE remains removed | **yes** — real probe confirms |
| 15 | Composition hardening unchanged | **yes** |
| 16 | Scientific constants unchanged | **yes** — the same **21**, compared against `f9c23fe`; `protocol.py` has zero diff |
| 17-19 | `chunking.py`, contamination, sealed TEST | **yes** |
| 20-23 | No encoder, forward, optimizer, training | **yes** |
| 24-25 | Focused and full suites pass | **yes** — 3 209 passed |
| 26-27 | `git diff --check` clean; nothing staged | **yes** |
| 28 | Audit 029 revised in place; no Audit 030 | **yes** |
| 29 | Stage-6 resume claimed PASS? | **NO** |

---

## X. CONSISTENCY CLEANUP — AUDIT-ONLY (2026-08-23)

**Baseline:** `f9c23fedd4b5dd85b206454886a3e5bade3cfa86`.
**Scope: this audit file only.** No implementation code, no test, no scientific
configuration and no decision entry was touched. The runner-wiring repair of §W
is **accepted as it stands**; nothing about it is re-litigated here.

### X.1 The 21-vs-22 constant count, resolved from evidence

The chat report accompanying §W said "**22** scientific constants: NONE
changed", while the audit itself said **21**. One of them was wrong, and the
disagreement was resolved by re-running the comparison rather than by picking a
number.

The enumerated set in §V.6 was expanded from its group labels ("all seeds",
"both grids") into explicit names and re-compared by value:

| Baseline | Constants compared | Changed |
|---|---|---|
| `4c72639` (Revision-3c hardening, §V.6) | **21** | **NONE** |
| `f9c23fe` (this repair, §W.7) | the **same 21** | **NONE** |

**The audit's 21 is correct; the chat report's 22 was a miscount.** The extra
name was `RAW_BASE_POLICY`, and it must **not** be counted: it is the constant
Revision 3c *added* (`protocol.py` is `+6/−0` against `4c72639`), so it has no
prior value to be stable against. Including it would let a newly-introduced
operational identity constant pose as evidence that nothing moved.

Both figures now carry their baseline explicitly, because they are genuinely two
comparisons — a different working tree against a different commit — that happen
to cover the same 21 names and return the same answer. Additionally,
`unmark/stage1/protocol.py` has **zero diff** against `f9c23fe`: the
runner-wiring repair did not touch the file at all.

### X.2 Stale CURRENT metadata corrected

The header still described the audit as it was created on 2026-08-22 —
"**Execute none of it**", "No real Stage-1 run, no corpus download, no model
load". Those statements were true then and are **false as current metadata**:
the corpus has been downloaded and inspected in Colab, the real pinned tokenizer
probe has passed, and Stages 1-5 pass on all 1 118 224 documents.

The header now separates **Scope — as created** (explicitly labelled historical
and superseded, wording preserved) from **Scope — CURRENT**, and adds explicit
**What HAS run on real data** and **What has NOT run** rows plus the current
baseline HEAD. No historical section was deleted; §L, §M, §A1, §A2 and the
superseded status blocks remain exactly as they were, still labelled.

### X.3 One evidence conflation removed

§A previously read "Stage 6 has run partially (5 000 documents, ~29.5 docs/s)"
directly alongside §W's "never reached one checkpoint interval". Both are true
but they describe **two different runs**, and side by side they read as a
contradiction. §A now distinguishes them: a pre-checkpoint **timing** run at
`4c72639` that chunked 5 000 documents and committed nothing, versus the
`f9c23fe` run that crashed at document 0 with checkpointing present. §W.9
limitation 1 is sharpened the same way.

### X.4 What this cleanup did NOT do

It did not weaken or strengthen the verdict, did not add any real-data claim,
and did not remove any limitation. The §W findings stand unchanged: the missing
`repository_head`, the internally resolved fail-closed 40-character HEAD with no
caller override, the second latent `RAW_BASE_POLICY` import crash, the failed
probe having created no artifacts, 17 real-parser tests, and 3 209 passed /
97 skipped.

### X.5 Self-audit for the cleanup

| # | Check | Result |
|---|---|---|
| 1 | Only Audit 029 changed | **yes** — no code, test, config or decision file touched |
| 2 | Another doc carries the same stale claim? | **no** — grep across `docs/` and `*.md` found the wording only here |
| 3 | CURRENT metadata no longer says nothing has executed | **yes** — §X.2 |
| 4 | Historical text still clearly historical | **yes** — original wording preserved and labelled superseded |
| 5 | 21-vs-22 resolved from an actual re-run, not a preference | **yes** — §X.1 |
| 6 | Baseline stated for every count | **yes** — `4c72639` and `f9c23fe` named at each site |
| 7 | `RAW_BASE_POLICY` excluded from the unchanged set | **yes**, with the reason |
| 8 | Verdict still runner-wiring repair PASS | **yes** — unchanged |
| 9 | Stage-6 checkpoint / resume claimed PASS? | **NO** |
| 10 | Full prepare claimed PASS? | **NO** |
| 11 | PRE-TRAIN readiness claimed? | **NO** |
| 12 | Training or full-run authorised? | **NO** to both |
| 13 | Next step stated as the real Drive START → checkpoint → kill → RESUME probe | **yes** — §A and the header |
| 14 | Real evidence exaggerated anywhere? | **no** — one conflation removed (§X.3) |
| 15 | `git diff --check` clean; nothing staged | **yes** |

---

**STATUS: AUDIT 029 CONSISTENCY CLEANUP PASS — READY TO COMMIT RUNNER-WIRING REPAIR**
**VERDICT UNCHANGED: REVISION 3C RUNNER-WIRING REPAIR PASS — READY FOR REAL DRIVE RESUME PROBE**
**CLEANUP WAS AUDIT-ONLY: NO CODE, NO TEST, NO SCIENTIFIC CONFIG, NO DECISION TOUCHED (§X)**
**CURRENT METADATA NOW DESCRIBES REALITY; "EXECUTE NONE OF IT" LABELLED HISTORICAL**
**CONSTANT COUNT RESOLVED: 21, NOT 22 — `RAW_BASE_POLICY` WAS NEWLY ADDED, NOT UNCHANGED**
**21 COMPARED AGAINST `4c72639` (§V.6) AND THE SAME 21 AGAINST `f9c23fe` (§W.7) — NONE CHANGED**
**NEXT STEP IS THE REAL DRIVE START -> CHECKPOINT -> KILL -> RESUME PROBE, NOTHING LARGER**
**NO STAGE-6 CHECKPOINT COMMITTED, NO FULL PREPARE, NO PRE-TRAIN, NO TRAINING AUTHORISED**

**REAL PROBE CRASHED AT STAGE 6: `args.repository_head` NEVER EXISTED ON `prepare-corpus` (§W)**
**HEAD NOW DERIVED FROM THE EXECUTING TREE VIA `git rev-parse HEAD`, FAIL-CLOSED, NO CLI OVERRIDE**
**A SECOND LATENT CRASH FOUND BY THE NEW END-TO-END TEST: `RAW_BASE_POLICY` WAS UNIMPORTED**
**BLIND SPOT CLOSED: 17 TESTS NOW RUN THROUGH THE REAL PARSER INTO `run_prepare_corpus`**
**FAILED PROBE LEFT NO ARTIFACTS — NOTHING IS CREATED BEFORE THE CRASH LINE**
**REAL TOKENIZER PROBE PASSED; STAGES 1-5 PASSED ON ALL 1 118 224 DOCUMENTS**
**STAGE 6 HAS STILL NEVER REACHED ONE CHECKPOINT INTERVAL — NO RESUME PASS CLAIMED**

~~**STATUS: REVISION 3C HARDENING PASS — READY FOR REAL RESUME/PERFORMANCE PROBE**~~ **— superseded by §W**
**DIRECT-BPE FAST PATH REMOVED: IT BYPASSED THE WRAPPER'S ADDED-TOKEN SPLIT (§V)**
**COMPOSITION NOW GATED ON THE TOKENIZER'S OWN ADDED TOKENS; UNKNOWN = UNSAFE**
**WRAPPER-SENSITIVE RUNS EXACT BEFORE AND AFTER THE 256-QUERY WINDOW**
**`protocol.py` IS +6/-0 ADDITIVE vs `4c72639`; 21 SCIENTIFIC CONSTANTS COMPARED, NONE CHANGED**
**TOP-LEVEL VERDICT NOW CURRENT; PRIOR VERDICTS MARKED HISTORICAL**

~~**STATUS (Revision 3c, first pass): REVISION 3C PASS**~~ **— hardened by §V**
**DURABLE STAGE-6 RESUME: APPEND-ONLY SHARDS, DOCUMENT-BOUNDARY COMMITS, FAILURE-ATOMIC**
**SECOND BLOCKER FOUND AND FIXED: THE PRE-3c WRITER HELD ~29.9 GB OF CHUNKS IN RAM**
**RESUMED OUTPUT BYTE-IDENTICAL TO UNINTERRUPTED, AT 8 SIMULATED DEATH POSITIONS**
**13 IDENTITY FIELDS FAIL CLOSED; COMPLETE MARKER WRITTEN LAST AND HASH-BOUND**
**DIRECT-BPE WITHDRAWN BY THE 3c HARDENING (§V); MULTIPROCESSING MEASURED 1.78x@2 BUT NOT ADOPTED**
**GPU VRAM DOES NOT ACCELERATE CPU TOKENIZATION — NOT CLAIMED**
**FULL 1.118 M RUN NOT PERFORMED; NOT AUTHORISED HERE**

~~**STATUS (Revision 3b): REVISION 3B REPAIR PASS — READY FOR REAL TOKENIZER/PERFORMANCE PROBE**~~ **— SUPERSEDED by Revision 3c**
**HISTORICAL `composed 5, exact 7` EXPLAINED: bb50823 COMPOSED OVER `\S+`, NOT THE TOKENIZER'S `\S+\n?`**
**REVISION 3a's "COMPOSITION FALSIFIED" READING IS WITHDRAWN — THE RUN UNIT WAS WRONG, NOT THE PROPERTY**
**REAL SAMPLE: `\S+` 1708/1920 FAILURES, `\S+\n?` 0 FAILURES, wrapper vs `_tokenize` 0**
**STAGE 6 CONFIRMED BLOCKER AT ~0.45 docs/s (WEEKS) — 3b REDUCES BPE WORK TO DISTINCT RUNS**
**34 BPE EVALUATIONS AND 514 AUTHORITATIVE CALLS FOR 72 216 QUERIES; OUTPUT IDENTICAL**
**NOT RUN ON THE REAL TOKENIZER; REAL STAGE 6 STILL NEVER COMPLETED**

~~**STATUS (Revision 3a): REVISION 3A REPAIR PASS — READY FOR REAL TOKENIZER PROBE**~~ **— SUPERSEDED by Revision 3b**
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

~~**STATUS (Revision 2): PERFORMANCE REPAIR PASS — READY FOR REAL PREPARE-CORPUS RE-RUN**~~ **— SUPERSEDED by Revision 3**
**DEFECT 2: STAGE-4 CONTAMINATION SCREEN CANONICALISED ALL 1 118 224 DOCUMENTS (>7 h) — REPAIRED (§Q)**
**CONTAMINATION CRITERION UNCHANGED: `sha256(canon(x))`, DECIDED ONLY AT TIER 3**
**PREFILTERS PROVEN NECESSARY CONDITIONS — 360 084 TRIALS, 0 COUNTEREXAMPLES**
**FULL-REPORT EQUIVALENCE VS THE PRE-OPTIMISATION ORACLE**
**REAL CORPUS NOT RE-PREPARED — NO PERFORMANCE CLAIM AGAINST IT**
**CHUNKING STILL UNVERIFIED ON REAL DATA — STAGE 5 WAS NEVER REACHED**

~~**STATUS (Revision 1): REPAIR PASS — READY FOR REAL CORPUS RE-RUN**~~ **— SUPERSEDED by Revision 2. Every line below is the state as of 2026-08-22, including "NO REAL STAGE-1 SCIENTIFIC EXECUTION OCCURRED", which is no longer current: see the header and §A.**
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
