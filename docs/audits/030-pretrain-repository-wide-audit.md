# Audit 030 — PRE-TRAIN repository-wide audit

| | |
|---|---|
| **Audit id** | 030 |
| **Created (UTC)** | 2026-08-23 |
| **Baseline HEAD** | `aa49785eadcbd67b64be28a5f67d725c79b41bbb` — clean tree, nothing staged |
| **Predecessor** | [029](029-stage1-runner-implementation.md) §AB — the first successful complete real Stage-6 prepare |
| **Type** | **Phase-boundary gate.** Repository-wide review of everything between a finished corpus and a first training run |
| **Executed here** | **Nothing.** No encoder load, no forward pass, no optimizer, no update, no weight download |
| **Revision 1** | 2026-08-23 — the audit as first written (§A-§Q). Findings F1-F6 |
| **Revision 2 — hardening** | 2026-08-23 — **F1, F2 and F3 repaired**, plus three further defects found while wiring F3. See **§R**. F4/F5/F6 unchanged |
| **Revision 3 — addendum** | 2026-08-23 — F3 re-read against the log: it was a **wiring defect**, the cadence and **best + last** semantics were already locked by **D-S1B-004**. The invented D-S1B-014 is **withdrawn**; `best` is now persisted too. Validation-cost concern **measured, not acted on**. See **§S** |
| **Revision 4 — final consistency check** | 2026-08-23 — **BLOCKER found in §S's own tool**: `--validation` loads the real encoder but runs **no forward pass**, so it does not measure validation wall-clock. Tool and audit corrected to say so; the measurement remains outstanding. Token sampling verified **deterministic**. See **§T** |
| **Revision 5 — measurement repair** | 2026-08-23 — the §T blocker is **repaired**: `--validation` now runs the authoritative `validation.evaluate` with real forwards, CUDA-synchronised timing, real GPU peak, clean-reference attribution and parameter-hash proof of zero updates. Profile sampling is partition-aware (**all 11 443 dev**). See **§U** |
| **Revision 6 — first real smoke** | 2026-08-23 — the **first real Colab pre-train smoke** was run at `8f07842`. It passed every corpus gate (byte-exact restore, manifest, `COMPLETE.json`) and stopped fail-closed at `tests/test_stage1_training_resume.py`: **6 failed, 8 passed**, all six dying in setup on `RunProvenance(**mine.to_dict())`. Traced to a **test-construction bug** — `to_dict()` is artifact serialization, not a constructor round trip — plus one real gap: the derived weights were serialized and never validated on read. **No model was loaded.** See **§V** |
| **Revision 7 — second real smoke** | 2026-08-23 — the **second real Colab no-update smoke** ran at `b84b4da`. Corpus gates, and the 29 + 17 + 22 provenance/resume/measurement tests, all **passed on real hardware** — confirming §V. It then failed closed in condition preparation with `EligibilityUnresolved`: the pinned syllable inventory is **deliberately not committed** and a fresh runtime had not provisioned it. Classified **A + C** — the artifact was already locked by D-B3A-001, but the check ran *after* model load and the **blocking D-S1A-008 was unimplemented**. **No model forward, no optimizer, no update.** See **§W** |
| **Decides** | Whether the *next* step — a bounded real **no-update model smoke** — may proceed. **It does not authorise training** |

---

## A. VERDICT

**PRE-TRAIN INVENTORY PROVISIONING REPAIR PASS — READY TO COMMIT AND RERUN NO-UPDATE SMOKE**

> **§W is the current verdict.** The second real smoke confirmed §V on real
> hardware and then failed closed before completing validation: the pinned
> Vietnamese syllable inventory is deliberately **not committed** (no upstream
> license statement) and a fresh runtime had not provisioned it. That failure was
> the **designed** contract working. The real defects it exposed were that the
> check ran **after** the encoder was loaded, and that **D-S1A-008** — a decision
> whose status line reads "BLOCKING for scientific Stage-1 training and the
> PRE-TRAIN audit" — had **never been implemented**, so no run artifact could name
> the inventory it used. Both are repaired; §T, §U and §V are preserved unchanged.
>
> **§W also corrects this audit's own earlier "No UNRESOLVED MISMATCH"**, which
> was true of the code and too strong as a pre-train readiness claim.
>
> **The real no-update smoke has NOT passed.** Validation never completed: no
> PhoBERT forward, no optimizer, no update has yet occurred.

~~**PRE-TRAIN REAL-SMOKE BLOCKER REPAIR PASS — READY TO COMMIT AND RERUN NO-UPDATE SMOKE**~~ **— superseded by §W**

> **§V was the previous verdict.** The first real Colab smoke passed every corpus
> gate and then stopped fail-closed at the training-resume gate, before any
> model was loaded. The cause was a **test-construction bug** — six foreign-run
> cases died in setup and had never demonstrated rejection — together with one
> real gap it exposed: the derived `lambda_align`/`lambda_clean` were written
> into every checkpoint and never validated on read. Both are repaired; §T and
> §U are preserved above unchanged.
>
> **The real no-update smoke has NOT passed.** No PhoBERT load, no forward, no
> validation, no optimizer, no update has yet occurred.

~~**PRE-TRAIN HARDENING FINAL PASS — READY TO COMMIT AND RUN REAL NO-UPDATE MODEL SMOKE**~~ **— superseded by §V**

> **§U was the previous verdict.** §T's blocker — `--validation` loading the real
> encoder and running **no forward pass** — is repaired: the tool now
> instruments the authoritative `validation.evaluate`, and a report cannot claim
> validation without real forwards, four conditions, and byte-identical
> parameters. §T is preserved above as the record of the defect. F1/F2/F3 are
> unaffected and still hold.
>
> **A PASS does not authorise training.**

> **§A-§Q are the audit as first written and are preserved unrevised.** F1, F2
> and F3 have since been repaired; **§R** records exactly what changed and what
> evidence now exists. The original verdict below stands as the audit's finding
> at the time.

**REVISION 1 (original verdict): PRE-TRAIN REPOSITORY-WIDE AUDIT PASS — READY FOR REAL NO-UPDATE MODEL SMOKE**

The corpus is complete and verified, the scientific configuration is single-sourced
and unchanged, official TEST is unreachable by any route I could construct, and
the corruption/validation/selection mathematics are deterministic and
seed-separated.

**Four findings.** None is a scientific mismatch, a leakage path, or a
corpus-integrity defect. Two must be closed before the *first* training run, two
before *long* runs. All four are stated below with their classification, and none
was silently fixed.

**A PASS here does not authorise training.** The next step is the bounded
no-update smoke in §N. Training remains unauthorised until that smoke is run and
reviewed.

---

## B. FINDINGS, CLASSIFIED

| # | Finding | Classification |
|---|---|---|
| **F1** | The training consumer never verifies `chunks.jsonl` against the manifest digest or `COMPLETE.json`; it records a digest it did not check | **MUST FIX BEFORE FIRST TRAIN** |
| **F2** | `PI_STRIP` is two independent literals: `contracts.PI_STRIP` governs corruption, `protocol.PI_STRIP` is what the manifest records | **MUST FIX BEFORE FIRST TRAIN** |
| **F3** | Training checkpointing is implemented but **never invoked**; `execute_stage` hard-codes `resume=None` and nothing calls `checkpoint_payload` | **SHOULD FIX BEFORE LONG RUNS** |
| **F4** | `load_prepared_chunks` materialises the whole corpus (~2.73 GB of Python objects) and discards every field except `chunk_id` and `text` | **MUST VERIFY IN REAL NO-UPDATE SMOKE** |
| **F5** | Stage-6 transform counters are structurally blind under `--prepare-workers > 1` | **NON-BLOCKING (observability)** |
| **F6** | Compiled proposal PDF is stale | **DOCUMENTATION ONLY** |

**No BLOCKER was found.** Each finding below is traced, not asserted.

---

## C. F5 first — the zero counters were NOT a bypassed length contract

Task 4 required this to be resolved before anything else, because if the
authoritative length contract had been bypassed the audit stops. **It was not.**

The full run printed:

```
length queries 0, BPE run evaluations 0, run-cache hits 0,
evictions 0, incremental appends 0, full fallbacks 0,
authoritative verifications 2
```

Reproduced locally: a **freshly constructed, never used** pair of length
functions reports **exactly** that signature.

| Counter | Fresh pair | Real run |
|---|---|---|
| `authoritative_queries` | **2** | **2** |
| `length_queries`, `bpe_run_evaluations`, `run_cache_hits`, `incremental_appends`, `full_fallbacks` | **0** | **0** |

`authoritative_queries` is 2 because `build_length_functions` creates **two**
`RunLengthComposer`s (reference and base) and each derives its special-token
count with one `authoritative_length("")` call in `__init__`.

**Why the main process's counters are blind.** `scripts/stage1_runner.py:191`
builds `transforms` in the main process and prints its counters at line 286. It
passes them at line 255 as `serial_length_functions`, which
`parallel.ordered_document_chunks` uses **only when `workers <= 1`**
(`parallel.py:146`). At `--prepare-workers 16`, every worker builds **its own**
length functions in `_initialise_worker` (`parallel.py:69`) — including its own
fail-closed 256-query authoritative verifier.

**So the contract ran 16 times over, in 16 processes, and the main process could
not see any of it.** Independent confirmation that it held: the run completed
with **no `Stage1ContractViolation`**, and the manifest reports
`overflow_count 0` — every one of 2 633 067 chunks was independently
length-checked inside `chunk_document` itself, which does not depend on these
counters at all.

**Classification: observability issue, science correct.** Pinned by
`test_a_freshly_built_length_pair_reports_exactly_two_authoritative_queries` and
`test_each_worker_builds_its_own_length_functions`, so this reading can be
re-derived and a refactor that shares main's functions with workers is caught.

---

## D. Task 1 — proposal → implementation traceability

Proposal reread end to end (758 lines), against `docs/spec/decisions.md`
(4 496 lines) and Audits 028/029.

| Requirement (proposal) | Implementation | Class |
|---|---|---|
| §4.2 `dec(x) = (b(x), τ(x), λ(x))`; NFD, enumerated Vietnamese marks, **recomposition of the base** | `orthography/decompose.py` | **EXACT MATCH** (restored by D-S1B-013; §AA) |
| §4.2 `rec(dec(x)) = canon(x)` | `recompose`, tested across 7 script families | **EXACT MATCH** |
| §4.2 nucleus-based tone placement (`hòa` → `hoà`) | `orthography/placement.py`, D-001 | **EXACT MATCH** |
| §4.3 candidate = alphabetic span matching the inventory after stripping; pure function of the stripped form | `linguistics/classify.py` | **EXACT MATCH** |
| §5.x split **before** chunking; no truncation; preserve text | `stage1/chunking.py`; manifest asserts `split_before_chunk`, `truncation=false` | **EXACT MATCH** |
| §5.x protected unit must not be split | `source_letter_runs` + `protects_a_vietnamese_candidate` | **IMPLEMENTATION CLARIFICATION** — narrowed from "any Unicode alphabetic" to Latin-script (D-S1B-013). Recorded, not silent |
| §4.4 `T(b(x))` defines the grid; no post-strip segmenter | `lengths.build_length_functions` | **EXACT MATCH** (D-B3B1A-001) |
| §4.4 every non-Vietnamese subword carries `N/A` in both channels | `Eligibility.NOT_APPLICABLE` | **EXACT MATCH** |
| §6 corruption `p ~ U(0,1)`, redraw per visit | `contracts.CorruptionRatePolicy` | **EXACT MATCH** |
| §6 `pi_strip` scope mixture | `contracts.scope_for`, D-S1B-003 | **EXACT MATCH** — but see **F2** |
| §5 objective: cosine, pooled, attention-masked mean over non-special | `stage1/objective.py` | **EXACT MATCH** |
| §5 frozen encoder, adapter only | `verify_model_contract`, `build_optimizer` | **EXACT MATCH — NOT YET EXECUTED** |
| §6 11-run plan, budget 20k → 40k | `stage1/selection.py` | **EXACT MATCH — NOT YET EXECUTED** |
| Official TEST sealed | `OFFICIAL_TEST_ACCESSIBLE = False`, no flag, screen refuses | **EXACT MATCH** |
| Chunking safe-cut **coordinate system** | source coordinates (D-S1B-012) | **IMPLEMENTATION CLARIFICATION** — proposal is silent on coordinates, correctly |

**No UNRESOLVED MISMATCH.** Everything in the training half is **NOT YET
EXECUTED**, which is precisely why the no-update smoke exists.

**Drift check.** The two clarifications (D-S1B-012, D-S1B-013) both moved the
*implementation* toward the specification, not away from it: §4.2 already said
"recomposition of the base", and the implementation had drifted from it. The
editable proposal needs no change.

**F6 — stale compiled PDF.** The editable `unmark-proposal.md` is current; only
the compiled PDF is stale. It is a **release artifact, not a scientific input**:
no code reads it, and no run binds it. **It must be regenerated before external
release, not before training.** Not regenerated here — doing so cannot affect
science, and doing so unnecessarily would add an unreviewed binary diff to a
phase-boundary commit.

---

## E. Task 2 — scientific constants, re-derived from the repository

Enumerated by importing `unmark.stage1.protocol`, not from memory:

| Constant | Value |
|---|---|
| corpus / revision | `undertheseanlp/UVW-2026` / `a0a79294e4568137e25828bb3f2a4cde8546e1fb` |
| shard order | `('train.parquet', 'validation.parquet', 'test.parquet')` |
| `MAX_LENGTH` | **256** |
| `SPLIT_SEED` / `DEV_DOCUMENTS` | **51733** / **5000** |
| encoder / revision | `vinai/phobert-base` / `01daacda68afe13d83023d16ec647239e344a1e6` |
| `PI_STRIP` | **0.25** |
| `CORRUPTION_SEED` | **35422** |
| rate distribution / redraw | `uniform_0_1_per_example` / `per_visit` |
| `VALIDATION_CONDITIONS` | `('FULL', 'P50', 'P100', 'STRIP_ALL')` |
| `VALIDATION_CORRUPTION_SEED` | **19225** |
| `BATCH_SIZE` / `EVAL_EVERY_UPDATES` | **128** / **500** |
| budget / continuation | **20 000** → one continuation → **40 000** → `BUDGET_LIMITED` |
| `LR_PILOT_GRID` | `(1e-4, 3e-4, 1e-3)` |
| `R_PHASE1_GRID` | `(0.25, 0.5, 1.0, 2.0, 4.0)` |
| `TOTAL_NOMINAL_RUNS` | **11** |
| adapter trainable parameters | **3 551 232** |
| `PRECISION` / `ON_OVERFLOW` | `fp32` / `FAIL` |
| selection score | `max over VALIDATION_CONDITIONS of mean cosine distance to h(x)` |
| `RAW_BASE_POLICY` | `RAW_BASE` |
| `OFFICIAL_TEST_ACCESSIBLE` | **False** |

**No constant was changed by this audit.** All match Audit 028's lock.

### E.1 F2 — `PI_STRIP` is two independent literals

Searched for competing definitions of every constant. One real duplicate inside
Stage-1:

| Site | Role |
|---|---|
| `unmark/stage1/contracts.py:375` | **governs the science** — `CorruptionRatePolicy.pi_strip` defaults to it (`contracts.py:427`) and `is_locked_mixture` compares against it |
| `unmark/stage1/protocol.py:89` | **recorded in the manifest** (`protocol.py:270`) and printed by the runner |

`contracts.py` does **not** import `protocol.py`; these are two separately typed
`0.25` literals. They agree today, so the science is correct and the manifest is
honest. The risk is a future single-sided edit: the corruption engine would draw
against one value while every artifact recorded the other, and **nothing in the
repository would notice**.

**Classification: MUST FIX BEFORE FIRST TRAIN.** The fix is one line — make one
import the other — but it is a change to a locked scientific constant's
definition site, so it belongs to the researcher, not to an audit. Pinned
meanwhile by `test_pi_strip_has_one_value_even_though_it_has_two_definitions`,
which fails the moment they diverge.

**Other apparent duplicates are separate namespaces, not conflicts.**
`MAX_LENGTH`, `BATCH_SIZE` and `ENCODER_REVISION` also appear under
`unmark/evaluation/preg1_*`, which is the completed pre-G1 diagnostic phase with
its own protocol. Values agree; the namespaces are deliberately distinct.

---

## F. Task 3 — leakage

**Official UIT-VSFC TEST is unreachable.** Adversarially checked:

| Route | Result |
|---|---|
| CLI flags | **12 total** in `stage1_runner.py`; none contains "test". The only UIT-VSFC flags are `--uitvsfc-derived-train` and `--uitvsfc-official-validation`, both already-opened sources, and both reachable **only** from `prepare-corpus` |
| contamination screen | `screen_contamination` raises `CorpusContractViolation` on any key outside `CONTAMINATION_SCREEN_INPUTS`; asserted by test with `uitvsfc_official_test` |
| filesystem discovery | The **only** glob in `unmark/stage1/` is `directory.glob("*.tmp")` for orphan-temp cleanup in `checkpoint.py:199`. No `rglob`, `iterdir`, `listdir` or `os.walk` anywhere in Stage-1 |
| manifest | `require_compatible` refuses any manifest not recording `official_test_used=false` |
| smoke / debug | `smoke_check` takes `--prepared-corpus` only; `stage1_blocker_probe.py` takes an explicit shard and row |
| every textual occurrence | All 20 matches for "official test" across `unmark/`, `scripts/`, `configs/` are **seal assertions or refusals**, never a read path |

**Stage-1 train/dev isolation.** The split is document-level and happens
**before** chunking: `chunk_document` *takes* a partition and copies it onto
every chunk, and no code path assigns one. `chunk_corpus` refuses a document
without a partition, and `verify_no_parent_spans_partitions` asserts the
invariant independently. The real run confirms it end to end:
**`parents_spanning_both_partitions = 0`** over 1 118 224 parents.

**Contamination semantics** match the proposal: exact/canonical only,
`sha256(canon(x))`, no fuzzy matching, decided only at tier 3, with the tier-1/2
prefilters proven necessary conditions (Audit 029 §Q). Real result: **0 excluded
of 1 118 224**, 0 candidates.

**Selection leakage.** Stage-1 selection reads only the held-out **unlabeled** dev
signal. No downstream score enters it; `validation.py` binds
`VALIDATION_CORRUPTION_SEED` and records
`training_seed_affects_validation_corruption: False`.

**Assessment: no leakage path found.**

---

## G. Task 4 — prepared-corpus integrity

Traced source → split → slice → chunk → both pathways → serialisation →
manifest/checkpoint/COMPLETE → loader.

| Property | Evidence |
|---|---|
| no normalisation into chunks | `chunking.py` AST-asserted to call no `canon`/`normalize`/`replace`/`sub`; every chunk is `content[start:end]` |
| no truncation | `TRUNCATION_OFFERED = False`, `ON_OVERFLOW = FAIL`, manifest asserts `truncation=false`; real `overflow_count = 0` |
| no dropped documents | `parent_documents_total = 1 118 224` |
| exact source tiling | `verify_tiles_source`; `PreparedChunk.__post_init__` refuses any chunk whose text length disagrees with its range |
| stable ids | `{document_id}#{chunk_index}`, re-derived and asserted |
| deterministic ordering | JSONL line order is document order; the ordered collector emits strictly by index; byte-identical across 1/2/4/8 workers |
| both pathways ≤ 256 | `overflow_count = 0` over 2 633 067 chunks |
| partition membership | train 2 621 624 + dev 11 443 = 2 633 067 ✓; parents 1 113 224 + 5 000 = 1 118 224 ✓ |
| digest | `chunk_membership_digest` sorts its keys, so it is order-independent; external merge sort proven to produce the identical digest |
| COMPLETE | written **last**, binds artifact hashes; `ALREADY_COMPLETE` re-verified every artifact on a **cold second process** |

**Assessment: the prepared corpus is internally consistent and durably verified.**

---

## H. Task 5 — orthography / RAW_BASE

Re-audited after both repairs. NFC, NFD, tone marks, letter-forming marks, `đ`,
non-canonical placement, combining sequences, non-Vietnamese scripts,
Latin-but-not-Vietnamese, and OOV-valid Vietnamese are covered by 93 dedicated
tests plus 1 098 orthography tests.

### H.1 The `señor` → `senor` collision — classified, not fixed

`U+0303` (tilde) **is** the Vietnamese *ngã* tone mark; `U+0301` is *sắc*;
`U+0300` is *huyền*. So `señor` → `senor`, `café` → `cafe`, `règle` → `regle`.

**Is this consistent enough with the proposal to permit training? Yes, and the
proposal already decided it.** §4.3 states the candidate rule is "orthographic
and structural, never semantic", and `linguistics/classify.py` forbids language
identification, corpus frequency, context and dictionaries **by design**, with
the accepted error mode written out: "an English word whose letters happen to
form a valid stripped Vietnamese syllable is classified as Vietnamese."

The collision is the *same* accepted trade-off one layer down: without language
ID, a combining mark cannot be attributed to a language. Removing it would
require exactly the mechanism the proposal forbids.

**Crucially, it does not threaten the experiment's validity**, because the
property the design rests on is **invariance, not accuracy**: `b(x) = b(x̃)` for
every corruption rate. A Spanish `ñ` is stripped identically in the clean and
corrupted branches, so the base grid stays invariant and corruption remains a
purely channel-level phenomenon. The measured
`base_invariance_violations = 0` over the real corpus is the direct evidence.

**Classification: NON-BLOCKING, documented limitation.** It is now pinned by test
so a future change to it is deliberate. It is **not** a scientific ambiguity that
could materially alter the experiment, so PRE-TRAIN is not blocked on it.

---

## I. Tasks 6, 13 — the training input loader

This is where the audit found the most.

### I.1 F1 — the consumer never verifies the corpus it loads

`load_manifest` → `require_compatible` **is** called by all three training
commands and validates schema, protocol version, `official_test_used=false`,
corpus revision, `max_length`, `split_before_chunk`,
`chunks_inherit_parent_partition`, `truncation=false`, tokenizer revision and
split seed. That is a real gate and it works.

**But it validates the manifest's *declarations*, not the data.**
`execute.load_prepared_chunks` opens `chunks.jsonl` and reads it whole. It does
**not** recompute `chunk_membership_digest`, and it **never consults
`COMPLETE.json`** — the very artifact Stage 6 built to bind file hashes, and
which `ALREADY_COMPLETE` proved works.

`execute_stage:127` then reads `manifest["counts"]["chunk_membership_digest"]`
and writes it into `RunProvenance`. So a run **records a digest it never
checked**. If `chunks.jsonl` were truncated, swapped, or copied from a different
prepare, training would proceed and every artifact would carry provenance
describing data it was not trained on.

**Classification: MUST FIX BEFORE FIRST TRAIN.** Provenance that can silently
misdescribe its own training data is exactly the defect class this gate exists
to catch. The fix is small — have the consumer call `read_completion` (already
written and proven) and/or recompute the membership digest over what it loaded —
but it is a substantive change to the training path and belongs in a reviewed
commit, not folded silently into an audit.

### I.2 F4 — the loader materialises the whole corpus

`load_prepared_chunks` builds two `dict[chunk_id, text]` over all 2 633 067
chunks. Measured cost: **~1 038 B/entry → ~2.73 GB of Python objects**, on top of
parse churn, for a 2.15 GiB payload.

This is the same non-streaming pattern Audit 029 §U removed from the *writer*
(where it would have been ~29.9 GB). It was never applied to the *reader*. On the
observed 176.88 GiB runtime it is survivable, which is why this is not a blocker.

It also **discards** every field except `chunk_id` and `text` — including the
recorded `reference_length` / `base_length`, `source_start/end`, `document_id`
and `source_shard`. Nothing currently needs them (corruption keys on `chunk_id`),
but it means the loader cannot cross-check the lengths the prepare recorded.

**Classification: MUST VERIFY IN REAL NO-UPDATE SMOKE** — the smoke opens the
real 2.2 GB corpus and is the natural, cheap place to observe both peak RSS and
load time before an 11-run plan depends on them.

---

## J. Tasks 7, 8 — corruption and validation

**Corruption is exemplary and needs nothing.** `_unit_draw` is a pure
`blake2b(namespace | schema_version | seed | sample_id | visit)` digest scaled to
`[0, 1)`. Consequences, all structural rather than tested-into-existence:

* **no global RNG** — nothing to seed, nothing to leak between train and dev;
* **no worker-count dependence** — the draw is a function of the example, not of
  scheduling;
* **no ordering dependence** — same reason;
* **resume-deterministic by construction** — replaying a `(sample_id, visit)`
  reproduces the same `p` with no state to restore;
* `p ~ U(0,1)`, redraw per `visit`, `pi_strip` scope mixture, and **rate and
  scope are domain-separated** by distinct namespaces so neither is derivable
  from the other.

**Validation** uses `VALIDATION_CORRUPTION_SEED = 19225` at a fixed visit, on the
fixed grid `FULL, P50, P100, STRIP_ALL`. The held-out realisation is built
**once** and reused by every candidate, so candidates differ only by the model.
`prepare_condition_batch` mentions no run seed. Selection score is `max` over the
grid; checkpoint tie-break `d_clean` then earliest update; `r` tie-break `d_clean`
then smaller `r`; update 0 is required.

**Assessment: deterministic, seed-separated, no train/dev RNG contamination.**

---

## K. Tasks 9, 10 — model, adapter, run plan (static only)

Inspected statically; **no weights loaded**.

* `verify_model_contract` requires no encoder parameter with `requires_grad`,
  the encoder in `eval`, and exactly **3 551 232** trainable adapter parameters
  at `d = 768`;
* `build_optimizer` **refuses** any parameter that does not require grad —
  silently filtering the encoder out would hide a wiring error;
* weight decay `0.01` on `fusion.weight`/`gate.weight` only; `0.0` on biases,
  LayerNorm and both embedding tables;
* `PRECISION = fp32`, and `autocast`/`GradScaler`/`half`/`bfloat16` are
  AST-asserted absent;
* schedule `LR_SCHEDULE = CONSTANT`, `WARMUP = None`,
  `GRADIENT_ACCUMULATION_STEPS = 1`, `GRADIENT_CLIPPING = None`;
* run plan 3 + 5 + 3 = **11**, budget 20 000 → one continuation → 40 000 →
  `BUDGET_LIMITED`; `selection.py` verifies the LR grid it was given against
  `LR_PILOT_GRID`;
* `smoke_check` calls no `backward`, `step`, `zero_grad`, `build_optimizer` or
  `AdamW` — **structurally incapable of an update**.

**Diagnostic containment.** The pre-G1 frozen-head and burden diagnostics live in
`unmark/evaluation/preg1_*` with their own protocol namespace and their own
`SECONDARY_ANALYSIS_LABEL`. They cannot become Stage-1 scientific results:
Stage-1 artifacts are produced only by `execute_stage`, bind
`STAGE1_PROTOCOL_VERSION`, and `require_compatible` refuses to pool artifacts
across protocol versions.

**All of this is NOT YET EXECUTED**, which the smoke exists to change.

---

## L. Task 11 — F3, training resume

The Stage-6 prepare checkpoint system is proven on real data. **Training
checkpointing is a different system, and it is not wired.**

* `trainer.checkpoint_payload` and `trainer.verify_checkpoint` exist and are
  unit-tested; `train_run` accepts `resume=` and, given one, restores adapter
  state, optimizer, `visit` and the in-pass cursor;
* **but `checkpoint_payload` is called from nowhere** in `unmark/` or `scripts/`,
  and `execute_stage` passes `resume=None` unconditionally.

So a training run that dies mid-stage cannot be resumed: there is nothing to
resume from, and `--output-dir` refuses to reuse its directory.

**Does it block the first training run? No.** A restarted run is deterministic
and scientifically identical — the corruption draws are stateless (§J), the
sampler is a pure function of `(chunk_ids, seed, visit)`, and no partial artifact
is left behind. The cost is wall-clock, not validity.

**Does it matter? Yes, before long runs.** Stage-6 needed several real Colab
runtime deaths to complete, and the 11-run plan is 20 000–40 000 updates per run
on the same infrastructure. Losing a run to a runtime death is likely, not
hypothetical.

**Classification: SHOULD FIX BEFORE LONG RUNS.** Pinned by
`test_training_checkpoint_payload_is_currently_never_persisted`, which documents
the gap as a decision and **fails the moment persistence is wired** — at which
point resume-equivalence needs its own evidence, exactly as the prepare
checkpoint did.

---

## M. Tasks 12, 14, 15 — provenance, blind spots, code quality

**Provenance.** `RunProvenance` binds run seed, corruption seed, learning rate,
`r`, corpus manifest digest and repository HEAD; `verify_checkpoint` refuses to
resume into a different experiment. Output directories must not already exist, so
a run cannot overwrite another. **The gap is F1**: the manifest digest is
recorded but unverified.

**Blind spots, searched for by the patterns that produced the previous real
bugs:**

| Pattern | Status |
|---|---|
| tests constructing `Namespace` manually | Closed in Audit 029 §W; `test_stage1_prepare_cli.py` goes through the real parser |
| tokenizer doubles hiding wrapper semantics | Closed in §V; composition is gated on the tokenizer's own added tokens, direct BPE removed |
| synthetic Unicode assumptions | Closed in §AA; the Hangul case is now explicit, 7 script families tested |
| **multiprocessing instrumentation differences** | **F5, found here** — and now pinned |
| **paths never entered end-to-end** | **The entire training path.** This is the largest remaining blind spot and it is exactly what the no-update smoke is for |
| mocks bypassing real behaviour | The training tests use toy tensors and injected length functions; no test has ever opened the real prepared corpus |

**Highest remaining risk, stated plainly: no code in this repository has ever
read the real 2.2 GB prepared corpus.** Every consumer test uses fixtures. That
is why §N is bounded but real.

**Code quality.** No exception swallowing on scientific paths (violations are
re-raised unchanged); no mutable global RNG; no nondeterministic iteration in
output paths (the membership digest sorts); writes are temp → fsync → replace;
JSON schemas are validated on load. Direct BPE remains absent.

**Audit evidence accuracy** was treated as correctness: Audit 029's header,
verdict and §AB were checked against the real numbers, historical failures were
left intact and labelled, and the collector-wait figure is explained rather than
reported as a defect.

---

## N. Task 17 — the exact real no-update model smoke

**Do not run this until this audit is reviewed.** It is the next step, and it is
the only thing that may load the model.

**Allowed:** model load, a small number of forward passes.
**Forbidden:** `optimizer.step()`, any parameter update, any training loop,
`loss.backward()`.

```bash
# Colab, GPU runtime. Exact HEAD, exact prepared corpus, exact pins.
cd /content/unmark-draft
git rev-parse HEAD            # must print aa49785eadcbd67b64be28a5f67d725c79b41bbb
git status --porcelain        # must print nothing

python - <<'PY'
import json, pathlib
d = pathlib.Path("/content/drive/MyDrive/UNMARK/stage1-checkpoints/aa49785eadcb")
c = json.loads((d / "COMPLETE.json").read_text())
print(json.dumps(c["counts"], indent=2, sort_keys=True))
assert c["counts"]["chunk_membership_digest"] == \
    "250859a57d745675c5dba2c7a35df08ccc123988bece873b0c9b29c6e78413d6"
PY

# The prepared payload and the COMPLETE marker live under different roots, so
# the completion directory is explicit. The smoke verifies the corpus BEFORE the
# model is loaded (F1) and reports load time and peak RSS (F4).
python scripts/stage1_runner.py smoke \
    --prepared-corpus /content/unmark-stage1-prepared-aa49785eadcb \
    --completion-dir /content/drive/MyDrive/UNMARK/stage1-checkpoints/aa49785eadcb \
    --revision 01daacda68afe13d83023d16ec647239e344a1e6 \
    --repository-head "$(git rev-parse HEAD)"
```

**Note (§R):** the HEAD printed by `git rev-parse` will be the *hardening*
commit, not `aa49785`. That is correct and expected — the prepared corpus is
bound by `COMPLETE.json` and its artifact hashes, not by the commit that reads
it. The hardening changed no scientific constant and did not touch Stage 6.

It must report, and a reviewer must check:

| # | Must verify | Why |
|---|---|---|
| 1 | `COMPLETE.json` digest equals `250859a5…78413d6` | binds the smoke to *this* corpus |
| 2 | the consumer opens the real corpus; **peak RSS and load time recorded** | **F4** |
| 3 | manifest counts match §AB.5 exactly | consumer/manifest compatibility |
| 4 | all **5 000** dev parents and **11 443** dev chunks present | dev set is what selection will use |
| 5 | a few train chunks tokenize on **both** pathways, every length ≤ 256 | reference and RAW_BASE agree with the recorded lengths |
| 6 | collator shapes, attention masks, padding, special tokens correct | never exercised on real data |
| 7 | encoder is frozen, adapter has exactly **3 551 232** trainable parameters | model contract |
| 8 | loss is **finite** on a forward pass | numerics |
| 9 | **zero parameter updates** — parameter hashes identical before and after | the smoke's own boundary |
| 10 | no optimizer constructed | structural |

Report `PASS`/`FAIL` with the counts. **Training remains unauthorised until that
report is reviewed.**

---

## O. Tests

Focused suites and the full lightweight repository suite were run. No test was
weakened.

| Suite | Result |
|---|---|
| `tests/test_stage1_pretrain_audit.py` | **8 passed** (new, this audit) |
| `tests/test_stage1_non_vietnamese_orthography.py` | 93 passed |
| `tests/test_stage1_source_coordinates.py` | 26 passed |
| `tests/test_stage1_chunking.py` | 35 passed |
| `tests/test_stage1_lengths.py` | 325 passed |
| `tests/test_stage1_checkpoint.py` | 41 passed |
| `tests/test_stage1_parallel.py` | 18 passed |
| `tests/test_stage1_prepare_cli.py` | 17 passed |
| `tests/test_stage1_runner_contract.py` | 43 passed |
| orthography / decompose / alignment | 1 098 passed |
| **Full repository** | **3 354 passed, 97 skipped, 0 failed** |

The 97 skips are torch-gated; **no model weights were downloaded**, and those
tests are delegated to the Colab smoke.

---

## P. Decision log

**No new decision-log entry was created, and that is deliberate.** This audit
made no specification decision, no clarification, no narrowing and no deviation:
it *verified* D-S1B-001 … D-S1B-013 against code and real artifacts. Manufacturing
an entry because an audit occurred would dilute the log's meaning.

Two findings **may** require entries **when they are fixed**, by whoever fixes
them: F1 (whether the consumer verifies `COMPLETE.json`, the membership digest,
or both) and F3 (training checkpoint cadence and resume-equivalence evidence).
Neither is decided here.

---

## Q. Self-audit

| # | Question | Answer |
|---|---|---|
| 1 | Proposal reread completely? | **yes** — 758 lines, end to end |
| 2 | Traceability matrix complete? | **yes** — §D, 15 requirements classified |
| 3-4 | Constants re-derived; none changed? | **yes** — §E, imported from the repository |
| 5-6 | Official TEST unreachable; leakage review complete? | **yes** — §F, six route classes checked adversarially |
| 7 | Parent partition isolation verified? | **yes** — structural + `parents_spanning_both_partitions = 0` |
| 8-9 | Completion evidence recorded; COMPLETE/ALREADY_COMPLETE reviewed? | **yes** — Audit 029 §AB |
| 10 | Real blocker history preserved? | **yes** — §P–§AA untouched |
| 11-12 | Orthography matches proposal; `señor` classified? | **yes** — §H, NON-BLOCKING with reasoning |
| 13-14 | Training loader and sampler reviewed? | **yes** — §I; **F1**, **F4** |
| 15-17 | Corruption, resume and validation RNG reviewed? | **yes** — §J, stateless by construction |
| 18-20 | Selection metric, LR plan, `r` plan reviewed? | **yes** — §J, §K |
| 21-24 | Adapter, trainable parameters, optimizer groups, scheduler reviewed? | **yes** — §K, statically |
| 25-27 | Run namespaces, training checkpoint identity, provenance reviewed? | **yes** — §L, §M; **F3** |
| 28 | Prepared-corpus consumer compatibility reviewed? | **yes** — §I, delegated to the smoke |
| 29-30 | Multiprocessing and real-tokenizer blind spots reviewed? | **yes** — §C, §M |
| 31-33 | No direct BPE; no normalisation/truncation/drop; nothing silently overwritten? | **yes** |
| 34-35 | No model update occurred; TEST still sealed? | **yes** |
| 36-38 | Full suite passes; `git diff --check` clean; nothing staged? | **yes** — 3 354 / 97 / 0 |
| 39-40 | Audit 029 updated in place; Audit 030 created? | **yes** |
| 41 | Decision log updated only if needed? | **yes** — deliberately **not** updated (§P) |
| 42-43 | Smoke specified; all findings classified? | **yes** — §N, §B |
| 44 | Same substantive audit returned in chat? | **yes** |

---

## R. PRE-TRAIN HARDENING — F1, F2, F3 REPAIRED

**Date:** 2026-08-23 **Baseline:** `aa49785eadcbd67b64be28a5f67d725c79b41bbb`

Sections A-Q above are the audit **as first written** and are not revised: F1-F6
were real, and the record of what was found matters as much as the fix. This
section records what was repaired and what evidence now exists.

| Finding | Original class | Now |
|---|---|---|
| **F1** consumer never verifies the corpus it loads | MUST FIX BEFORE FIRST TRAIN | **RESOLVED** (§R.1) |
| **F2** `PI_STRIP` is two independent literals | MUST FIX BEFORE FIRST TRAIN | **RESOLVED** (§R.2) |
| **F3** training checkpointing never invoked | SHOULD FIX BEFORE LONG RUNS | **RESOLVED, and three further defects found while wiring it** (§R.3) |
| **F4** loader materialises ~2.73 GB | MUST VERIFY IN SMOKE | **UNCHANGED — still delegated to the real smoke**, which now measures it (§R.4) |
| **F5** Stage-6 counters blind under workers>1 | NON-BLOCKING (observability) | **UNCHANGED — deliberately untouched** (§R.5) |
| **F6** stale compiled PDF | DOCUMENTATION ONLY | **UNCHANGED — release work** (§R.6) |

**No scientific constant changed.** 32 re-compared against `aa49785`: none moved.
`unmark/stage1/protocol.py` is **byte-unchanged**.

### R.1 F1 — the consumer now verifies the corpus it loads

`checkpoint.verify_prepared_corpus(prepared_dir, completion_dir) -> VerifiedCorpus`.
**One authoritative path**: it reuses the same `COMPLETE.json`
`write_completion_marker` produces, the same `verify_file`, and the same
relative-name artifact binding the `ALREADY_COMPLETE` path proved on real data.
No second hash verifier was written.

It fails closed unless, in order: the marker exists and parses; `complete` is
true; the schema matches; the identity is the **locked** protocol
(`require_locked_protocol`, new — `require_match` compares two identities, but a
consumer has only one and the repository's own constants to compare it to); every
bound artifact re-hashes correctly **from disk**; both `chunks.jsonl` and
`manifest.json` are actually bound; `load_manifest`/`require_compatible` accepts
the manifest; and the manifest's counts equal the marker's counts.

**On the portable-path question §R asked about**: `COMPLETE.json` binds artifacts
by **relative name plus size and sha256**, never by absolute path, so a prepared
corpus restored to a different root verifies identically — asserted by
`test_a_relocated_prepared_corpus_still_verifies`. No `/content/`,
`/content/drive/` or `aa49785eadcb` appears anywhere in library code. Because the
real run persists the payload and the checkpoint under different roots, the
completion directory is an **explicit** CLI argument (`--completion-dir`,
defaulting to `<prepared-corpus>/_checkpoint` for the co-located case) rather
than inferred.

**Provenance now records a verified value.** `execute_stage` reads
`verified.chunk_membership_digest`, which can only exist once the bytes have been
checked, instead of `manifest["counts"]["chunk_membership_digest"]`. All three
training commands call `_verified_corpus(args)` **before anything else**, and so
does `smoke_check` — before the model is loaded.

**Tests** (`tests/test_stage1_corpus_verification.py`, **28**): acceptance of a
valid synthetic COMPLETE-bound corpus and of a relocated one; and rejection of a
missing marker, a malformed marker, an incomplete prepare, **one flipped bit** in
`chunks.jsonl`, a truncated payload, an appended payload, either bound file
missing, a marker that fails to bind either artifact, a foreign manifest, a
marker whose counts disagree with the manifest, a marker with no membership
digest, and **ten** identity fields off the locked protocol. Plus two structural
tests that provenance uses the verified digest and that every training command
verifies first.

### R.2 F2 — one authoritative `PI_STRIP`

`contracts.py` now does `from unmark.stage1.protocol import PI_STRIP`. Exactly
one numeric definition remains in the repository:
`unmark/stage1/protocol.py:89`.

**Direction, and why.** `protocol.py` is authoritative because it declares itself
"the locked Stage-1 protocol. One source of truth" and already *derives* rather
than types — its seven role seeds come from `derive_seeds`. `contracts.py` is a
mechanism module (policies, config objects), so it consumes the locked value.
The direction is **acyclic**: `protocol` imports only
`unmark.evaluation.profiling`, which reaches `evaluation.contracts` and
`orthography` and never returns to `stage1.contracts`.

**Value unchanged: `PI_STRIP == 0.25`.** `protocol.py` is byte-unchanged; only
`contracts.py`'s duplicate literal was replaced by an import. Editing the
authoritative definition now necessarily changes **both** corruption behaviour
and recorded manifest identity, which is the property F2 required.

### R.3 F3 — training checkpoints, and three defects found while wiring them

**No new cadence was chosen, and no new decision entry was created.** Re-reading
the decision log settles it: **D-S1B-004 already locks** "eval cadence every
`500` updates ... **best + last checkpoint persistence**; optimizer and
corruption `visit` state persistence". F3 was therefore a pure
**implementation-wiring defect** — the machinery existed and `execute_stage`
never invoked it — not an undecided operational question.

The implementation now follows that locked contract:

| D-S1B-004 clause | Implementation |
|---|---|
| eval cadence every 500 updates | `CHECKPOINT_EVERY_UPDATES = EVAL_EVERY_UPDATES`, so the two cannot drift |
| **best + last** persistence | `training-checkpoint-last.pt` (the resume target) **and** `training-checkpoint-best.pt` |
| optimizer state persisted | `optimizer_state` in the payload |
| corruption `visit` state persisted | `sampler_state` carries `seed`, `visit`, `position`, corpus digest |
| deterministic streams and experiment identity preserved | `verify_checkpoint` refuses on ten identity fields |

**"Best" is the already-locked rule, not a new comparison.** The loop asks
`selection.select_checkpoint(result.points)` — lowest score, then lower
`d_clean`, then earliest update — and writes `best` only when that rule names the
current update. There is exactly one definition of "best" in the repository.

**Contents.** Schema version, provenance, adapter state, optimizer state,
`global_update`, sampler state (`seed`, `visit`, `position`, corpus digest),
`cap`, `budget_limited` and **`points`**.

**Atomicity.** temp → flush → `fsync` → `os.replace` → directory fsync. A crash
mid-write leaves the previous checkpoint intact; no `.tmp` survives publication.

**Identity binding.** `verify_checkpoint` refuses on run seed, corruption seed,
LR, `r`, corpus manifest digest, backbone checkpoint and revision, protocol
version, precision — and now **repository HEAD**.

**CLI.** `--resume` on `lr-pilot`, `r-phase1`, `final-main`. Without it an
existing `--output-dir` is still refused, so a fresh run cannot overwrite
evidence and the operator is never asked to delete it. With it the directory must
exist and every checkpoint is identity-verified before use. **Nothing
auto-resumes.** Per-run namespace `run-<label>/_checkpoint`.

**Three further defects, found only because the wiring forced the state to be
examined:**

1. **`points` were read on resume but never written.** `train_run` restored
   `resume.get("points", [])`; `checkpoint_payload` never stored them. A resumed
   run would have silently lost every validation measurement before the
   interruption — and **selection consumes exactly those**. Now persisted and in
   `REQUIRED_CHECKPOINT_KEYS`.
2. **`repository_head` was recorded but never compared.** `require_match`
   checked nine fields and omitted it, so a checkpoint from one commit could
   resume under another whose trainer or corruption code had changed. Now
   compared.
3. **The 20 000 → 40 000 continuation did not continue.** It called `train_run`
   again with `resume=None`, rebuilding the optimizer and restarting the sampler
   at `visit 0`, while the locked budget rule requires preserving "adapter,
   optimizer, `visit`, cursor and streams". It now resumes from the checkpoint
   the first leg writes at exactly `cap`. **This restored an already-locked rule
   rather than changing one** — had it required changing the rule, this audit
   would have stopped and said so.

**Equivalence evidence.** `tests/test_stage1_training_resume_state.py` (**23**,
torch-free so it runs in the ML-free venv) proves interrupted + resumed equals
uninterrupted by **exact** equality of the numeric state, the full
`(chunk_id, visit)` sequence consumed, the update count, `visit`, the sampler
cursor and the validation history — at **five** interruption points: on a
validation boundary, after a pass boundary is crossed, several passes in, one
boundary before the end, and at the final update. The chunk count is deliberately
not a multiple of the batch size, so a pass boundary falls *inside* a batch. A
mutation test proves the comparison actually catches a reset cursor. The payload
round-trips through JSON, so serialisation is exercised.
`tests/test_stage1_training_resume.py` (**torch-gated**) adds exact adapter-tensor
and optimizer-state equality with a real `AdamW`; it skips locally and runs in
Colab, where it needs no model download.

### R.4 F4 — still delegated, and now measured

`load_prepared_chunks` is unchanged: **correctness first**, and the observed
runtime has ~176.88 GiB. `smoke_check` now reports load time, the train/dev chunk
counts and process RSS after the load, so the smoke produces the measurement that
would justify — or not — any later optimisation.

### R.5 F5 — deliberately untouched

Stage-6 behaviour and logging are unchanged. `chunking.py`, `corpus.py`,
`lengths.py`, `parallel.py` and `decompose.py` are **byte-unchanged**.

### R.6 F6 — unchanged

The compiled PDF is still stale and remains release work, not a training
blocker. Not regenerated.

### R.7 Tests

| Suite | Result |
|---|---|
| `tests/test_stage1_corpus_verification.py` | **28 passed** (new) |
| `tests/test_stage1_training_resume_state.py` | **23 passed** (new) |
| `tests/test_stage1_training_resume.py` | **skipped locally** (torch-gated; runs in Colab) |
| `tests/test_stage1_pretrain_audit.py` | 9 passed (2 F3 gap-markers **replaced**, §R.8) |
| `tests/test_stage1_runner_contract.py` | 43 passed |
| `tests/test_stage1_schedule.py`, `test_stage1.py` | 90 passed, 14 skipped |
| **Full repository** | **3 406 passed, 98 skipped, 0 failed** (was 3 354 / 97) |

### R.8 Two tests replaced, not weakened

`test_training_checkpoint_payload_is_currently_never_persisted` and
`test_execute_stage_does_not_yet_resume` **asserted the F3 gap**, and the first
said in as many words that it "will fail the moment persistence is wired, at
which point resume equivalence needs its own evidence". It duly failed. Both are
replaced by the post-hardening contract — persistence is wired, resume is
explicit, the continuation genuinely continues — so the gap cannot silently
reopen. A third was added for the continuation defect.

### R.9 Boundaries

**No model weight downloaded, no encoder loaded, no forward pass, no backward,
no optimizer step, no training.** The equivalence tests use a tiny synthetic
module and a pure-Python accumulator. Official UIT-VSFC TEST remains sealed and
unreachable. The prepared corpus was not modified and Stage 6 was not rerun.

### R.10 Hardening self-audit

| # | Question | Answer |
|---|---|---|
| 1-2 | F1 repaired; does training verify the payload it consumes? | **yes** — before model load, in all three commands and the smoke |
| 3 | COMPLETE.json used by the authoritative existing verifier? | **yes** — same marker, same `verify_file`, no second verifier |
| 4 | Can modified/truncated/swapped data train? | **no** — 1-bit, truncation, append, substitution and absence all refused |
| 5 | Provenance records a verified identity? | **yes** — `verified.chunk_membership_digest` |
| 6-9 | F2 repaired; one authoritative value; still 0.25; shared source? | **yes / yes / yes / yes** |
| 10-13 | F3 repaired; persisted; crash-safe; explicit fail-closed resume? | **yes** — implementing **D-S1B-004**'s locked best+last contract; no new decision |
| 14-18 | Model, optimizer, sampler/cursor/visit, update count, validation schedule restored? | **yes** — all asserted |
| 19-20 | Run identity and prepared-corpus identity verified on resume? | **yes** — including repository HEAD, newly enforced |
| 21-22 | Interrupted ≡ uninterrupted; crosses validation/pass boundaries? | **yes** — exact equality at 5 points, pass boundary inside a batch |
| 23 | Stage-6 code untouched? | **yes** — byte-unchanged |
| 24-25 | F4 left for the smoke; F5 non-blocking? | **yes / yes** |
| 26-27 | No scientific constant changed; TEST inaccessible? | **yes** — 32 compared, none moved |
| 28-32 | No encoder, forward, backward, optimizer step, training? | **yes** |
| 33-35 | Full tests pass; `git diff --check` clean; nothing staged? | **yes** — 3 406 / 98 / 0 |
| 36-37 | Audit 030 updated in place; decision log updated only where needed? | **yes** — §R only. **No decision entry created**: D-S1B-004 already locked the contract, and duplicating it would have been worse than leaving it |
| 38 | Real no-update smoke still required? | **YES — it is the next step, and it is unchanged (§N)** |

---

## S. ADDENDUM — F3 WAS A WIRING DEFECT, AND THE VALIDATION-COST CONCERN IS UNMEASURED

**Date:** 2026-08-23. Two corrections and one preparation.

### S.1 Correction: no new cadence decision was needed, and the one I made is withdrawn

§R first recorded a new decision entry, **D-S1B-014**, choosing a checkpoint
cadence. **That was wrong and it has been removed.** Re-reading the decision log
settles the question: **D-S1B-004** already locks, in its
`LOCKED — A-PRIORI ENGINEERING` tier,

> "eval cadence every `500` updates; ... **best + last checkpoint persistence**;
> optimizer and corruption `visit` state persistence"

So F3 was never an undecided operational question. It was a pure
**implementation-wiring defect**: the machinery existed, `execute_stage` never
invoked it. Creating a second entry would have duplicated a locked decision and,
worse, made it look as though the cadence had been chosen here.

`docs/spec/decisions.md` is therefore **unchanged by this hardening**, and every
reference in the code and in §R now cites **D-S1B-004**.

### S.2 Correction: the implementation was incomplete against that locked contract

Writing only a single overwriting checkpoint satisfied "last" and **not
"best"**. Now both are persisted:

| D-S1B-004 clause | Implementation |
|---|---|
| eval cadence every 500 updates | `CHECKPOINT_EVERY_UPDATES is EVAL_EVERY_UPDATES` — asserted identical, so they cannot drift |
| **best + last** persistence | `training-checkpoint-last.pt` (the resume target) **and** `training-checkpoint-best.pt` |
| optimizer state persisted | `optimizer_state` |
| corruption `visit` state persisted | `sampler_state` — `seed`, `visit`, `position`, corpus digest |
| deterministic streams, experiment identity | `verify_checkpoint`, ten fields incl. repository HEAD |

**"Best" is the already-locked rule.** The loop calls
`selection.select_checkpoint(result.points)` — lowest score, then lower
`d_clean`, then earliest update — and writes `best` only when that rule names the
current update. A second comparison written inside the trainer is how a
selection rule silently forks, so an AST test requires `select_checkpoint` to
appear in `train_run`'s call graph.

### S.3 Nothing was changed in response to the validation-cost concern

The external concern is **plausible but unmeasured**, so this commit changes
**none** of: the 500-update eval cadence, the full held-out dev set, the four
validation conditions, or the selection metric. No validation subset, no
1 000-update cadence, no reference-embedding cache, no retrieval-based exclusion
rule.

**Any future change to dev subset, eval cadence, or a hard collapse-exclusion
threshold must be an explicit pre-training scientific decision recorded in the
log — never a silent performance optimisation.**

### S.4 A finding the measurement request surfaced: the recorded lengths are not recorded

Requirement 1 asks for a profile "using **RECORDED** lengths". **They are not in
the artifact.** `chunk_record` persists `chunk_id`, `document_id`, `partition`,
`chunk_index`, `text`, `source_start`, `source_end`, `source_shard` — and
nothing else. Both `reference_length` and `base_length` were computed and
*enforced* during Stage 6 (that is how `overflow_count = 0` was established) but
never written down.

Consequences, stated plainly rather than worked around:

* a token-length profile can only be **recomputed** with the pinned tokenizer,
  which is a full re-tokenization pass over 2 633 067 chunks;
* adding the lengths to the payload would change the artifact bytes and require
  **re-running Stage 6**, which is forbidden and not worth it;
* so the profiler reports **two** things and labels them differently:
  **exact and free** (character lengths, chunks-per-parent — from the payload,
  every chunk) and **recomputed on a sample**, marked
  `recomputed_not_recorded`.

Classification: **DOCUMENTATION / FUTURE-ARTIFACT**, non-blocking. If a future
prepare is ever re-run for another reason, persisting the two lengths would make
this free.

### S.5 The measurement tool

`scripts/stage1_pretrain_measurements.py` (new). It **verifies the corpus first**
(F1), **streams** the payload rather than materialising it (so the profile does
not reintroduce F4), constructs **no optimizer**, and calls no `backward`,
`step`, `zero_grad`, `AdamW`, `build_optimizer`, `train_run` or `execute_stage` —
AST-asserted. `--profile` needs no model at all.

| Requirement | How |
|---|---|
| 1 profile, train/dev separately | `--profile`: p25/50/75/90/95/99/max, fractions ≤32/≤64/≤128, chunks-per-parent, per partition. Character lengths exact over all 2 633 067 chunks; token lengths via `--tokens` on a sample, labelled recomputed |
| 2 full validation wall-clock | ~~`--validation`~~ **NOT DELIVERED — see §T.1.** The mode loads the real encoder but times only condition *preparation*; it runs no forward pass. The measurement remains outstanding |
| 3 clean-reference h(x) timing | **NOT DELIVERED — see §T.1.** No forward pass runs, so no h(x) timing exists |
| 4 collapse diagnostics | the locked per-batch SD of `h'` is retained unchanged in `MonitorWindow`; self-retrieval metrics are **not** added in this commit (§S.6) |

### S.6 Reference-embedding caching — semantics assessed, implementation deferred

**Asked for: an assessment, not an implementation.** Here it is.

**Is it semantically valid?** For the clean reference `h(x)` — **yes, in
principle**. The encoder is frozen and in `eval`, the held-out realisation is
built **once** and reused by every candidate, and the reference pathway depends
only on `canon(x)` and the pinned tokenizer. None of that varies across
candidates or across updates, so `h(x)` for a given dev chunk is a constant of
the experiment, not of the run.

**What a cache identity would have to bind**, or it silently serves the wrong
vectors:

| # | Field | Why |
|---|---|---|
| 1 | `chunk_membership_digest` (verified) | ties the cache to *this* prepared corpus |
| 2 | chunk id **and** a digest of the chunk text | a chunk whose text changed must miss |
| 3 | `ENCODER_CHECKPOINT` + `ENCODER_REVISION` | different weights, different `h(x)` |
| 4 | Transformers version | tokenizer or model-code changes move the vectors |
| 5 | `MAX_LENGTH`, `RAW_BASE_POLICY` | change the input the encoder sees |
| 6 | pooling identity (`attention_masked_mean_non_special`) | changes what `h` *is* |
| 7 | `PRECISION` (`fp32`) | fp16/bf16 would give different numbers |
| 8 | condition (`FULL`/`P50`/`P100`/`STRIP_ALL`) + `VALIDATION_CORRUPTION_SEED` + visit | only `FULL`'s clean reference is candidate-invariant; the corrupted conditions depend on the corruption draw |
| 9 | repository HEAD | orthography or pooling code can change under it |

**Deferred deliberately.** The benefit is unquantified: §S.5 measurement 3 is
exactly what would quantify it. A cache that is wrong is far worse than a
validation pass that is slow, and nine identity fields is a real surface. **Wait
for the timing.**

### S.7 Collapse diagnostics — retained, not extended

The already-locked per-batch SD of `h'` in `MonitorWindow` is **unchanged**.
Self-retrieval Recall@1 / MRR / positive-vs-hardest-negative margin are *not*
added here: they were offered as optional, they are diagnostic-only, and adding
a metric to a monitoring surface immediately before a phase boundary invites it
being read as a gate. **No automatic exclusion threshold exists, and none is
proposed.** If such a rule is ever wanted it must be an explicit recorded
decision.

### S.8 Tests

| Suite | Result |
|---|---|
| `tests/test_stage1_pretrain_measurements.py` | **6 passed** (new) |
| `tests/test_stage1_training_resume_state.py` | **26 passed** (+3 for best/last) |
| `tests/test_stage1_corpus_verification.py` | 28 passed |
| `tests/test_stage1_pretrain_audit.py` | 9 passed |
| **Full repository** | **3 415 passed, 98 skipped, 0 failed** |

### S.9 The smoke, restated with the measurements

Run **after** this commit is reviewed, in the order below. **Step 1 requires no
model. Steps 2 and 3 are real model execution — no-update, but the encoder is
loaded.** Step 2 is **incomplete** (§T.1) and must not be read as validation
cost.

```bash
# 1. Descriptive profile -- no model, streams the payload.
python scripts/stage1_pretrain_measurements.py --profile --tokens \
    --prepared-corpus /content/unmark-stage1-prepared-aa49785eadcb \
    --completion-dir  /content/drive/MyDrive/UNMARK/stage1-checkpoints/aa49785eadcb

# 2. INCOMPLETE (§T.1): loads the real encoder but times condition preparation
#    only -- no forward pass. NOT validation wall-clock.
python scripts/stage1_pretrain_measurements.py --validation \
    --prepared-corpus /content/unmark-stage1-prepared-aa49785eadcb \
    --completion-dir  /content/drive/MyDrive/UNMARK/stage1-checkpoints/aa49785eadcb

# 3. The no-update model smoke itself (§N), unchanged.
python scripts/stage1_runner.py smoke \
    --prepared-corpus /content/unmark-stage1-prepared-aa49785eadcb \
    --completion-dir  /content/drive/MyDrive/UNMARK/stage1-checkpoints/aa49785eadcb \
    --revision 01daacda68afe13d83023d16ec647239e344a1e6 \
    --repository-head "$(git rev-parse HEAD)"
```

**The validation timing decides nothing by itself.** It produces the number the
cost concern has so far lacked. If it turns out to be material, the response is
an explicit recorded decision — not a quiet change to cadence, dev subset,
conditions or metric.

---

## T. FINAL CONSISTENCY CHECK — A BLOCKER IN §S's OWN MEASUREMENT TOOL

**Date:** 2026-08-23. Two points were checked before commit. The first found a
real defect, in work this audit itself added.

### T.1 BLOCKER — `--validation` does not measure validation

§S.9 said "Steps 1-2 need no model" and §S.5 said `--validation` measures
"full four-condition validation wall-clock ... `torch.no_grad` ... peak GPU
memory ... clean-reference h(x) timing". **Both were false.** Traced through the
implementation and confirmed on the call graph:

| Claim | Reality |
|---|---|
| "needs no model" | **False.** It calls `build_objective(revision)` → `AutoModel.from_pretrained` — it loads the real pinned encoder |
| "validation wall-clock" | **False.** It times only `prepare_condition_batch` — tokenisation and corruption preparation |
| forward passes | **Zero.** `evaluate` is imported and **never called**; the objective is never invoked |
| `torch.no_grad` | **Absent** |
| peak GPU memory | **Meaningless** — nothing is placed on the GPU, so the figure would be ~0 |
| clean-reference h(x) timing | **Does not exist** |
| `--reference`, `--collapse` | **Advertised in the module docstring but never implemented** — no such flags exist |

This is precisely the substitution the measurement request forbade:
*"Do not replace real validation timing with tokenisation-only or synthetic
timing."* §S.5's table asserted a capability that the code does not have.

**Classification: BLOCKER.** Not because anything scientific is wrong — no
protocol value moved, no training path is affected, and the F1/F2/F3 repairs in
§R are untouched and still hold — but because an audit that advertises a
measurement it cannot take is worse than one that takes none. The cost concern
would have been "answered" with a number that measures tokenisation.

**Deliberately NOT rewritten here.** Making `--validation` real means batching
the dev set through the objective, device placement, `torch.no_grad`, and CUDA
synchronisation for honest wall-clock. That is a design change requiring review,
not something to slip into a consistency check minutes before a commit. What was
done instead is to **make the tool and this audit tell the truth**:

* the mode is renamed in its output to
  `condition_preparation_only_NOT_validation_wall_clock`, with
  `incomplete: true`, `forward_passes: 0` and an explicit warning field;
* its docstring states plainly that it must not be read as validation cost;
* the module docstring no longer advertises `--reference` or `--collapse`.

**Consequence: the validation-cost concern remains unmeasured**, exactly as it
was before §S. Measuring it is now a named, separate piece of work.

### T.2 The recomputed token profile is reproducible

Verified against the implementation, not the prose:

| Property | Value |
|---|---|
| sample size | `--sample`, default **20 000 per partition** (a cap) |
| selection method | every **97th** payload line, in document order, until each partition's cap fills |
| deterministic? | **Yes** — fixed stride over a fixed line order; no RNG, no seed, no dependence on process, environment or iteration order |
| seed | **None**, and none needed |
| chosen from observed statistics? | **No** — a constant stride fixed in advance |
| train / dev separately | **Yes**, but see below |
| tokenizer | `vinai/phobert-base` @ `--revision` (default the locked `01daacda…`) |
| Transformers version | now **recorded in the output** |
| both pathways | **Yes** — `reference_length` and `base_length` |

**Deterministic, so it was not redesigned** — per instruction. What was added is
the *identity*, so the numbers are reproducible from their own report: stride,
requested and **obtained** counts per partition, `deterministic: true`,
`seed: null`, tokenizer checkpoint and revision, whether that revision is the
locked one, Transformers version, and both pathway names.

**One honest caveat, reported rather than fixed.** The counter is shared across
partitions, and dev is ~0.43 % of the payload, so on the real corpus the stride
yields roughly **117 of 11 443** dev chunks while train fills its 20 000 cap.
Reproducible, but the **dev token profile is thin**. The `obtained` field now
surfaces this in the output rather than leaving it to be inferred. Changing it
would be a redesign, which the instruction excluded; if a denser dev sample is
wanted it is a small, explicit change.

### T.3 Everything else re-verified

| Check | Result |
|---|---|
| `EVAL_EVERY_UPDATES` | **500** |
| dev | **5 000** parents / **11 443** chunks |
| `VALIDATION_CONDITIONS` | `('FULL', 'P50', 'P100', 'STRIP_ALL')` |
| selection metric | unchanged |
| `PI_STRIP` | **0.25**, single-sourced (`protocol.py:89` only) |
| best + last persistence | follows **D-S1B-004**; `select_checkpoint` decides "best" |
| new decision-log entry | **none** |
| `docs/spec/decisions.md` diff vs `aa49785` | **empty** |
| Stage-6 implementation | `chunking.py`, `corpus.py`, `lengths.py`, `parallel.py`, `decompose.py` **byte-unchanged** |
| official TEST | unreachable |
| model load / forward / backward / optimizer / training in this check | **none** |
| full suite | **3 415 passed, 98 skipped, 0 failed** |
| `git diff --check` / staged | clean / **0** |

### T.4 Verdict of this check

**A code defect was found, not merely a wording slip**, so this is not a
documentation-only pass. §R's F1/F2/F3 repairs are unaffected and remain sound;
the defect is confined to the §S measurement tool, which now reports its own
incompleteness instead of overstating it.

---

## U. REAL VALIDATION MEASUREMENT PATH REPAIRED

**Date:** 2026-08-23. **§T is preserved above as the record of the defect** —
it was real, it was found before commit, and deleting it would hide that the
tool this audit added had to be repaired.

### U.1 The defect, and the repair

| §T found | Now |
|---|---|
| `evaluate` imported, **never called** | `validation.evaluate` is **called** — the same function `execute_stage` hands to `train_run` as `evaluate_fn` |
| zero forward passes | real encoder **and** adapter forwards, counted |
| no `torch.no_grad` | the authoritative `evaluate` supplies it; the proxy **records** whether grad was enabled and fails closed if it ever was |
| no device placement | model moved to CUDA when available; `--require-cuda` refuses to report a CPU number as a GPU one |
| meaningless GPU peak | `reset_peak_memory_stats` immediately before the recurring region; peak **allocated** and **reserved** reported; process RSS never labelled GPU |
| no clean-reference timing | `reference_representation` timed separately from `adapted_representation` |
| `--reference` / `--collapse` advertised, absent | removed from the docstring; only implemented modes are documented |
| placeholder mode name | deleted; a test forbids its return |

### U.2 One authoritative evaluator — instrumented, not reimplemented

`InstrumentedObjective` is a **transparent proxy**. Every call delegates to the
real objective, so `evaluate` runs its own `objective.eval()`, its own
`torch.no_grad()`, its own batching, distance, pooling and aggregation. The tool
observes; it does not compute.

Asserted structurally: `representation_distance`, `collate_stage1_batch`,
`masked_mean` and `cosine_similarity` are **not** called anywhere in the
measurement script — they come from the evaluator or nowhere.

**One-time vs recurring, the distinction §T's predecessor got wrong.**
`execute_stage` builds `prepared_by_condition` **once** before the run loop and
reuses it for every evaluation. So condition preparation is *setup*; only
`evaluate` recurs, every `EVAL_EVERY_UPDATES = 500` updates. They are timed and
reported separately, and the projection multiplies **only** the recurring
figure — the old tool would have charged a one-time cost 41 times.

Reported: `corpus_verification`, `prepared_corpus_load`,
`one_time_condition_setup`, `recurring_validation_total`, per-condition
estimate, batches per condition, and observed forward-call counts.

### U.3 Clean-reference h(x): honest, because the evaluator already separates it

`evaluate` calls `reference_representation` once per batch **per condition** —
so the clean reference is recomputed four times over, and that is exactly the
candidate-invariant work a frozen cache would remove. Measuring it needed no
rewriting: the proxy times that call separately.

Reported as `reference_forward_seconds` / `adapted_forward_seconds` with call
counts, so the cache's ceiling is a measured fraction rather than a guess.
**No cache is implemented** (§S.6 stands).

### U.4 The no-update boundary, enforced not asserted

Every parameter is hashed **before and after** the full measurement, split into
**trainable** and **frozen encoder** digests, and the report carries
`parameter_hash_before`, `parameter_hash_after` and `parameters_identical`.
`validation_failures` refuses to report PASS unless:

* `forward_passes > 0`;
* all four locked conditions executed;
* grad was never enabled during a forward, and no output required grad;
* **parameters are identical**;
* no optimizer was constructed, no backward, no step;
* CUDA timing was synchronised when on CUDA.

The CLI exits non-zero on FAIL. Structurally, `backward`, `step`, `zero_grad`,
`AdamW`, `build_optimizer`, `train_run`, `execute_stage` and
`save_training_checkpoint` are **unreachable** from the measurement script —
AST-asserted.

### U.5 Profile sampling repaired — `partition-local-stride-v2`

§T.2 recorded that v1 was reproducible but starved dev: one counter shared
across partitions gave dev roughly **117 of 11 443**.

v2 counts **within each partition**, with a stride derived from that partition's
own population: `stride = max(1, population // requested)`. On the real corpus:

| Partition | Population | Requested | Stride | Obtained |
|---|---|---|---|---|
| dev | 11 443 | 20 000 | **1** | **11 443 — the whole population** |
| train | 2 621 624 | 20 000 | **131** | **20 000**, spread across the full order |

Still deterministic and **data-independent**: selection depends only on
partition-local position — never on token length, text, corruption outcome,
observed statistics or labels. No RNG, no seed, and **no `hash()`** (salted per
process; asserted absent from the call graph, not from the prose — the
docstring legitimately mentions it).

The output records `sampling_scheme_version`, `selection_method`,
`deterministic`, `seed: null`, `data_independent`, and per partition the
`population_count`, `requested_count`, `obtained_count`, `stride` and
`complete_population`. Token lengths remain labelled **`recomputed_not_recorded`**
with tokenizer checkpoint, revision, whether that revision is the locked one,
Transformers version and both pathway names — because the prepared artifact
never persisted them (§S.4).

**This profiling is descriptive only.** It affects no training, validation,
selection or scientific decision.

### U.6 Tests

| Suite | Result |
|---|---|
| `tests/test_stage1_measurement_contract.py` | **31 passed** (new, **torch-free** — runs in the ML-free venv) |
| `tests/test_stage1_validation_measurement.py` | **skipped locally**, torch-gated; tiny injected model, no PhoBERT download |
| **Full repository** | **3 446 passed, 99 skipped, 0 failed** |

The fail-closed matrix runs **locally, every run**: zero forwards, each of the
four conditions missing in turn, grad enabled, an output requiring grad,
parameter mutation, optimizer constructed, a backward, a step, and
unsynchronised CUDA timing — each must fail; a CPU report must not be forced to
synchronise. Sampling: dev-below-cap taken whole, train spread over its
population, identical across repeated runs, partition-aware, independent of the
data, full identity recorded.

Three of my own test bugs were caught and fixed while writing these, one of them
the recurring prose-matching defect: `"hash(" not in SOURCE` matched the
docstring that *forbids* `hash()`. It now checks the call graph.

### U.7 What did NOT change

F1, F2 and F3 are untouched and re-verified: prepared-corpus verification before
model load, provenance on the **verified** digest, one `PI_STRIP` (still 0.25,
single-sourced), D-S1B-004's best+last with cadence `is` `EVAL_EVERY_UPDATES`,
explicit fail-closed resume, `points` persisted, `repository_head` verified, and
the 20k→40k continuation genuinely continuing.

**No decision-log entry**, and `git diff aa49785 -- docs/spec/decisions.md` is
**empty**. Stage-6 implementation byte-unchanged. Prepared corpus untouched.
No reference cache, no validation subset, no cadence change, no retrieval
exclusion rule. Every scientific constant re-verified unchanged.

### U.8 Self-audit

| # | Question | Answer |
|---|---|---|
| 1-2 | Real forwards; same authoritative evaluator as training? | **yes / yes** — `validation.evaluate`, instrumented by proxy |
| 3-6 | All 11 443 dev chunks; four conditions; batch 128; seed 19225? | **yes** — all read from the locked constants, none redefined |
| 7-8 | Recurring wall-clock honest; one-time setup separated? | **yes** — projection multiplies the recurring figure only |
| 9-10 | CUDA synchronised; GPU peak real? | **yes** — sync before/after; allocated **and** reserved; RSS never labelled GPU |
| 11 | Clean-reference timing honest? | **yes** — the evaluator already separates the call; timed, not fabricated |
| 12-15 | No optimizer; no backward; zero steps; parameter hashes identical? | **yes** — enforced, fails closed |
| 16 | Can a zero-forward run masquerade as validation? | **NO** — fails closed, tested |
| 17-19 | Dev profile is all 11 443; train deterministic; still `recomputed_not_recorded`? | **yes / yes / yes** |
| 20-25 | F1/F2/F3 intact; D-S1B-004 authoritative; decisions.md unchanged; constants unchanged; Stage 6 byte-unchanged; TEST unreachable? | **yes** to all |
| 26-27 | No model download/load; no local forward/backward/optimizer/training? | **yes** — torch is absent locally; the torch half is Colab-gated |
| 28-30 | Full suite passes; `git diff --check` clean; nothing staged? | **3 446 / 99 / 0**; clean; **0** |
| 31-33 | §T preserved; §U added; no Audit 031? | **yes / yes / yes** |
| 34 | Real Colab no-update smoke still required after commit? | **YES** |

---

**STATUS: PRE-TRAIN HARDENING FINAL PASS — READY TO COMMIT AND RUN REAL NO-UPDATE MODEL SMOKE**
**§T's BLOCKER REPAIRED: `--validation` NOW RUNS THE AUTHORITATIVE `validation.evaluate`**
**REAL ENCODER + ADAPTER FORWARDS, COUNTED; ALL FOUR CONDITIONS; BATCH 128; SEED 19225**
**ONE-TIME CONDITION SETUP SEPARATED FROM RECURRING WORK — THE PROJECTION USES RECURRING ONLY**
**CUDA SYNCHRONISED AROUND TIMING; PEAK ALLOCATED AND RESERVED; RSS NEVER LABELLED GPU**
**CLEAN-REFERENCE h(x) TIMED VIA THE EVALUATOR'S OWN CALL — NO CACHE IMPLEMENTED**
**EVERY PARAMETER HASHED BEFORE AND AFTER; ZERO FORWARDS CANNOT MASQUERADE AS VALIDATION**
**PROFILE SAMPLING IS PARTITION-AWARE: ALL 11 443 DEV, 20 000 TRAIN AT STRIDE 131**
**F1/F2/F3 INTACT; decisions.md STILL UNCHANGED; STAGE 6 BYTE-UNCHANGED; NO CONSTANT MOVED**
**A PASS DOES NOT AUTHORISE TRAINING — THE REAL NO-UPDATE SMOKE IS STILL REQUIRED**

~~**STATUS: PRE-TRAIN HARDENING FINAL CONSISTENCY FAIL — DO NOT COMMIT AS-IS (§T)**~~ **— resolved by §U**
**BLOCKER: §S's `--validation` LOADS THE REAL ENCODER BUT RUNS NO FORWARD PASS**
**IT TIMES `prepare_condition_batch` ONLY — TOKENISATION STANDING IN FOR VALIDATION COST**
**`--reference` AND `--collapse` WERE ADVERTISED IN THE DOCSTRING AND NEVER IMPLEMENTED**
**TOOL AND AUDIT NOW STATE THIS; THE VALIDATION-COST MEASUREMENT REMAINS OUTSTANDING**
**THE F1/F2/F3 REPAIRS IN §R ARE UNAFFECTED AND STILL HOLD**
**TOKEN SAMPLING IS DETERMINISTIC (STRIDE 97, NO SEED); IDENTITY NOW RECORDED IN OUTPUT**
**CAVEAT REPORTED, NOT REDESIGNED: DEV YIELDS ~117 OF 11 443 UNDER THE SHARED COUNTER**
**NO PROTOCOL VALUE MOVED; decisions.md STILL HAS NO HARDENING DIFF; STAGE 6 BYTE-UNCHANGED**

~~**STATUS: PRE-TRAIN HARDENING PASS — READY TO COMMIT, THEN RUN THE REAL NO-UPDATE MODEL SMOKE**~~ **— superseded by §T**
**F3 IS A WIRING DEFECT: D-S1B-004 ALREADY LOCKED CADENCE 500 AND best+last PERSISTENCE**
**THE INVENTED D-S1B-014 IS WITHDRAWN; `decisions.md` IS UNCHANGED BY THIS HARDENING**
**`best` IS NOW PERSISTED TOO, CHOSEN BY THE ALREADY-LOCKED `select_checkpoint` RULE**
**NO CHANGE TO EVAL CADENCE, DEV SET, THE FOUR CONDITIONS, OR THE SELECTION METRIC**
**VALIDATION-COST CONCERN IS PLAUSIBLE BUT UNMEASURED — THE SMOKE WILL TIME IT (§S)**
**REFERENCE-EMBEDDING CACHE ASSESSED AS SEMANTICALLY VALID BUT DEFERRED PENDING TIMING**
**FINDING: reference_length/base_length ARE NOT PERSISTED, SO A PROFILE MUST RECOMPUTE THEM**
**F1 RESOLVED: THE TRAINING CONSUMER NOW VERIFIES THE PAYLOAD IT LOADS, BEFORE MODEL LOAD**
**F2 RESOLVED: ONE AUTHORITATIVE `PI_STRIP`, STILL EXACTLY 0.25, `protocol.py` BYTE-UNCHANGED**
**F3 RESOLVED AS A WIRING DEFECT: D-S1B-004's LOCKED best+last CONTRACT IS NOW IMPLEMENTED**
**NO NEW DECISION ENTRY — THE CADENCE AND PERSISTENCE SEMANTICS WERE ALREADY LOCKED**
**THREE FURTHER DEFECTS FOUND WHILE WIRING F3: `points` NEVER PERSISTED, `repository_head`**
**NEVER COMPARED, AND THE 20k->40k CONTINUATION SILENTLY RESTARTING INSTEAD OF CONTINUING**
**INTERRUPTED == UNINTERRUPTED BY EXACT EQUALITY AT FIVE INTERRUPTION POINTS**
**F4 STILL DELEGATED TO THE SMOKE, WHICH NOW MEASURES LOAD TIME AND PEAK RSS**
**32 SCIENTIFIC CONSTANTS RE-COMPARED AGAINST `aa49785` — NONE CHANGED**
**NO ENCODER, NO FORWARD, NO BACKWARD, NO OPTIMIZER STEP, NO TRAINING; TEST STILL SEALED**
**A PASS HERE DOES NOT AUTHORISE TRAINING — THE BOUNDED NO-UPDATE SMOKE (§N) IS NEXT**

~~**STATUS: PRE-TRAIN REPOSITORY-WIDE AUDIT PASS — READY FOR REAL NO-UPDATE MODEL SMOKE**~~ **— superseded by §R**
**NO BLOCKER FOUND; 2 MUST-FIX-BEFORE-FIRST-TRAIN, 1 SHOULD-FIX-BEFORE-LONG-RUNS, 1 SMOKE-VERIFY**
**F1 THE TRAINING CONSUMER RECORDS A CORPUS DIGEST IT NEVER VERIFIES — MUST FIX**
**F2 `PI_STRIP` IS TWO INDEPENDENT LITERALS; CORRUPTION AND THE MANIFEST READ DIFFERENT ONES — MUST FIX**
**F3 TRAINING CHECKPOINTING IS IMPLEMENTED BUT NEVER INVOKED — BEFORE LONG RUNS**
**F4 THE LOADER MATERIALISES ~2.73 GB AND HAS NEVER OPENED THE REAL CORPUS — VERIFY IN SMOKE**
**F5 STAGE-6 ZERO COUNTERS ARE MAIN-PROCESS BLINDNESS, NOT A BYPASSED LENGTH CONTRACT**
**OFFICIAL UIT-VSFC TEST UNREACHABLE BY ANY ROUTE CONSTRUCTED IN THIS AUDIT**
**CORRUPTION IS STATELESS AND SEED-SEPARATED; VALIDATION CANNOT SEE A TRAINING SEED**
**NO ENCODER LOADED, NO FORWARD PASS, NO OPTIMIZER, NO UPDATE, NO WEIGHTS DOWNLOADED**
**A PASS HERE DOES NOT AUTHORISE TRAINING — THE BOUNDED NO-UPDATE SMOKE (§N) IS NEXT**

---

## V. FIRST REAL NO-UPDATE SMOKE — PROVENANCE ROUND-TRIP FAILURE

**Revision 6.** The first real Colab pre-train smoke was run against the
committed hardening HEAD. It is the first time any of this code has executed on
the real prepared corpus in a GPU runtime. It **did not reach the model**, and
this section records that failure rather than replacing it with the repair.

### V.1 What the real smoke established before it stopped

| Gate | Result |
|---|---|
| Hardening HEAD | `8f07842a1434b40ec4f4ffa2a2681da499fd1fc6` |
| Exact-HEAD verification | **PASS** |
| Clean-repository verification | **PASS** |
| Prepared artifact | `aa49785eadcb` |
| Byte-exact prepared restore | **PASS, byte exact** |
| Persistence manifest verification | **PASS** |
| `COMPLETE.json` / member digest | **PASS** |
| `pytest -q tests/test_stage1_training_resume.py` | **6 failed, 8 passed** |
| Model validation | **NOT RUN** |
| PhoBERT real forward | **NOT RUN** |
| Optimizer | **NOT RUN** |
| Training | **NOT RUN** |

The corpus half of the smoke is therefore **real evidence and it passed**: the
prepared payload restores byte-exactly in a fresh runtime and verifies against
its own manifest and completion marker. The failure is downstream of that, in
the training-resume gate, and it stopped the smoke fail-closed **before** any
model was loaded — which is the behaviour the gate ordering was designed for.

### V.2 The failure

All six failures were the same construction error inside
`test_a_foreign_run_cannot_resume`:

```
mine   = provenance()
theirs = RunProvenance(**{**mine.to_dict(), field: value})
TypeError: RunProvenance.__init__() got an unexpected keyword argument 'lambda_align'
```

The six intended mutations — `run_seed`, `corruption_seed`, `learning_rate`,
`r`, `corpus_manifest_digest`, `repository_head` — therefore **never reached
`verify_checkpoint`**. The test that exists to prove foreign runs are rejected
had never once demonstrated it in a torch-enabled environment.

### V.3 The contract, re-derived

`RunProvenance` (`unmark/stage1/trainer.py`) is a `@dataclass(frozen=True)`.

| | |
|---|---|
| **Constructor fields (10)** | `run_seed`, `corruption_seed`, `learning_rate`, `r`, `corpus_manifest_digest`, `repository_head`, `backbone_checkpoint`, `backbone_revision`, `protocol_version`, `precision` |
| **`to_dict()` keys (12)** | the 10 above **plus** `lambda_align`, `lambda_clean` |
| **The extra two** | **Derived, not stored.** `weights` is a `@property` computing `lambdas_for_r(self.r)`; `to_dict()` splices in `self.weights.to_dict()` |

`lambda_align` is therefore **not an alias and not a compatibility field**. It is
a *derived scientific quantity*, recorded so an artifact states the weights its
objective actually used. Its authoritative definition is `protocol.py:121`:

```
lambdas_for_r(r) = ( S/(1+r),  S·r/(1+r) )     with S = LAMBDA_SCALE_SUM = 2.0
```

so `lambda_align + lambda_clean == S` and `lambda_clean / lambda_align == r`.
`r` and `lambda_align` do have a required mathematical relationship, and it is
one-directional: `r` is the identity, the lambdas follow from it and cannot be
set independently.

**`to_dict()` is artifact serialization, not a constructor round trip.** That is
now stated in its own docstring, and `DERIVED_KEYS` is declared on the class.

### V.4 Is the production path affected? — traced, and no

The decisive question was whether a real persisted checkpoint can ever be
reconstructed with `RunProvenance(**serialized_dict)`. It cannot. The full
lifecycle:

| Step | Code | Direction |
|---|---|---|
| construct | `execute.py:150` — the **only** production construction site | explicit keyword arguments from the *plan*, never `**dict` |
| serialize | `checkpoint_payload` → `"provenance": provenance.to_dict()` | object → dict |
| persist | `save_training_checkpoint` → `_publish` → `torch.save` | dict → disk |
| load | `load_training_checkpoint` → `torch.load` | disk → **raw dict** |
| verify | `verify_checkpoint` → `provenance.require_match(payload["provenance"])` | **dict compared against a freshly constructed identity** |

There is **no `from_dict` anywhere in the repository**, and nothing rebuilds a
`RunProvenance` from an artifact. `scripts/stage1_runner.py:478`
(`_load_selection`) reads result artifacts as plain dicts and touches only
scalars. This is not an omission — it is a **stronger** design than round
tripping: a run's identity comes from its plan and its environment, so a
corrupted or foreign checkpoint can never define which experiment it belongs to,
only fail to match one.

**Classification: CASE A — test-construction bug.** No production path had the
defect. The audit did not assume this from the traceback; it was established by
tracing every construction, serialization and read site.

### V.5 The one real gap the trace did expose

`require_match` compared all **10** constructor fields — no identity field was
unguarded — but the **2 derived keys were serialized into every checkpoint and
never read back by anything**. Under honest production they cannot disagree with
`r` (same pure function, and `repository_head` is compared, so `lambdas_for_r`
itself cannot have changed). A corrupted, truncated or hand-edited artifact
could: it could claim `r = 1.0` while carrying `lambda_align = 99.0`, and no gate
in the repository would have looked. They were the only scientific quantity a
checkpoint carried that nothing ever checked.

`require_match` now validates them, after the identity comparison so that a
foreign `r` is still reported as an `r` mismatch:

* a derived key **missing** → refused;
* a derived key **inconsistent with the artifact's own `r`** → refused as
  *"internally inconsistent"*.

**No scientific value changed.** The relationship being enforced is the one
already locked in `protocol.py`; enforcing a locked formula is not a new
decision, so **no decision-log entry was created** and `docs/spec/decisions.md`
is **byte-unchanged**.

### V.6 Repair

| Layer | Change |
|---|---|
| `trainer.py` | `DERIVED_KEYS` declared; `to_dict()` docstring states the non-round-trippable contract; `require_match` validates the derived keys fail-closed |
| `test_stage1_training_resume.py` | `dataclasses.replace` — the authoritative constructor-level derivation — replaces `RunProvenance(**to_dict())` |
| both resume test files | `pytest.raises(Exception)` → `pytest.raises(TrainerContractViolation)` **plus the mutated field asserted in the message** |
| `test_stage1_provenance_contract.py` | **new, torch-free**: the contract and the lifecycle round trip, running in the ML-free venv on every run |

The `pytest.raises(Exception)` weakness is the reason this survived: it accepts
*any* failure, so a setup error is indistinguishable from the rejection under
test. The torch-free twin in `test_stage1_training_resume_state.py` had the same
weakness and has been tightened even though its construction was already
correct.

### V.7 Re-verification evidence (appended, not replacing V.2)

| Check | Result |
|---|---|
| All six foreign identities reach `verify_checkpoint` | **6/6**, each rejected on its own field |
| Every **10** constructor fields gated | **10/10** rejected with the field named |
| `RunProvenance(**to_dict())` raises `TypeError` | pinned as the **contract** |
| No production `RunProvenance(**…)` | AST-verified across `unmark/` and `scripts/` |
| Lifecycle round trip, dict + JSON | every identity field preserved exactly |
| Lifecycle round trip, real `torch.save`/`torch.load` | added, Colab-gated |
| Derived-key tamper / omission | refused |
| Full lightweight suite | **3 475 passed, 99 skipped, 0 failed** |

Because the six repaired cases live in a torch-gated module, they were also
executed **locally** by running the repaired function body outside that module
against the real `verify_checkpoint` — they touch no torch API — giving 6/6 here
in the ML-free venv. The authoritative torch evidence still comes from the next
Colab rerun.

### V.8 What is still NOT established

The real no-update smoke has **not** passed. It stopped at the resume gate, so
everything downstream remains unevidenced: **no PhoBERT load, no real forward,
no validation, no optimizer, no update, no training**. Stage 6 is
byte-unchanged, the prepared corpus was not touched, and official UIT-VSFC TEST
remains unreachable.

**STATUS: PRE-TRAIN REAL-SMOKE BLOCKER REPAIR PASS — READY TO COMMIT AND RERUN NO-UPDATE SMOKE**
**FIRST REAL SMOKE PASSED EVERY CORPUS GATE AND STOPPED FAIL-CLOSED AT THE RESUME GATE**
**CASE A — TEST-CONSTRUCTION BUG; NO PRODUCTION PATH RECONSTRUCTS PROVENANCE FROM AN ARTIFACT**
**`to_dict()` IS ARTIFACT SERIALIZATION, NOT A CONSTRUCTOR ROUND TRIP — NOW STATED AND PINNED**
**ONE REAL GAP CLOSED: THE DERIVED WEIGHTS WERE SERIALIZED AND NEVER VALIDATED ON READ**
**ALL SIX FOREIGN IDENTITIES NOW REACH `verify_checkpoint`; ALL TEN FIELDS ARE GATED**
**NO SCIENTIFIC CONSTANT CHANGED; `decisions.md` BYTE-UNCHANGED; NO NEW DECISION REQUIRED**
**NO MODEL LOAD, NO FORWARD, NO OPTIMIZER, NO UPDATE, NO TRAINING; TEST STILL SEALED**
**THE REAL NO-UPDATE SMOKE IS STILL REQUIRED — THIS DOES NOT CLAIM IT PASSED**

---

## W. SECOND REAL NO-UPDATE SMOKE — SCIENTIFIC ELIGIBILITY INVENTORY BLOCKER

**Revision 7.** The second real Colab no-update smoke ran at the committed §V
HEAD. It got further than the first: every corpus gate and every repaired test
passed. It then failed closed **before completing validation**, because the
pinned Vietnamese syllable inventory was not present in the fresh runtime.

### W.1 What the real rerun established

| Gate | Result |
|---|---|
| HEAD | `b84b4daac0f2be31266e171d3f56a71611a421e0` |
| Runtime | Python 3.13.15, Torch 2.11.0+cu128, Transformers 4.57.6, RTX PRO 6000 Blackwell, 176.88 GiB RAM |
| Exact HEAD / clean repository | **PASS** |
| Prepared corpus byte-exact restore | **PASS** |
| `COMPLETE.json` / membership digest | **PASS** |
| `tests/test_stage1_provenance_contract.py` | **29 passed** |
| `tests/test_stage1_training_resume.py` | **17 passed** |
| `tests/test_stage1_validation_measurement.py` | **22 passed** |
| Real descriptive corpus profile | **PASS** |
| **Real validation** | **FAIL CLOSED** |
| Exception | `unmark.corruption.eligibility.EligibilityUnresolved` |
| Policy requested / available | `SCIENTIFIC` / **`UNRESOLVED`** |
| Real validation forward | **NOT COMPLETED** |
| Optimizer | **NONE** |
| Updates | **ZERO** |
| Training | **NOT STARTED** |

**§V is confirmed by real evidence.** The 29 + 17 provenance and resume tests
that §V repaired all passed in the torch-enabled runtime — the six foreign-run
cases that previously died in setup now really execute.

### W.2 Classification: **A + C**

**Not B.** The inventory was **already scientifically locked**, exactly and
completely, by **D-B3A-001** (2026-08-19), which closed GAP-2 and D-B2-003:

| | |
|---|---|
| source | `all-vietnamese-syllables.txt` by `hieuthi` |
| gist | `0f5adb7d3f79e7fb67e0e499004bf558` |
| **immutable revision** | `135a4d9716e49a981624474156d6f247b9b46f6a` |
| **sha256** | `78eeb840d50455b14bd564da5aed7318d96468b8deaad5986b77bf5c538315d2` |
| bytes | 116 290 |
| shape | 17 974 raw → 17 954 unique canonical → **2 489 unique stripped forms** (15 465 collisions) |
| license | **NO_EXPLICIT_LICENSE** |

**A — provisioning.** The raw list is deliberately **not committed**: "A public
gist is not a licence, so the raw list is **not committed**. It is fetched into
the git-ignored `.resources-cache/` … and everything scientific fails loudly
without it." A fresh Colab runtime therefore starts without it **by design**.
`active_eligibility_policy()` is *computed* from whether the inventory loads and
verifies, precisely so a missing cache re-arms the guard. **The exception the
smoke hit was the designed contract working, not a defect.**

**C — implementation defects around that locked artifact.** Three, all real:

| # | Defect | Severity |
|---|---|---|
| **W-1** | The check ran **after** the encoder was downloaded and resident. `execute_stage`, `smoke_check` and `validation_timing` all called `build_objective` before anything touched the inventory | operational |
| **W-2** | **D-S1A-008 was never implemented.** Its status line reads "**BLOCKING** for scientific Stage-1 training and the PRE-TRAIN audit", and it requires a scientific run to persist seven named inventory fields. `RunProvenance` recorded none of them | **MUST-FIX, scientific provenance** |
| **W-3** | The manifest declares three counts and a byte size under "*Expected shape (verified on load)*", but `load_manifest` parsed only `expected_entry_count`. The other two counts and `size_bytes` were **checked by nothing** | fail-closed gap |

### W.3 Was a wrong inventory able to reach training? — No

Traced before repairing. `load_inventory` hashes the cached bytes against the
pinned SHA-256 on every load, and `InventoryChecksumMismatch` **subclasses**
`InventoryUnavailable`, so `try_load_inventory` degrades a corrupted cache to
`None` → policy `UNRESOLVED` → `corrupt()` raises. The guard also fires **before**
`get_condition`, so every condition including `FULL` is gated, and no Stage-1
module ever passes `SELF_CHECK`. A wrong or absent inventory **cannot** silently
produce scientific corruption.

What it *could* do was report a corrupted cache as a *missing* one. The preflight
now distinguishes the two.

### W.4 The fetcher, audited statically (not executed for the audit)

| | |
|---|---|
| URL | `https://gist.githubusercontent.com/hieuthi/0f5adb7d3f79e7fb67e0e499004bf558/raw/135a4d97…/all-vietnamese-syllables.txt` |
| Mutable "latest"? | **No** — the revision SHA is *in the path*; no `/raw/main/`, no `HEAD` |
| Hash verified before publication? | **Yes** — the write is unreachable unless the digest matches |
| Advances the pin? | **Never** — mismatch is a hard refusal with an explicit "this is a scientific spec change" message |
| Size guard | 8 MiB ceiling, 60 s timeout |
| Output | `.resources-cache/vietnamese-syllables/all-vietnamese-syllables.txt` (git-ignored) |
| Parsing | `canon()` → `strip_to_base()` → `casefold()` → set membership |

**Reproducible and pinned, not dependent on mutable network state.** One
weakness found and repaired: publication used `path.write_bytes`, so a lost
runtime mid-write could leave a truncated file. It now uses the repository's
temp → fsync → replace → dir-fsync discipline, after the digest check.

### W.5 License boundary

Repository evidence (manifest `license_notes`) states no LICENSE file, no license
field, and no statement in the gist description were found. The architecture
separates the three cases deliberately:

| | |
|---|---|
| **Fetch as an external research input** | done, by the operator, into a git-ignored cache |
| **Redistribute inside this repository** | **avoided** — this is why only provenance is committed |
| **Redistribute inside released artifacts** | not done; the run artifact records *identity*, never content |

This audit gives no legal conclusion beyond what that evidence supports. From an
engineering standpoint the conclusion is unambiguous: the repository intentionally
avoids vendoring, so the reproducibility contract must be an **external-artifact
contract**, and the inventory was **not** vendored by this repair.

### W.6 Repair — all restoration of already-locked decisions

| Layer | Change |
|---|---|
| `unmark/stage1/preflight.py` **(new)** | `verify_scientific_inputs()` — fail-closed, **before** model load. Calls the authoritative `load_inventory`; adds no second verifier. Distinguishes missing / wrong-hash / wrong-shape. Refuses `SELF_CHECK`. Refuses an unresolved policy |
| `unmark/linguistics/inventory.py` | parses and verifies `size_bytes` and the two other declared counts — **W-3**, restoring the manifest's own claim |
| `unmark/stage1/trainer.py` | `RunProvenance.inventory` — **D-S1A-008's seven fields**, compared by `require_match`, so resume rejects inventory drift |
| `unmark/stage1/execute.py` | preflight before `build_objective` in `execute_stage` **and** `smoke_check`; provenance binds the *verified* identity |
| `scripts/stage1_pretrain_measurements.py` | preflight before `build_objective`; the verified identity is recorded in the report |
| `scripts/fetch_vietnamese_syllable_inventory.py` | atomic publication |
| `unmark/stage1/parallel.py`, `unmark/corruption/eligibility.py` | stale comments corrected — see W.7 and W.9 |

**`InventoryIdentity` is exactly D-S1A-008's seven fields and no more.**

An earlier revision of this section said the counts were "not compared against
any constant — because no decision locks one". **That was wrong for the counts**,
and it contradicted W-3 in this same section. D-B3A-001 *does* lock them
(`17,974 raw → 17,954 unique canonical → 2,489 unique stripped forms`), the
manifest declares them, and the repair makes `load_inventory` verify them
fail-closed. The correct statement separates four distinct roles:

| Role | Quantities | Where enforced |
|---|---|---|
| **1. Identity fields** — persisted in `RunProvenance`, compared by `require_match`, so resume rejects drift | the **seven** D-S1A-008 fields: `inventory_schema_version`, `source_name`, `source_author`, `source_revision`, `sha256`, `size_bytes`, `license_status` | `RunProvenance.inventory` |
| **2. Raw artifact identity** — binds the exact bytes | `source_revision` + `sha256` | `load_inventory` hashes the cached bytes on every load |
| **3. Locked shape assertions** — verified fail-closed on load, **not** duplicated into `RunProvenance` because D-S1A-008 does not require them | `size_bytes` (**also** an identity field), `expected_entry_count` **17 974**, `expected_unique_canonical_entry_count` **17 954**, `expected_unique_stripped_form_count` **2 489** | `load_inventory` — all four, since W-3 |
| **4. Report-only derived evidence** — recorded so two runs can be compared; **not** an identity field and **not** compared against any constant, because no decision locks one | `parsed_membership_digest` | preflight `report` only |

`collisions_after_stripping` (**15 465**, also stated in D-B3A-001) belongs to
none of these categories on its own: it is `unique_canonical − unique_stripped`,
so once **3** verifies 17 954 and 2 489 it is *entailed* and cannot differ. It is
reported for readability, not separately checked.

`size_bytes` deliberately appears in both **1** and **3**: D-S1A-008 names it as a
field a run must persist, and the manifest declares it as shape to verify. The
three counts appear **only** in **3** — locked and checked, but not promoted into
the run identity, because `sha256` already makes any different byte sequence a
different identity and D-S1A-008 does not list them.

The only genuinely unlocked quantity is the **parsed-membership digest**. It stays
report-only: promoting it would invent a scientific constant the decision log never
locked, which this audit must not do.

**Inspection confirms the implementation already matches this distinction** —
`InventoryIdentity` has exactly seven fields, `load_inventory` raises on any of the
four shape values, and `parsed_membership_digest` is produced into the report and
compared to nothing. No code behaviour was changed by this correction.

### W.7 Stage 6 does **not** depend on the inventory — verified, no rerun

`unmark/stage1/parallel.py` builds a classifier in each worker, and its comment
claimed *"dropping it would change chunk boundaries"*. **That comment was wrong**,
and it is the kind of wrong that would have implied the prepared corpus is
inventory-dependent. Verified two ways:

* **structurally** — `classifier` is a parameter of `safe_cut_offsets` and is
  **never read in its body** (AST);
* **empirically** — over 2 000+ adversarial strings, the cut set is identical for
  `None`, for a classifier claiming *everything* is Vietnamese, and for one
  claiming *nothing* is.

This matches **D-S1B-013**, which requires the cut predicate to be lexicon-free:
membership would permit a cut **inside a genuine Vietnamese word**. Where a cut
may land is orthographic (unit boundaries, maximal letter runs), not lexical.

> **NO Stage-6 rerun is required. The prepared corpus remains authoritative and
> was not touched.** The inventory is needed for SCIENTIFIC corruption and
> classification, never for the already-produced chunk bytes.

The misleading comment has been corrected.

### W.8 Reconciling Audit 030's earlier "No UNRESOLVED MISMATCH"

**The earlier statement was too strong, and is corrected here rather than
explained away.**

What §A–§Q checked was **implementation semantics**: proposal §4.3's rule against
`classify.py`. That comparison was and remains **correct** — the membership rule
is an exact match, and D-B3A-001 closed GAP-2 in code.

What it did **not** check was whether the mandatory **external artifact** that
rule consumes would be **present** in a fresh runtime, nor whether **D-S1A-008** —
a decision whose own status line says "**BLOCKING for scientific Stage-1 training
and the PRE-TRAIN audit**" — had been implemented. It had not. A repository-wide
pre-train gate should have caught an unimplemented blocking decision; that is the
miss, and it is squarely within the scope §A–§Q claimed.

The precise correction:

> *Implementation semantics are complete and match the proposal. The required
> external scientific artifact was **not provisioned** in a fresh runtime, and the
> blocking decision requiring its identity to be persisted (D-S1A-008) was **not
> implemented**. "No UNRESOLVED MISMATCH" was true of the code and false of the
> pre-train readiness gate.*

§A–§Q's wording is preserved unchanged above.

### W.9 Other stale claims corrected

`unmark/corruption/eligibility.py` still described GAP-2 as open — "the Vietnamese
syllable inventory … does not exist in this repository", and
`VIETNAMESE_SYLLABLE_INVENTORY` was documented "**Not implemented.**" Both have
been false since D-B3A-001. Corrected to describe the real state: implemented, and
`UNRESOLVED` is now a *deployment* condition, not an open scientific gap.

### W.10 The real corpus profile

Recorded as **descriptive evidence only** in
[`docs/experiments/stage1-prepared-corpus-profile-result.md`](../experiments/stage1-prepared-corpus-profile-result.md).
Chunks: train **2 621 624**, dev **11 443**. Chunks per parent: dev mean 2.29
(p50 1, p99 20, max 481); train mean 2.35 (p50 1, p99 22, max 2 479).

Token lengths, recomputed under the locked tokenizer over **all 11 443 dev** and a
deterministic **20 000** train sample:

| Stream | mean | p50 | p75 | p99 | max |
|---|---|---|---|---|---|
| dev reference | 164.20 | 211 | 233 | 256 | 256 |
| dev RAW_BASE | 181.06 | 253 | 256 | 256 | 256 |
| train reference | 164.81 | 211 | 230 | 256 | 256 |
| train RAW_BASE | 182.71 | 254 | 256 | 256 | 256 |

**`over_max_length`: 0.**

This **falsifies** the informal inference that 2.35 chunks/document implies most
training chunks are short. The median document yields one chunk, but that chunk is
commonly **at or near the 256-token ceiling** — median RAW_BASE is 253–254, and
p75 onward is exactly 256. **No chunking or `MAX_LENGTH` change was made.**

### W.11 The Colab provisioning contract — for review, not executed

Claude did **not** access Drive and did **not** run the fetcher for this audit;
the local cache was already provisioned and was only *verified*
(`--verify-only` → present and matching, no network). The cell below is the
proposed contract to review **before** the next rerun:

```bash
# 1. Provision the pinned inventory (network; ~116 KB). Idempotent: if a valid
#    cache is already restored from Drive this verifies and exits 0 without
#    downloading. Refuses on any checksum mismatch and never advances the pin.
python scripts/fetch_vietnamese_syllable_inventory.py

# 2. Verify without network, and fail the cell if it does not match the pin.
python scripts/fetch_vietnamese_syllable_inventory.py --verify-only

# 3. Persist to Drive so later fresh runtimes need no network.
#    NOTE: .resources-cache/ is git-ignored; this copies ONLY the verified file.
mkdir -p "$DRIVE/unmark-resources/vietnamese-syllables"
cp .resources-cache/vietnamese-syllables/all-vietnamese-syllables.txt \
   "$DRIVE/unmark-resources/vietnamese-syllables/"

# 4. On a later runtime, restore BEFORE step 2 and let --verify-only be the gate:
mkdir -p .resources-cache/vietnamese-syllables
cp "$DRIVE/unmark-resources/vietnamese-syllables/all-vietnamese-syllables.txt" \
   .resources-cache/vietnamese-syllables/
```

The restored copy is never trusted on provenance: step 2 re-hashes it against the
pinned SHA-256, and `verify_scientific_inputs` re-verifies again before model
load. Redistribution boundary unchanged — the file goes to the operator's own
Drive, not into git and not into a released artifact.

### W.12 What is still NOT established

The real no-update smoke has **not** passed. Validation never completed, so there
is still **no real forward evidence**, no optimizer, no update, no training. This
section does not claim otherwise.

**STATUS: PRE-TRAIN INVENTORY PROVISIONING REPAIR PASS — READY TO COMMIT AND RERUN NO-UPDATE SMOKE**
**CLASSIFICATION A + C — THE ARTIFACT WAS ALREADY LOCKED BY D-B3A-001; PROVISIONING AND BINDING WERE NOT**
**THE `EligibilityUnresolved` FAILURE WAS THE DESIGNED CONTRACT WORKING, NOT A DEFECT**
**W-2 IS A MUST-FIX: D-S1A-008 WAS BLOCKING AND UNIMPLEMENTED; PROVENANCE NOW BINDS THE INVENTORY**
**AUDIT 030's EARLIER "NO UNRESOLVED MISMATCH" WAS TOO STRONG AND IS CORRECTED, NOT RATIONALISED**
**STAGE 6 IS LEXICON-FREE — VERIFIED STRUCTURALLY AND EMPIRICALLY; NO RERUN; CORPUS UNTOUCHED**
**NO NEW SCIENTIFIC DECISION — NO INVENTORY WAS CHOSEN, NO CONSTANT CHANGED, NOTHING VENDORED**
**NO MODEL LOAD, NO FORWARD, NO OPTIMIZER, NO UPDATE, NO TRAINING; TEST STILL SEALED**
**A THIRD REAL NO-UPDATE SMOKE IS STILL REQUIRED — THIS DOES NOT CLAIM IT PASSED**
