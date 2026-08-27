# Audit 043 — Real Stage-1 MAX_LENGTH Overflow: Root Cause

**Scope:** production-blocking root-cause investigation of the first real Stage-1
training abort. **No Stage-1 production repair applied** — the only code change
in this investigation is the read-only diagnostic scanner and its tests (§8a).
Audits 001–042 untouched.
**Date:** 2026-08-27

---

## 1. Starting HEAD

```
$ git rev-parse HEAD
e495f7417fe41ac97aaaf9c2ea6aba0e89afb3e9      (clean)
```

## 2. Real-Run Evidence

The first W&B-monitored `lr-pilot` entered real scientific training:

```
stage = lr_pilot   candidate = 1/3   lr = 1e-4   r = 1.0   seed = 21230
cap = 20000        batch = 128
```

Confirmed `train_progress` at updates **50, 100, 150, 200, 250**. Last confirmed
state: `global_update = 250`, `position = 32000`, sample-visits 32 000. Then:

```
REFUSED: parallel preparation failed and will not fall back to serial:
base sequence length 257 exceeds max_length 256.
Stage-1 does not truncate, because trimming ids without the channel metadata
would desynchronise the B3 projection (max_length policy is OPEN).
```

**0 validation events. 0 checkpoint events. Output directory empty.**

The message is raised at `unmark/stage1/contracts.py:264` via
`TruncationPolicy.check(len(base_ids), "base sequence")` at
`unmark/stage1/data.py:345`, and re-raised fail-closed by
`unmark/stage1/preparation.py:235`.

## 3. The Apparent Contradiction

Stage-6 recorded `overflow_count = 0` for this exact corpus
(`chunks.jsonl` sha256 `5e4c5e0c…`, membership digest `250859a5…`), and its
chunker actively refuses over-length chunks — `unmark/stage1/chunking.py:281-287`:

```python
if chunk.reference_length > max_length or chunk.base_length > max_length:
    raise ChunkingViolation(...)
```

So Stage-6 checked **both** lengths and admitted every chunk. Yet Stage-1 saw
257. The two therefore measure **different objects**.

## 4. Complete Call Graph and Length Contract

| boundary | object counted | run unit | specials | source |
|---|---|---|---|---|
| Stage-6 chunk emission | `base_length(content[begin:finish])` | — | included | `chunking.py:270-277` |
| Stage-6 length function | `base_runs.length(transforms.base(text))` | **`PHOBERT_RUN = \S+\n?`** | included via `build_inputs_with_special_tokens` | `lengths.py:217`, `build_length_functions` |
| Stage-6 authoritative definition | `len(build_inputs_with_special_tokens(convert_tokens_to_ids(tokenize(transform(x)))))` | whole string | included | `lengths.py` docstring |
| Stage-6 **in-memory** `Chunk` | `text`, `reference_length`, `base_length` | — | — | `chunking.py:270-277` |
| **persisted** `chunks.jsonl` row | `chunk_id`, `chunk_index`, `document_id`, `partition`, `source_end`, `source_shard`, `source_start`, `text` — **no lengths** | — | — | `checkpoint.py::chunk_record` |
| Stage-1 read | `row["text"]` | — | — | `execute.py::load_prepared_chunks` |
| Stage-1 base ids | `project_text(...) -> content_ids` | **`_CHUNK_PATTERN = \S+`** | added after | `data.py:181-215`, `alignment/manual.py:63,147` |
| Stage-1 special tokens | `build_inputs_with_special_tokens(content_ids)` | — | +2 | `data.py::_with_special_tokens` |
| Stage-1 assertion | `truncation.check(len(base_ids), "base sequence")` | — | — | `data.py:345` |

`reference_length` and `base_length` are checked by Stage-6 *before* persistence
and are then **discarded**: `checkpoint.py::chunk_record` writes the eight fields
above and no lengths (see §9e).

Special-token accounting is **identical on both sides** — both call the
tokenizer's own `build_inputs_with_special_tokens`, adding the same constant.
**Hypotheses A, B and C (special-token off-by-one) are therefore excluded**: the
disagreement is in the *content* ids, not the wrapper.

## 5. Root Cause — Two Different BPE Run Units

`unmark/stage1/lengths.py:217` defines the tokenizer's decomposition unit and
states the contract in terms that leave no ambiguity:

```
PHOBERT_RUN = re.compile(r"\S+\n?")

    `PhobertTokenizer._tokenize` decomposes with ``re.findall`` over this same
    pattern and calls ``bpe`` on each resulting run independently.

    The trailing newline is **part of the run**, so ``bpe("gamma\n")`` is not
    ``bpe("gamma")`` -- BPE's end-of-word marker lands on a different final
    character. Composing over plain ``\S+`` instead is exactly the defect that
    produced ``composed 5, exact 7`` at bb50823, and it failed 1708 of 1920 real
    slice cases. This regex is the contract; ``\S+`` must never be used for it.
```

`unmark/alignment/manual.py:63` — the path Stage-1 actually builds ids with:

```python
_CHUNK_PATTERN = re.compile(r"\S+")
```

and `whitespace_chunks` documents it as *"the units PhoBERT's BPE actually
operates on"* — which `lengths.py` establishes, with measurement, is **false**.

`project_text` (`data.py:205-214`) then tokenizes **per `\S+` chunk**:

```python
for chunk in whitespace_chunks(base_text):
    tokens = tuple(tokenizer.tokenize(chunk.text))
    ...
    content_ids.append(piece.token_id)      # one id per aligned piece, 1:1 with tokens
```

So:

* **Stage-6 measures** `len(tokenize(whole base_text)) + specials` — equivalently
  the sum over `\S+\n?` runs, which is the tokenizer's own decomposition.
* **Stage-1 builds** the sum over `\S+` runs + specials.

Chunk text can contain newlines: `chunking.py` cuts at `_SEGMENT = \s+|\S+`
boundaries and persists the raw slice `content[begin:finish]`. Whenever a chunk
contains a newline, the two decompositions differ, BPE's end-of-word marker lands
on a different final character, and the token counts can disagree.

**Deterministic local demonstration** (no transformers, no model, no training) —
a stub tokenizer that decomposes with the real `PHOBERT_RUN` and models the
documented `bpe("gamma\n") != bpe("gamma")` effect:

```
text                          Stage-6 whole  Stage-1 chunks  delta
'aaaa bbbb cccc'                          6               6     +0
'aaaa bbbb\ncccc'                         5               6     +1  <-- DISAGREE
'aa\nbb\ncc\ndd'                          4               4     +0
'aaaaaa\nbbbbbb\ncc'                      5               7     +2  <-- DISAGREE
```

A chunk admitted by Stage-6 at exactly 256 therefore arrives at Stage-1 as
**257** when its newline-bearing runs tokenize longer under `\S+`. That is the
observed failure, exactly.

**Hypothesis F (corruption changes length) is excluded** by production itself:
`data.py:322-337` asserts `clean_base == corrupt_base` and
`clean_content_ids == corrupt_content_ids` before the length check, so
corruption cannot alter the base length. **G (revision/config drift)** is
excluded: one tokenizer instance serves both paths within a run, and Stage-6
recorded the same pinned revision. **H (parallel preparation)** is excluded:
`preparation.py:235` only *re-raises*; the violation originates in
`prepare_example`.

## 6. Why Stage-6 Said Zero and Stage-1 Says 257

Stage-6 was **right about the definition it enforced** — the authoritative
whole-string length — and `overflow_count = 0` is a true statement about that
quantity. Stage-1 does not consume that quantity; it constructs a *different*
tokenization whose length it then checks against the same bound. Nothing is
inconsistent about either number in isolation; the two implementations disagree
about what the tokenizer's run unit is, and only one of them is backed by
measurement (1708 of 1920 real slice cases).

## 7. Offending Chunk Identity — **MEASURED**

The repaired reproducer (§8a) ran successfully on Colab against the real
prepared corpus and the real pinned tokenizer, and measured the first offence:

| field | value |
|---|---|
| `sampler_position` | **33 147** |
| `batch_offset` | 123 |
| `would_fail_at_update` | **259** |
| `chunk_id` | `Mô_đun:Inflation/data#572` |
| `text_sha256_16` | `7d99f2dba18e45c0` |
| `contains_newline` | **true** |
| Stage-6 authoritative base length | **256** |
| Stage-1 realised base length | **257** |
| delta | **+1** |

This is the mechanism of §5, measured on the real corpus: a chunk Stage-6
admitted at exactly the bound, which Stage-1 rebuilds one token longer, and it
contains a newline exactly as the run-unit difference predicts.

**The earlier update-251 derivation was WRONG and is superseded by this
measurement.** That derivation reasoned from "last telemetry was update 250,
one batch per step, no look-ahead" to sampler positions 32 000…32 127. The
measured offence is at position **33 147**, in the batch for update **259** —
eight updates further on. The audit flagged it at the time as *"a derivation,
not a measurement"*, and the measurement is what stands.

**This does not change the count of confirmed optimizer updates.** Telemetry
confirmed **250** updates (events at 50/100/150/200/250); the run then continued
past the last emitted progress event before aborting during preparation for
update 259. **No claim is made that 258 optimizer updates occurred** — 250
remains the telemetry-confirmed historical figure (§2, §11).

Preparation is dispatched per batch and the abort happens while *preparing* the
batch, so the last emitted progress event and the offending position are not
expected to coincide. That is why the reproducer, not arithmetic, is
authoritative here.

## 8. Deterministic No-Training Reproducer

`scripts/stage1_length_contract_scan.py` — read-only. **No model is loaded, no
optimizer is constructed, no backward is called, no scientific artifact is
written, and official TEST is never touched.** It writes only its own diagnostic
JSON report and resume state, both outside the scientific output tree.

Four modes; the first two existed for this section, the last two were added by
§9c:

* `--reproduce` walks the **real** `DeterministicSampler` order at seed 21230,
  batch 128, and reports the first chunk whose Stage-1 realised base length
  exceeds `MAX_LENGTH`, together with its update, batch offset, sampler
  position, visit, `chunk_id`, text sha256 (16 hex), character count, newline
  presence, and both lengths.
* default mode streams the TRAIN partition and reports scope (§9).
* `--validate` is the real-tokenizer equivalence gate (§9c, §9d).
* `--benchmark` times the worker sweep (§9c).

It reports **hashes, ids and lengths only** — never raw chunk or document text.

## 8a. Scanner Failure on First Real Execution, and Its Repair

The reproducer of §8 was run on Colab at diagnostic HEAD `42cf7e91` and **did not
complete**:

```
python -B scripts/stage1_length_contract_scan.py \
    --prepared-corpus /content/drive/MyDrive/UNMARK/stage1-prepared/aa49785eadcb \
    --reproduce --seed 21230 --max-updates 1000

File "scripts/stage1_length_contract_scan.py", line 87, in authoritative_base_length
    from unmark.linguistics import canon
ImportError: cannot import name 'canon' from 'unmark.linguistics'
```

It loaded the real prepared corpus and walked the deterministic sampler first, so
the failure cost minutes and produced **no offender and no scope scan**. No
scientific training occurred.

**Cause.** The first version of the scanner rebuilt the base text itself —
`decompose(canon(text), eligibility_classifier=classifier).base_text` — and
guessed the import. `canon` is exported from **`unmark.orthography`**
(`unmark/stage1/data.py:35`: `from unmark.orthography import Eligibility, canon,
decompose`), not from `unmark.linguistics`, whose exports are inventory and
classifier utilities only.

**Why the earlier local checks missed it.** The import sat **inside a function
body**, so importing the module could not surface it; and the only local
exercises were `--help` and the fail-closed missing-corpus path, neither of which
reaches `authoritative_base_length`. Nothing statically resolved the nested
imports.

**Repair — delegation, not a corrected guess.** `authoritative_base_length` now
takes the `base_length` callable returned by
`unmark.stage1.lengths.build_length_functions(tokenizer)` — the *same* objects
`scripts/stage1_runner.py::_length_functions` builds for the chunker. The
scanner therefore measures precisely the quantity Stage-6 enforced, with **zero
duplicated normalisation, zero reimplemented canon rules and zero approximate
fallback**. It no longer imports or calls `canon` or `decompose` at all.

One subtlety, checked rather than assumed: `ComposedTransforms.base` computes
`decompose(canon(t)).base_text` **without** an eligibility classifier, while
`project_text` passes one. The classifier provably does not change `base_text` —
`decompose`'s docstring states the round-trip is unaffected, and it was verified
on five Vietnamese and newline-bearing samples locally — so both paths base their
ids on the same string. This removes a candidate second mechanism and leaves the
run-unit difference of §5 as the sole one.

**Regression tests added** (`tests/test_stage1_length_contract_scanner.py`, 21
tests, torch-free). The decisive one statically resolves **every**
`from X import Y` in the scanner — function bodies included — against the real
modules, which is exactly what would have caught this before Colab. A mutation
check asserts `unmark.linguistics` has no `canon` while `unmark.orthography`
does. Others prove the scanner no longer normalises anything itself, imports no
torch and calls no `step`/`backward`/`train_run`/`build_optimizer`, executes
`authoritative_base_length` end to end on five synthetic texts (three
newline-bearing) through the real `build_length_functions`, and agrees exactly
with an independently constructed Stage-6 `base_length` on all of them.

Local tests use a stub tokenizer that decomposes with the real `PHOBERT_RUN`,
because `transformers` is absent from the ML-free venv by design. **The
import/API contract is proven locally; the real-tokenizer numbers require
Colab** and are listed in §12.

## 9. Scope Scan

> The description below is the **final** design (§9f). An earlier version of
> this section described computing both quantities for every row and producing a
> complete all-row delta histogram; that is **superseded** — the prepared schema
> persists no `base_length`, so it would require authoritative tokenization of
> all 2 621 624 rows for a quantity that does not change the repair decision.

**FULL TRAIN SCOPE means exact OVERFLOW SCOPE.** For **every** TRAIN row the
scanner computes the exact validated Stage-1 **fast** realised length, and
reports as must-have, exact outputs:

* total scanned;
* count `<= 256` and count `> 256`;
* maximum Stage-1 realised length;
* realised-length histogram and overflow-length histogram;
* the exact overflow offender population and count.

For **every offender** it additionally recomputes the full production
`project_text` length *and* the authoritative Stage-6 length, and requires
`fast == full` and `Stage-6 <= 256`. For **non-offenders** it performs
deterministic Stage-6 spot-checks that re-test that guarantee.

The all-row Stage-6-vs-Stage-1 delta histogram is **nice-to-have only** and is
intentionally reported as `null`, with its reason inline.

**Partial result, from the first real Colab scope scan.** The scan was started
and then **intentionally stopped before completion** because it was far too
slow (§9a). At the point it was stopped:

```
scanned         100 000
over            0
disagreements   92 566
```

Two things follow, and only two. First, the Stage-6/Stage-1 disagreement is
**not rare** — it affected **92.6 %** of the first 100 000 train chunks, which
is consistent with §5 acting on any chunk containing a newline. Second, a
disagreement is **not the same as a violation**: `over = 0` in that window, so
almost all disagreements leave the realised length at or below 256. Only chunks
sitting at the bound become violations, which is exactly what the measured
offender in §7 is.

**What this does and does not settle.** The Stage-6/Stage-1 **length
disagreement** is already demonstrated to be **widespread in the measured
100 000-row window** — 92 566 of 100 000, a 92.566 % disagreement rate — which
is what §5's run-unit mechanism predicts for any newline-bearing chunk. The
mechanism is not a one-chunk curiosity.

What remains **UNKNOWN** is the *violation* scope, which is a different and
narrower quantity:

* the full TRAIN overflow count;
* the full overflow offender population;
* the maximum Stage-1 realised length over the partition;
* the realised-length and overflow-length histograms.

The **all-row delta histogram is deliberately not on this list.** The systemic
Stage-6-vs-Stage-1 disagreement mechanism is already established by the 100 000-row
exact partial result and by the real confirmed offender; computing it for all
2.62 M rows would change no repair decision.

100 000 of 2 621 624 rows is 3.8 % of the partition and the scan was stopped
rather than completed, so **no exact full-corpus percentage is extrapolated
here.** The mismatch mechanism is widespread in the measured 100k window; the
full TRAIN overflow-violation scope remains unknown until the optimised scan
completes (§12 step 4).

## 9a. Scope-Scan Performance Repair

The first scope scan reached 100 000 rows in roughly one hour — about **36 ms
per row**, which projects to **~26 hours** for 2 621 624 chunks. That is not an
acceptable cost for a read-only diagnostic, so the scanner was optimised.
**Correctness was treated as strictly more important than speed**, and every
change is either an exact-equality-proven shortcut or a caching/parallelism
change that cannot alter results.

**Profiled, not guessed.** Measured on a realistic ~200-token Vietnamese chunk:

| stage | cost | share |
|---|---|---|
| `overlay_orthography` + `project_piece` loop | 41.0 ms | **71.3 %** |
| `decompose(canon(text))` | 14.7 ms | 25.5 % |
| `tones` dict — `O(regions x syllables)` | 1.4 ms | 2.5 % |
| `whitespace_chunks` + tokenize + `align_chunk` | 1.5 ms | 2.6 % |
| `character_letter_labels`, `_regions` | 0.4 ms | 0.7 % |

The dominant cost was the **B3 channel projection**, which a *length* does not
need at all, and it was being paid on every one of 2.6 million rows.

**Four changes.**

1. **Reuse the persisted Stage-6 `base_length`** instead of recomputing it. **(SUPERSEDED — no such field is persisted; see §9e.)**
   Prepared rows already carry it. This is a shortcut, so it is verified rather
   than trusted: the prepared corpus is required to exist first; the
   authoritative `base_length` is **recomputed on every `--verify-every` row**
   (default 1000) and any disagreement aborts the whole scan with
   `DIAGNOSTIC_FAILURE`; and an offender is **always** recomputed, so no reported
   violation ever rests on the JSON alone.

2. **An exact count-only Stage-1 fast path** (`RealisedLengthCounter`). It
   computes `len(content_ids)` and nothing else, reusing the production
   transformations — `ComposedTransforms.base`, `whitespace_chunks`,
   `align_chunk`. Three correctness points: it counts `align_chunk(...).pieces`
   rather than `len(tokens)`, because alignment returns `pieces=()` on failure
   and a token count would over-count exactly those rows; it keeps Stage-1's own
   `\S+` unit, so it reproduces the semantics under investigation rather than
   silently correcting them; and `transforms.base` omits the eligibility
   classifier, which provably does not affect `base_text` (§8a). Verified equal
   to the full `project_text` path on 410 inputs including empty,
   whitespace-only, newline-only and leading/trailing-whitespace cases.

3. **Two memos of pure functions** — `ComposedTransforms`'s existing per-segment
   memo, plus a chunk-text → piece-count memo, since the piece count depends on
   `chunk.text` alone. Both are per-process caches that change speed and never
   results, and a test asserts warm and cold counters agree.

4. **Streaming, bounded parallelism, and resume.** Rows are streamed and batched
   (`--batch-size`, default 2000); the 2.2 GB corpus is never loaded into memory
   and never shipped to workers. `--workers N` uses `spawn` with one
   tokenizer/classifier/counter per process. Aggregation is order-independent
   and the offender list is sorted and bounded, so **the report does not depend
   on worker count**. `--checkpoint-every` persists diagnostic aggregates beside
   the report, and `--resume` verifies the diagnostic identity before continuing.
   Only aggregates are persisted — never scientific state.

**Measured speedup**, both sides on identical inputs with the same tokenizer,
10 000 synthetic rows drawn Zipfian from the real pinned inventory:

```
OLD  recompute authoritative + full project_text     3.888 ms/row
NEW  persisted + count-only fast path, 1 worker      0.115 ms/row   (8 674 rows/s)
measured speedup                                     33.71x
peak RSS                                             36 MiB
```

> **SUPERSEDED (§9e).** The projections below assumed the scanner could reuse a
> persisted Stage-6 `base_length`. The prepared row schema does not persist one,
> so they never applied to the real corpus. The redesigned algorithm is
> benchmarked in §9f. They are kept only as a record of the reasoning.

**Projection, explicitly labelled as such.** Scaling the *real observed* Colab
rate of 36 ms/row by the measured 33.71×:

| configuration | projected full 2 621 624-row runtime |
|---|---|
| observed, before | ~26.2 h |
| projected, 1 core | ~0.78 h |
| projected, 4 workers | ~12 min |
| projected, 8 workers | ~6 min |

These are projections from a local benchmark that used a stub tokenizer on both
sides; the ratio is measured, the absolute Colab runtime is not. **The real
figure must be confirmed by the rerun in §12**, and no completed-scan result is
claimed here.

## 9b. Exit Semantics

The reproducer originally returned exit code **1** after *successfully* finding
an offender, and the notebook wrapper read that as a scanner crash. A successful
measurement and a broken diagnostic must never share an exit code, so:

```
0   SUCCESS_NO_VIOLATION      completed; nothing over max_length
1   DIAGNOSTIC_FAILURE        could not complete; nothing measured
2   SUCCESS_VIOLATION_FOUND   completed; a violation WAS found
```

Every mode also prints a machine-readable `"status"` field carrying the same
three names, which is what callers should branch on. **stdout is exactly one
JSON document**; progress, resume notices and the "wrote …" line go to stderr,
so piping to `jq` works. Violations are never hidden behind exit 0.

## 9c. Local-SSD Staging, Real-Tokenizer Validation, Worker Benchmark, GPU

**Local-SSD staging (`--stage-local DIR`).** Google Drive is a network
filesystem, and streaming 2.2 GB of JSON lines from it row by row is dominated by
I/O latency rather than by the length computation being measured. The scanner can
now stage `chunks.jsonl` onto Colab's local SSD, verified at both ends:

1. `verify_prepared_corpus` must succeed on the Drive prepared corpus, so the
   expected size and sha256 come from the re-hashed COMPLETE marker rather than
   from a filename. **This corpus uses split roots**, so `--completion-dir` is
   **mandatory** alongside `--stage-local`: the default is
   `<prepared-corpus>/_checkpoint`, which does not exist here and produces the
   measured 0.11-second `DIAGNOSTIC_FAILURE`;
2. the source file's own size and sha256 are recomputed and must match;
3. the copy is made — the Drive source is opened **read-only and never written**;
4. the local file's size and sha256 are recomputed and must match the source.

Any mismatch aborts with `DIAGNOSTIC_FAILURE`. Copy elapsed time and throughput
(MB/s) are reported. An existing local copy that verifies is reused; one that
does not is recopied. **The local copy is a disposable cache**: it carries no
scientific authority, is never written back, and deleting it costs only the copy.

**Real-tokenizer correctness gate (`--validate`).** Loads the pinned PhoBERT
tokenizer and compares, on real TRAIN chunks, the optimised
`RealisedLengthCounter.length` against the full production `project_text`
realised length.

Selection is **hard-capped and stratified**. `--validate-rows` bounds the
expensive full-`project_text` comparisons, plus **at most one** more: the known
offender, forced in whenever it is present.

Strata are keyed on the **Stage-1 fast realised length**, never on persisted
Stage-6 metadata (there is none — §9e), and assigned **rarest-first**:

| stratum | condition | quota |
|---|---|---|
| `overflow` | Stage-1 fast `> 256` | 15 % |
| `boundary` | Stage-1 fast `>= 248` | 25 % |
| `newline` | contains a newline | 35 % |
| `ordinary` | stride | 25 % |

A row that is both overflowing and newline-bearing counts as `overflow`, so the
measured 92.6 % newline rate cannot crowd the rare strata out. Before any
fast-length work a **sound character pre-filter** is applied — since
`realised <= len(text) + specials`, a row at `>= 248` needs ~246 characters, so
the 200-character filter provably cannot drop a near-boundary or overflowing
row. Each stratum keeps a **deterministic bounded reservoir** whose replacement
index comes from a hash of the row's own `chunk_id`: no RNG state, identical on
every rerun, spread across the whole corpus. See §9d for the superseded
selectors.

It reports `compared`, `mismatch_count`, `max_stage6_vs_stage1_delta`,
`offender_reconfirmed` and a `selection` block (rows streamed, quotas,
availability per stratum, newline and near-boundary counts, whether the offender
was forced in), and reconfirms the §7 offender by hash (`7d99f2dba18e45c0`,
expecting Stage-6 256 / Stage-1 257). **`mismatch_count` must be zero**;
anything else exits `DIAGNOSTIC_FAILURE`. No model, no optimizer, no backward.

**Worker benchmark (`--benchmark`).** Times the **same fixed deterministic real
subset** at each of `--benchmark-workers` (default `1,2,4,8`; add `16` only if
the runtime's CPU and RAM make it sensible). Per configuration it reports rows,
elapsed seconds, rows/sec, CPU utilisation, peak RSS and an **aggregate
digest** over everything that must not depend on worker count. All digests must
be identical, and the fastest configuration that actually completed is
recommended. **8 workers is not assumed optimal** — the recommendation is
whatever measures fastest.

**GPU decision.** Every operation this diagnostic performs is CPU-side:
Unicode canonicalisation, orthographic decomposition, `re` scanning for
whitespace runs, PhoBERT's BPE merge loop over short strings, and dictionary
memo lookups. None has an exact GPU implementation in this repository, and a GPU
rewrite of BPE or of the orthography would be a reimplementation of frozen
scientific transformations — precisely what this diagnostic must not do.

> **GPU not used because no exact GPU implementation provides a measured
> benefit.** The scan scales with CPU cores (`--workers`) instead. A test
> asserts the scanner imports no `torch`/`cupy`/`numba`, so this is a checked
> property and not just a statement.

## 9d. Validation Selection — Bounded and Stratified

**`--validate-rows` is a hard cap on expensive work.** The expensive operation
is the full production `project_text`, and it runs **once per selected row and
never otherwise**. Selection itself streams only cheap metadata — `json.loads`,
a substring test, an integer compare — over the partition, which is the trade
this warrants. The bound is `--validate-rows` **plus at most one**: the known
offender, always included when present even if sampling would have missed it.
So expensive comparisons <= `--validate-rows` + 1, and a test counts real calls
into the production path and asserts exactly that.

**The first implementation was bounded but badly skewed, and was repaired.** It
took "every stride-th row plus every newline-bearing and every near-boundary
row, stop at `wanted`". Measured on a population shaped like the real one
(92.6 % newline-bearing, rare near-boundary rows, offender placed at 60 % depth):

| | before | after |
|---|---|---|
| selected (cap 5000) | 5000 | bounded, <= cap + 1 |
| near-boundary selected | **2** | every one available |
| corpus span covered | **2.7 %** | **99.9 %** |
| known offender included | **no** | **yes**, forced |

The fix was quotas plus deterministic reservoir sampling. **That first fix was
itself superseded**: its `boundary` stratum keyed on a persisted
`base_length >= 248`, which the prepared schema does not carry, so on the real
corpus it selected **zero** near-boundary rows (§9e). The final selector keys
`overflow`/`boundary` on the Stage-1 **fast** length behind a sound character
pre-filter — see §9c for the authoritative description.

The report includes a `selection` block — rows streamed, quotas, rows available
per stratum, newline and near-boundary counts selected, and whether the offender
was forced in — so the composition of any validation run is auditable after the
fact.

## 9e. REAL Colab Validation, and a Falsified Schema Assumption

**The real-tokenizer equivalence gate PASSED.** Run against the real prepared
corpus, split-root COMPLETE marker, pinned PhoBERT tokenizer and local-SSD
staging:

```
compared            = 2886
mismatch_count      = 0
status              = SUCCESS_NO_VIOLATION
official_test_used  = false
```

Offender reconfirmed:

```
chunk_id                         = Mô_đun:Inflation/data#572
text_sha256_16                   = 7d99f2dba18e45c0
Stage-6 authoritative length     = 256
Stage-1 fast length              = 257
Stage-1 full project_text length = 257
delta                            = +1
matches_audit_043                = true
```

So on the measured real validation set the optimised `RealisedLengthCounter`
**is** the full Stage-1 production path, and §7's offender reproduces exactly.

**But the same run falsified an optimisation assumption.** The offender row was
inspected directly; its top-level keys are

```
chunk_id  chunk_index  document_id  partition
source_end  source_shard  source_start  text
```

There is **no `base_length`**. Confirmed in the producer:
`unmark/stage1/checkpoint.py::chunk_record` emits exactly those eight fields.
The chunker computes `reference_length` and `base_length` on its in-memory
`Chunk`, uses them in its own guard, and **discards them**.

Audit 043 previously asserted *"Prepared rows already carry Stage-6
`base_length`."* **That is false**, and two things followed from it:

1. **The near-boundary sampler was inert.** Selection keyed `boundary` on
   `row.get("base_length")`, which is always `None`, so the real validation
   reported `available_per_stratum: {newline: 2 535 745, ordinary: 885}` and
   `selected_near_boundary = 0`. The 256->257 offender appeared **only because
   it was forced by hash**. A real diagnostic bug, not a reporting artifact.
2. **The persisted-length speed shortcut never existed.** The worker recomputed
   the authoritative Stage-6 length whenever the persisted value was not an int
   — i.e. on every row.

> **Every performance projection in §9a that relied on reusing a persisted
> `base_length` is SUPERSEDED.** It was never applicable to this artifact. The
> redesigned algorithm is benchmarked in §9f.

No `base_length` field was synthesised and the prepared corpus was not modified.

## 9f. Redesigned Scope Scan — Exact Overflow Scope

**The scientific quantity actually needed** before repair is: *how many TRAIN
chunks have Stage-1 realised base length > 256?* Nothing more is required to
decide whether the Stage-1 repair is confined to Stage-1 or whether regeneration
must be discussed.

**Is recomputing Stage-6 length on all 2 621 624 rows necessary? No — and the
guarantee is proven from source, not assumed:**

1. `chunking.py:220-221` — the subdivision predicate `fits` requires **both**
   `reference_length(piece) <= max_length` **and** `base_length(piece) <= max_length`;
2. `chunking.py:281-286` — after emission, any chunk exceeding either raises
   `ChunkingViolation`, aborting the run;
3. `manifest.py:99-102` — building the manifest **raises** on a non-zero
   `overflow_count`: *"After correct pre-chunking this must be zero;
   on_overflow=FAIL is a guard, not a policy."*;
4. `checkpoint.py::verify_prepared_corpus` re-hashes `chunks.jsonl` and
   `manifest.json` against the COMPLETE marker.

A Stage-6 run that completed and wrote a COMPLETE marker therefore **cannot**
have admitted a chunk with authoritative `base_length > 256`; the first such
chunk aborts it. The verified bytes are that run's output. **Every admitted
prepared chunk satisfies Stage-6 `base_length <= 256`.**

**The algorithm.**

* **A.** Every TRAIN row: the exact validated Stage-1 **fast** realised length
  only.
* **B.** Count `<= 256` and `> 256`, the maximum realised length, and full
  realised-length and overflow-length histograms.
* **C.** Every offender: recompute the **full production `project_text`** length
  *and* the authoritative Stage-6 length; require `fast == full` and
  `Stage-6 <= 256`. A violation is never reported on the shortcut's word alone.
* **D.** Deterministic Stage-6 spot-checks on non-offenders, which **re-test**
  the guarantee above rather than assuming it. Any spot-check exceeding 256, or
  any offender where `fast != full`, aborts with `DIAGNOSTIC_FAILURE`.

**MUST-HAVE versus NICE-TO-HAVE.** The report now says so explicitly. Must-have
and exact over every row: overflow count, offender metadata with recomputed
Stage-6, max realised length, overflow histogram. Nice-to-have and **not
computed**: the all-row Stage-6-vs-Stage-1 delta histogram, which without a
persisted `base_length` would cost authoritative tokenization on all 2.6 M rows
for a quantity that does not change the repair decision. The disagreement is
already known to be systemic — 92 566 of 100 000 in the earlier exact partial
scan, plus the confirmed 256->257 offender — and `spot_check_delta_histogram`
keeps a deterministic sample of it. The report carries
`all_row_stage6_delta_histogram: null` with the reason inline, so its absence is
explicit rather than an omission.

**FULL TRAIN SCOPE now means exact OVERFLOW SCOPE**, not an all-row delta
histogram.

**Corrected validation sampling.** The `boundary` stratum no longer touches any
persisted field. Strata are rarest-first — `overflow` (fast > 256) 15 %,
`boundary` (fast >= 248) 25 %, `newline` 35 %, `ordinary` 25 % — keyed on the
Stage-1 **fast** length. To avoid measuring 2.6 M rows during selection, a
**sound** character pre-filter is applied first: every BPE token covers at least
one character and `base_text` is a per-character mapping of `canon(text)` with
marks removed, so `realised <= len(text) + specials`; a row at `realised >= 248`
needs ~246 characters, and the filter is set at 200. **No near-boundary or
overflowing row can be filtered out**, and a test asserts it. The cap remains
`--validate-rows` + 1 (the forced offender).

**Measured speedup of the FINAL algorithm** — both sides on identical inputs
with the same tokenizer, 10 000 rows in the **real 8-field schema**:

```
OLD  authoritative every row + full project_text   4.559 ms/row
FINAL fast every row; Stage-6 only on offenders
      and spot-checks                              0.123 ms/row  (8 150 rows/s)
measured speedup                                   37.09x
peak RSS                                           40 MiB
```

This is **measured locally with a stub tokenizer**, so the ratio is measured and
the absolute Colab runtime is not. The real per-row cost must come from the §9c
benchmark on real rows; no runtime projection for the real corpus is claimed
here.

## 9g. Line-Index Convention

The validation report records `line_index = 2098760` for the offender, while a
direct `enumerate(file, 1)` inspection finds it on physical line `2098761`.

**These are the same row.** `line_index` is **0-based** — the index of the row
within the streamed partition — so physical line = `line_index + 1` under
1-based numbering. Reports now carry `line_index_base: 0` explicitly, and the
convention is documented in `_sample_rows`. This is not a second offender.

## 10. Classification

**A — STAGE-1 IMPLEMENTATION BUG. Root cause CONFIRMED on the real corpus
(§7); full TRAIN overflow scope PENDING (§9, §12 step 4).**

The prepared corpus is scientifically valid under the authoritative length
definition, which Stage-6 both documented and enforced on `reference_length`
*and* `base_length`. Stage-1's preparation path constructs its base ids over a
run unit that this repository's own measured contract states *"must never be
used"*, producing a tokenization that is not the tokenizer's whole-string
tokenization, and can be longer.

It is **not C (specification ambiguity)**: the specification is unusually
explicit. `lengths.py` names the unit, cites the failing-case count, and forbids
the alternative. Two modules make contradictory factual claims about PhoBERT's
decomposition, and only `lengths.py`'s is evidenced.

**Minimal scientifically correct repair (NOT applied here).** Make Stage-1's
alignment/preparation operate on the tokenizer's own unit `\S+\n?`, so
`project_text`'s content ids equal `tokenize(whole base_text)` — the quantity
Stage-6 measured. That makes all three agree by construction, and the natural
regression test is the invariant nobody had: for every chunk,
`len(project_text(...)[1]) == len(tokenize(base_text))`.

**This is not a one-line regex swap, and must not be treated as one.**
`align_chunk` fails closed when `tokens[-1].endswith(CONTINUATION_MARKER)`
(`manual.py:36`), and a newline-terminated run is exactly where the end-of-word
marker moves. The alignment, `reconstruct_surface`, and the character-range
mapping that carries the B3 channel metadata all have to remain exact over the
new unit. It needs design and its own audit.

**It also changes the realised input representation** for every newline-bearing
chunk — different base grid, different channel projections. That is acceptable
only because **no scientific Stage-1 run has ever completed**, and it must be
recorded as such rather than slipped in as a bug fix.

## 11. Regeneration, Frozen Science, and the Aborted Run

**Stage-6 regeneration: NOT required, on current evidence.** The corpus
satisfies the definition it was built and verified against. Chunk boundaries,
corpus membership, the membership digest `250859a5…` and `chunks.jsonl`
sha256 `5e4c5e0c…` all stand.

> **If the repair were instead placed in Stage-6** — re-chunking so that the
> *Stage-1-realised* length is the admission criterion — it **would change chunk
> boundaries, corpus membership, the membership digest and the prepared-chunks
> SHA**. That is a scientific-data and provenance change requiring explicit human
> approval and full regeneration. This audit does **not** recommend it, and the
> §9 scan should decide it rather than assumption.

**Frozen scientific configuration: unchanged.** `MAX_LENGTH = 256`, truncation
OFF, overflow FAIL, tokenizer/model revision, seeds, grids, batch, cadence and
budget are all untouched and must stay so. Nothing here proposes raising
`MAX_LENGTH`, enabling truncation, skipping samples or catching the overflow.

**The 250-update attempt: NOT RESUMABLE.**

| | |
|---|---|
| confirmed updates | 250 |
| confirmed sample-visits | 32 000 |
| validation events | 0 |
| checkpoint events | 0 |
| scientific output directory | empty |

The first checkpoint would have been written at update 500. No checkpoint
exists, so `--resume` has nothing to resume; `load_training_checkpoint` returns
`None` and a fresh run would start at 0. No checkpoint may be fabricated from
telemetry or W&B. **The W&B run is historical evidence of an ABORTED diagnostic
attempt and must never be presented as a completed candidate.**

**Observability finding, recorded separately as operational, not causal.** W&B
worked correctly: project `unmark-stage1`, candidate 1 created, metrics 50–250
uploaded, and the failure text captured in `output.log`. The Colab monitor's
stdout did not visibly stream the training output even though W&B received it —
an **operational UX defect in the monitor**, not the cause of the scientific
abort, and deliberately not repaired here (source inspection showed it is not
needed for root-cause reproduction).

## 12. Files Changed, Fingerprints, Tests, and Required Colab Diagnostics

**Files changed** (cumulative, this investigation):
`scripts/stage1_length_contract_scan.py` (read-only diagnostic; created,
repaired per §8a, then optimised per §9a/§9b),
`tests/test_stage1_length_contract_scanner.py` (new), and this audit.
**No scientific module was modified.**

Two different questions, deliberately reported separately.

**(a) Scientific production modules — BYTE-IDENTICAL to the investigation
baseline.** Recomputed after the final scanner repair of §8a:

```
unmark/         48d96b98a32dabd825ed10f583fa670efad053a3604eea8e7bfff3ba5baf5089
configs/        d76156925cc6af8f642c08cb4bdf7eb42af0d7002d9f16b104f03552a3a26a29
requirements/   29e3bd63b85a67dad88f7111df245d6b48450624994bf4da776071a976e0ae50
```

Authoritative check:

```
$ git diff --name-only e495f7417fe41ac97aaaf9c2ea6aba0e89afb3e9 \
      -- unmark configs requirements docs/spec unmark-proposal.md
(empty)
```

Nothing scientific changed at any point in this investigation.

**(b) Execution-relevant repository fingerprint — CHANGED, and expected to.**
Same definition the audit has used throughout (`unmark` + `scripts` + `configs`
+ `requirements`, all files, sha256 of sorted per-file sha256s):

```
before this investigation   66b13c39fa5fe01d9bbeb146708cd2bdc7fb4c786fc0aa29b48cddfc24b2d141
after the OPTIMISED scanner b7905176ecb9dfaa565ee2b2ec8d4c68e6b0a7293e32f1756786a193c2c046e5
  (scripts/ tree hash       1ecf9f49c7e80afa39c1648c53bfc589f77a8b05556ee319e2a53eae078aab66)
```

Superseded intermediate values, kept so the trail is legible: `2386b9e5…` was
the first (never-successful) scanner, `64e659e9…` the §8a repair, and
`cf656997…` the §9a optimisation before the §9c staging/validation/benchmark
facilities were added.

The earlier value recorded here, `2386b9e5…`, described the **first** version of
the scanner and is superseded: that version never ran successfully, and its
`authoritative_base_length` was replaced in §8a.

The fingerprint moves **only** because the read-only diagnostic lives under
`scripts/`, an execution-relevant path for the clean-tree provenance guard. It
must therefore be committed before any scientific run, and it is inert with
respect to training: it loads no model, constructs no optimizer, and calls no
backward.

**Local tests, after the final scanner repair:**

```
$ .venv/bin/python -B -m pytest -q -p no:cacheprovider \
      tests/test_stage1_length_contract_scanner.py
76 passed

$ .venv/bin/python -B -m pytest -q -p no:cacheprovider
3862 passed, 106 skipped in 147.51s (0:02:27)
```

The 55 tests added by the §9a-§9g work prove: the count-only fast path
equals the full `project_text` path on every corpus text including edge cases;
the memo changes speed and not results; a scan over real-shaped rows (no
`base_length`) succeeds; an offender
is always recomputed even with spot checks disabled; aggregation is independent
of arrival order (three shuffles) so the report cannot depend on worker count;
the offender list is bounded and deterministic; a resumed scan equals an
uninterrupted one field by field; a resume from a foreign diagnostic identity is
refused; the three exit codes are distinct and a violation never shares one with
a failure; and the report never carries corpus text. The §9c additions prove: `sha256_of`
matches `hashlib`; staging verifies source then copy and reports throughput;
staging never modifies the Drive source; a source that does not match the
verified identity is refused; an already-verified local copy is reused and a
corrupted one recopied; a scan really reads the staged copy; `--validate`
reports zero mismatches, reconfirms the offender by hash, samples
deterministically, and **fails closed when the fast path is deliberately
broken**; the benchmark reports a measured recommendation; the aggregate digest
ignores timing but not results; and the GPU decision is both stated and true
(no `torch`/`cupy`/`numba` import). §9d adds: `--validate-rows` caps expensive comparisons; the production path runs exactly once per selected row; a skewed 92.6 %-newline corpus no longer starves the rare strata; the known offender is always forced in; selection is stable across runs; the reservoir is bounded and deterministic; and a broken fast path is still detected on a skewed corpus. §9e-§9g add: the fixture matches the production row schema exactly and carries no `base_length`; the scanner never reads `base_length` off a row; a scan over real-shaped rows succeeds; an unverified extra metadata field cannot change scientific counts; offenders carry both a recomputed Stage-6 length and a full `project_text` length that must agree; a Stage-6 guarantee violation fails closed; the all-row delta histogram is explicitly null with its reason; and the character pre-filter cannot miss a near-boundary row.

The 21 diagnostic-scanner tests cover:

* **nested `from X import Y` resolution, function bodies included** — every
  import in the scanner is resolved against the real module. This is the check
  that would have caught the §8a Colab `ImportError` before runtime, and it is
  paired with a mutation check asserting `unmark.linguistics` has no `canon`
  while `unmark.orthography` does;
* **correct delegation** to
  `build_length_functions(tokenizer)` → `base_length`, asserted structurally
  (`build_length_functions` is imported) and behaviourally;
* **newline-bearing examples** — three of the five synthetic texts contain
  newlines, which is the only condition under which the §5 mechanism appears;
* **no duplicated canonicalisation/normalisation** — the scanner neither imports
  nor calls `canon` or `decompose`;
* **no torch, no optimizer, no backward, no `train_run`/`execute_stage`** —
  checked on the AST's imports and call names, not on source text, because an
  earlier draft of this very test matched the scanner's own docstring;
* **exact agreement with an independently constructed Stage-6 `base_length`** on
  all five synthetic texts.

The mechanism demonstration in §5 also runs locally and deterministically.

**What is still Colab-only, and is NOT claimed here.** These tests use a stub
tokenizer that decomposes with the real `PHOBERT_RUN`, because `transformers` is
absent from the ML-free venv by design. They prove the scanner's **import and API
contract** and its delegation — not any number about the real corpus. **No real
tokenizer, real-corpus, offender or scope result is claimed from this
environment.**

**Exact Colab diagnostics still required:**

```
# --- split roots. --completion-dir is MANDATORY with --stage-local: the
# --- default is <prepared-corpus>/_checkpoint, which does not exist here and
# --- reproduces the measured 0.11-second DIAGNOSTIC_FAILURE.
DRIVE=/content/drive/MyDrive/UNMARK
PREPARED=$DRIVE/stage1-prepared/aa49785eadcb
COMPLETION=$DRIVE/stage1-checkpoints/aa49785eadcb
SSD=/content/unmark-stage1-diagnostics
OUT=$DRIVE/stage1-diagnostics/e495f7417fe4

# 1. DONE (§7) -- first offender measured. Rerun only to reconfirm:
python -B scripts/stage1_length_contract_scan.py \
    --prepared-corpus $PREPARED --reproduce --seed 21230 --max-updates 1000
#   expect exit 2 (SUCCESS_VIOLATION_FOUND), position 33147, update 259

# 2. REQUIRED -- real-tokenizer equivalence gate on the FINAL redesigned HEAD
python -B scripts/stage1_length_contract_scan.py \
    --prepared-corpus $PREPARED --completion-dir $COMPLETION \
    --stage-local $SSD --validate --validate-rows 5000 \
    --offender-hash 7d99f2dba18e45c0 \
    --report $OUT/length-contract-validate.json
#   must report mismatch_count = 0 and reconfirm the offender (256 -> 257)

# 3. REQUIRED -- benchmark the REDESIGNED algorithm (§9f), not the superseded one
python -B scripts/stage1_length_contract_scan.py \
    --prepared-corpus $PREPARED --completion-dir $COMPLETION \
    --stage-local $SSD --benchmark --benchmark-rows 50000 \
    --benchmark-workers 1,2,4,8,16 \
    --report $OUT/length-contract-benchmark.json
#   read "recommended_workers"; all aggregate digests must be identical

# 4. REQUIRED -- exact full TRAIN OVERFLOW scope
python -B scripts/stage1_length_contract_scan.py \
    --prepared-corpus $PREPARED --completion-dir $COMPLETION \
    --stage-local $SSD --partition train \
    --workers <recommended_workers> \
    --checkpoint-every 100000 --progress 100000 \
    --report $OUT/length-contract-full-train.json
#   exit 0 = SUCCESS_NO_VIOLATION, 2 = SUCCESS_VIOLATION_FOUND, 1 = DIAGNOSTIC_FAILURE
#   re-run the identical command with --resume to continue after an interrupt
```

**All three are read-only, train nothing, and execute zero scientific optimizer
steps.**

* **Step 2** is *only* the real-tokenizer equivalence gate: optimised
  `RealisedLengthCounter` == full production `project_text`. It settles whether
  the §9f fast path is exact. It says nothing about how many chunks overflow.
* **Step 3** benchmarks worker counts on the redesigned algorithm and
  recommends one.
* **Step 4** determines the **exact full TRAIN overflow scope**: the overflow
  count, the maximum Stage-1 realised length, the realised-length and
  overflow-length histograms, the exact offender evidence (each with a
  recomputed full `project_text` length and a recomputed authoritative Stage-6
  length), and sampled Stage-6 spot-check/delta evidence. It **does not**
  produce a complete all-row Stage-6-vs-Stage-1 delta histogram; that is
  intentionally not computed (§9f).

**Colab status, stated precisely.**

| item | status |
|---|---|
| real-tokenizer equivalence (§9e) | **PASS, retained** — `compared = 2886`, `mismatch_count = 0`, offender reconfirmed at Stage-6 256 / Stage-1 fast 257 / Stage-1 full 257 |
| one validation rerun on the FINAL redesigned committed HEAD | **REQUIRED** as an acceptance/regression gate |
| worker benchmark of the FINAL algorithm | **NOT YET MEASURED ON COLAB** |
| full TRAIN overflow scope | **NOT YET MEASURED** |

The §9e evidence stands on its own — the fast path was proven equal to the full
production path on 2 886 real chunks — but the scanner has since been redesigned
after the schema bug, so that gate must be re-run once on the final HEAD before
its result is carried forward.

§9a's speedup figures are measured locally with a stub tokenizer and are marked
superseded there; §9f's 37.09x is likewise a local measurement with a stub
tokenizer on both sides. **No real-corpus runtime, benchmark result or overflow
scope is claimed anywhere in this audit.**

## 13. Official UIT-VSFC TEST

**SEALED / UNUSED.** Not opened, inspected, mounted, searched, tokenized,
scanned or evaluated. The diagnostic reads only `chunks.jsonl` from the prepared
corpus and records `official_test_used: false`.

## 14. Verdict

**PRODUCTION BLOCKED — CLASS A: STAGE-1 IMPLEMENTATION BUG. ROOT CAUSE CONFIRMED
ON REAL CORPUS. FULL TRAIN OVERFLOW SCOPE PENDING.**
Stage-1 must not be launched until the base-length contract is repaired and the
scope scan has run. No repair to Stage-1 production is applied in this audit.

**Provenance, stated precisely.** No additional training or scientific optimizer
step was executed by this root-cause investigation. The previously aborted real
`lr-pilot` attempt remains recorded with **250 confirmed optimizer updates and
no checkpoint** (§2, §11); this audit neither adds to nor erases that history.

**Now measured (§7).** The first offender is
`Mô_đun:Inflation/data#572` at sampler position **33 147**, batch offset 123, in
the batch for update **259**: Stage-6 **256**, Stage-1 **257**, delta **+1**,
`contains_newline = true`. The source-level mechanism of §5 is therefore
confirmed on the real corpus with the real tokenizer, and the earlier
update-251 derivation is superseded.

**Still UNKNOWN, and deliberately not claimed: the full TRAIN scope.** The scope
scan reached 100 000 of 2 621 624 rows (3.8 %) — `disagreements = 92 566`,
`over = 0` — and was **intentionally stopped** because it was projecting to ~26
hours. The scanner has since been optimised (§9a) and must be rerun to
completion (§12). No extrapolation from the first 3.8 % is made, The mechanism itself is
no longer in question -- only how many chunks it pushes past the bound.

*End of Audit 043.*
