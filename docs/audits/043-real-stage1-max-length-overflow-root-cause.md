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
| persisted chunk | `text`, `reference_length`, `base_length` | — | — | `chunking.py:270-277` |
| Stage-1 read | `row["text"]` | — | — | `execute.py::load_prepared_chunks` |
| Stage-1 base ids | `project_text(...) -> content_ids` | **`_CHUNK_PATTERN = \S+`** | added after | `data.py:181-215`, `alignment/manual.py:63,147` |
| Stage-1 special tokens | `build_inputs_with_special_tokens(content_ids)` | — | +2 | `data.py::_with_special_tokens` |
| Stage-1 assertion | `truncation.check(len(base_ids), "base sequence")` | — | — | `data.py:345` |

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

## 7. Offending Chunk Identity

**Not yet identified** — this requires the real tokenizer and the real corpus,
neither of which exists in the local ML-free environment. A read-only reproducer
is provided (§8) and the exact Colab command is in §12.

The real run's last telemetry was `update 250 / position 32000`. Per Audit 030
§AG the loop consumes exactly one batch per step with no look-ahead, so the
offence lies in the batch for update **251** — sampler positions 32 000…32 127.
That is a *derivation, not a measurement*: the reproducer walks the real sampler
order from position 0 and reports the first genuine offence, which is
authoritative. Nothing in this audit assumes the offender is sample 32 001.

## 8. Deterministic No-Training Reproducer

`scripts/stage1_length_contract_scan.py` — read-only. **No model is loaded, no
optimizer is constructed, no backward is called, no artifact is written, and
official TEST is never touched.** Two modes:

* `--reproduce` walks the **real** `DeterministicSampler` order at seed 21230,
  batch 128, and reports the first chunk whose Stage-1 realised base length
  exceeds `MAX_LENGTH`, together with its update, batch offset, sampler
  position, visit, `chunk_id`, text sha256 (16 hex), character count, newline
  presence, and both lengths.
* default mode streams the TRAIN partition and reports scope (§9).

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

The scanner computes, per chunk, both quantities and reports: total scanned,
count ≤ 256, count > 256, max Stage-6 authoritative length, max Stage-1 realised
length, a histogram of violating lengths, a histogram of
`realised - authoritative` deltas, and stable ids for up to `--max-offenders`
offenders.

The delta histogram is the decisive artifact: it separates *"one unlucky
boundary chunk"* from *"a systemic contract mismatch across every
newline-bearing chunk"*. **Scope is currently UNKNOWN** and cannot be determined
locally — 2 621 624 chunks require the real tokenizer. The exact Colab command
is in §12.

## 10. Classification

**A — STAGE-1 IMPLEMENTATION BUG.**

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
`scripts/stage1_length_contract_scan.py` (read-only diagnostic; created, then
repaired per §8a), `tests/test_stage1_length_contract_scanner.py` (new), and this
audit. **No scientific module was modified.**

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
after the FINAL scanner     64e659e99574276cb4a6d609c6c414900fa10d459161f5fef4baffefec25a184
  (scripts/ tree hash       d272bbede84d3c0a640e9554aae344a336a8452f31074bc3e588e9d356e19515)
```

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
21 passed

$ .venv/bin/python -B -m pytest -q -p no:cacheprovider
3807 passed, 106 skipped, 0 failed
```

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
# 1. Identify the real first offender in true sampler order
python -B scripts/stage1_length_contract_scan.py \
    --prepared-corpus /content/drive/MyDrive/UNMARK/stage1-prepared/aa49785eadcb \
    --reproduce --seed 21230 --max-updates 1000

# 2. Quantify scope over the whole TRAIN partition
python -B scripts/stage1_length_contract_scan.py \
    --prepared-corpus /content/drive/MyDrive/UNMARK/stage1-prepared/aa49785eadcb \
    --partition train \
    --report /content/drive/MyDrive/UNMARK/stage1-diagnostics/e495f7417fe4/length-contract-full-train.json
```

Both are read-only, train nothing, and execute **zero** scientific optimizer
steps. Step 2 decides one candidate/small class versus systemic mismatch, and
therefore whether the repair is confined to Stage-1 (§10) or whether
regeneration must even be discussed (§11).

## 13. Official UIT-VSFC TEST

**SEALED / UNUSED.** Not opened, inspected, mounted, searched, tokenized,
scanned or evaluated. The diagnostic reads only `chunks.jsonl` from the prepared
corpus and records `official_test_used: false`.

## 14. Verdict

**PRODUCTION BLOCKED — STAGE-1 IMPLEMENTATION BUG (CLASS A), pending measurement.**
Stage-1 must not be launched until the base-length contract is repaired and the
scope scan has run. No repair to Stage-1 production is applied in this audit.

**Provenance, stated precisely.** No additional training or scientific optimizer
step was executed by this root-cause investigation. The previously aborted real
`lr-pilot` attempt remains recorded with **250 confirmed optimizer updates and
no checkpoint** (§2, §11); this audit neither adds to nor erases that history.

**Still unproven, and deliberately not claimed.** The offending chunk is
**NOT YET MEASURED** and the full TRAIN scope is **UNKNOWN**: the reproducer's
first real execution failed on its own ImportError (§8a) before measuring
anything. The classification rests on source evidence — two modules asserting
contradictory BPE run units, one of them backed by 1708 of 1920 measured slice
cases — which the repair to the scanner does not change. It becomes a *measured*
finding only after the Colab rerun in §12.

*End of Audit 043.*
