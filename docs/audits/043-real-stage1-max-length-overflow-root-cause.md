# Audit 043 — Real Stage-1 MAX_LENGTH Overflow: Root Cause

**Scope:** production-blocking root-cause investigation of the first real Stage-1
training abort. **No repair applied.** Audits 001–042 untouched.
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

**Files changed:** `scripts/stage1_length_contract_scan.py` (new, read-only
diagnostic) and this audit. **No scientific module was modified.**

```
unmark/ tree hash        48d96b98a32dabd825ed10f583fa670efad053a3604eea8e7bfff3ba5baf5089
                         (unchanged; git diff over unmark/ configs/ requirements/
                          docs/spec/ is empty)

production fingerprint   before  66b13c39fa5fe01d9bbeb146708cd2bdc7fb4c786fc0aa29b48cddfc24b2d141
(unmark+scripts+configs  after   2386b9e551217d41ea03981e085129a103c79b2f26f7a933e411b496d19677e5
 +requirements)
```

The fingerprint changes **only** because the diagnostic script lives under
`scripts/`, which is an execution-relevant path for the clean-tree guard. It
must be committed before any scientific run, and it is inert with respect to
training.

**Local tests:** the mechanism demonstration in §5 runs locally and
deterministically. The scanner's CLI and fail-closed behaviour were exercised;
its scanning path requires `transformers`, absent locally by design. **No CUDA
or real-corpus result is claimed from this environment.**

**Exact Colab diagnostics still required:**

```
# 1. Identify the real first offender in true sampler order
python -B scripts/stage1_length_contract_scan.py \
    --prepared-corpus /content/drive/MyDrive/UNMARK/stage1-prepared/aa49785eadcb \
    --reproduce --seed 21230 --max-updates 1000

# 2. Quantify scope over the whole TRAIN partition
python -B scripts/stage1_length_contract_scan.py \
    --prepared-corpus /content/drive/MyDrive/UNMARK/stage1-prepared/aa49785eadcb \
    --partition train --report /content/drive/MyDrive/UNMARK/length-scan.json
```

Both are read-only and train nothing. Step 2 decides one candidate/small class
versus systemic mismatch, and therefore whether the repair is confined to
Stage-1 (§10) or whether regeneration must even be discussed (§11).

## 13. Official UIT-VSFC TEST

**SEALED / UNUSED.** Not opened, inspected, mounted, searched, tokenized,
scanned or evaluated. The diagnostic reads only `chunks.jsonl` from the prepared
corpus and records `official_test_used: false`.

## 14. Verdict

**PRODUCTION BLOCKED — STAGE-1 IMPLEMENTATION BUG (CLASS A).** Stage-1 must not
be launched until the base-length contract is repaired and the scope scan has
run. No repair is applied in this audit. No training was performed and no
scientific optimizer step was executed.

*End of Audit 043.*
