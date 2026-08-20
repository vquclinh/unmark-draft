# B3B-1 — manual alignment on the real PhoBERT tokenizer

Evidence for [D-B3B1B-001](../spec/decisions.md#d-b3b1b-001--alignment-runs-over-whitespace-chunks-not-linguistic-spans),
[D-B3B1B-002](../spec/decisions.md#d-b3b1b-002--vocabulary-oov-is-not-alignment-failure-mixed-pieces-stay-open)
and [D-B3B1C-001](../spec/decisions.md#d-b3b1c-001--manual-alignment-is-validated-tone-ownership-is-decided-by-candidate-count).

Produced by `scripts/b3b1_phobert_alignment_probe.py` on Colab against the real
`vinai/phobert-base` tokenizer at the pinned revision. **No model weights are
loaded**: the probe uses the tokenizer only.

## Three runs, and why the first two do not count

| Run | Verdict | What it actually measured |
|---|---|---|
| `20260820T031644Z` | **SUPERSEDED** — 6/13 sequence consistency | Alignment attempted at *linguistic span* granularity. Also ran with every eligibility label `UNDECIDED` (3,960 of them), because `DEFAULT_MANIFEST` was a relative path and `py_vncorenlp.VnCoreNLP()` changes the working directory. |
| `20260820T03…Z` (A05C) | **DIAGNOSTIC ONLY** | Whitespace-chunk hypothesis, run to locate the granularity error. Not a validation run. |
| **`20260820T035339Z`** | **`B3B1_ALIGNMENT_PROBE_COMPLETE`** | The corrected probe. The numbers below are all from this run. |

The first run's 6/13 was **not** a property of fastBPE. It was a category error
on our side: BPE merges operate over maximal non-whitespace chunks, and a
linguistic span is not one. Feeding the tokenizer a sub-chunk fragment asks it a
question it was never trained to answer.

## Corrected run — `20260820T035339Z`

| Measure | Result |
|---|---|
| Sentences aligned | **2,489 / 2,489** |
| Curated diagnostic sentences | **42 / 42** |
| Token-sequence consistency | **13 / 13** |
| Token-**id** consistency | **13 / 13** |
| Whitespace chunks reconstructed exactly | **119 / 119** |
| Eligibility labels still `UNDECIDED` | **0** |
| Forms with an unknown vocabulary id | **1** |

Chunk-level surface reconstruction is exact everywhere: for every chunk, the
concatenation of its pieces' `@@`-stripped surfaces equals the chunk, and every
piece carries a half-open global character range in `b(x)`.

### Piece overlays

191 piece↔region overlays:

| Attribution | Count |
|---|---|
| Single Vietnamese candidate | 138 |
| No candidate (`N/A`) | 51 |
| Mixed contributors | **2** |
| Unresolved eligibility | **0** |

**Pieces spanning two distinct Vietnamese candidates: zero.**

Both mixed pieces mix exactly one candidate with punctuation:

```text
piece "en-"   <- candidate "tuyen" + "-"     (inside a URL)
piece ".com"  <- "." + candidate "com"       (inside a URL)
```

This is what closed the tone-ownership question. The audit-012 rule ("any
mixture ⇒ no tone") would have discarded two tone labels that were never
ambiguous, because nothing else in either piece was competing to own them.

### The unknown vocabulary id

One form (`khut`) maps to id 3, the unknown id. Its **surface is fully
recoverable** — `tokenizer.tokenize("khut")` returns `["khut"]`, and only
`convert_ids_to_tokens([3])` destroys it by returning `<unk>`. The probe never
round-trips ids to recover surfaces. Such a piece is `ALIGNED` with
`has_unknown_token_id = True` and keeps its channels. No string is special-cased.

## What this run does and does not establish

**Established.** Manual whitespace-chunk alignment is a **validated component**,
not a hypothesis: exact, deterministic, and complete on 2,489 sentences against
the real tokenizer. `offset_mapping` is not needed.

**Not established.** The backbone checkpoint is still not locked
([D-B3B0-002](../spec/decisions.md#d-b3b0-002)). This probe pinned a revision for
reproducibility, which is a provenance requirement, not a modelling decision.
