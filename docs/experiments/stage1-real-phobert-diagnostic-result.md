# Stage-1 real-PhoBERT integration diagnostic

**Status: `STAGE1_REAL_PHOBERT_DIAGNOSTIC_COMPLETE` — 31 of 31 checks passed.**

**This is not a training result and carries no performance interpretation.** It
answers exactly one question: *does the complete real-PhoBERT Stage-1
computation graph execute correctly, across all three branches, with the
intended gradient path?* It says nothing about whether Stage-1 works.

| | |
|---|---|
| **Run id** | `20260820T093520Z` |
| **Repository HEAD recorded by the run** | `6eb053f2b90b7c82fbfd50c5b33287551448691b` |
| **Status** | **`STAGE1_REAL_PHOBERT_DIAGNOSTIC_COMPLETE`** |
| **Checks passed** | **31 / 31**, none failed |
| **Real model weights loaded** | **yes** |
| **Backward calls** | **1** (diagnostic) |
| **Optimizer created** | **no** |
| **Parameter update performed** | **no** |
| **Scientific training performed** | **no** |

## Provenance

| | |
|---|---|
| Checkpoint | `vinai/phobert-base` |
| Requested revision | `01daacda68afe13d83023d16ec647239e344a1e6` |
| **Resolved tokenizer revision** | `01daacda68afe13d83023d16ec647239e344a1e6` |
| **Resolved model revision** | `01daacda68afe13d83023d16ec647239e344a1e6` |
| Tokenizer / model class | `PhobertTokenizer` (`is_fast=false`) / `RobertaModel` |
| Python / torch / transformers | 3.12.13 / 2.11.0+cu128 / 4.57.6 |
| Device | `cuda` — NVIDIA RTX PRO 6000 Blackwell Server Edition |
| Hidden size / vocab / pad id | 768 / 64001 / 1 |
| Diagnostic seed / visit | 20260820 / 0 |

Model provenance used the D-B4B-006 verifier: the cached **config** and
**weight** raw paths both lie under `snapshots/01daacda…/`,
`config._commit_hash` agrees, `refs_main` was `null` and **not required**,
`name_or_path_is_revision_evidence` is `false`, and the resolved blob path is
recorded as forensic only. `verified: true`, `failure_reasons: []`.

The revision remains a **probe** revision.
[D-B3B0-002](../spec/decisions.md#d-b3b0-002) remains **OPEN**.

## Run purpose — diagnostic, and stamped as such

```json
"run_config": {
  "purpose": "DIAGNOSTIC",
  "diagnostic_only": true,
  "values_are_scientific": false,
  "resolved_values": [],
  "lambda_align": 1.0,
  "lambda_clean": 1.0,
  "max_length": null,
  "on_overflow": "NOT_APPLICABLE",
  "corruption_scope": "TONE",
  "corruption_seed": 20260820
}
```

**`lambda_align = lambda_clean = 1.0` are diagnostic wiring values.** They are
not a tuning result, not a recommendation, and **resolve nothing**. The
`max_length: null` / `on_overflow: NOT_APPLICABLE` pair is the explicit
`TruncationPolicy.unbounded()` statement, not an omitted default.

This is the D-S1A-006 machinery working on a real run: `resolved_values` is
empty, so a `SCIENTIFIC` configuration still could not have been constructed.

## The three branches, on real weights

| Branch | Path |
|---|---|
| Reference | `canon(x)` → frozen tokenizer → **bare** frozen encoder → masked mean → `h_ref` |
| Adapted clean | `b(x)` → `T(b(x))` → **clean** channels → `A_φ` → frozen encoder → masked mean |
| Adapted corrupt | **same base grid** → **corrupted** channels → `A_φ` → frozen encoder → masked mean |

Pooled shape `[2, 768]` for all three.

## Real tokenizer, unequal branch lengths

| Sample | reference length | base length |
|---|---|---|
| `s1diag-0001` | 15 | 20 |
| `s1diag-0002` | 19 | 20 |

Padded widths: **reference 19, base 20** — genuinely different, which is the
situation §4.6's pooled-only alignment exists for. No token-level correspondence
was assumed anywhere.

## Corruption

| Sample | rate `p` | tone changes | letter changes |
|---|---|---|---|
| `s1diag-0001` | 0.7921992468868331 | **10** | **0** |
| `s1diag-0002` | 0.9568047460267669 | **11** | **0** |

```
canonical  Tôi đang học nghiên cứu tại Đại học Quốc gia Hà Nội.
corrupted  Tôi đang hoc nghiên cưu tai Đại hoc Quôc gia Ha Nôi.
base       Toi dang hoc nghien cuu tai Dai hoc Quoc gia Ha Noi.
```

Real tone marks were removed; **letter diacritics survived intact** — `ư` in
`cưu`, `ô` in `Quôc`/`Nôi`, `đ` in `Đại`. `letter_changes = 0` on both samples
confirms `TONE`-scope corruption did not touch the letter channel. B2 remained
authoritative; nothing about corruption was reimplemented for Stage-1.

## Objective

| | |
|---|---|
| `loss_align` | 0.5320950746536255 |
| `loss_clean` | 0.5478166341781616 |
| **total** | **1.079911708831787** |
| per-example align distances | 0.5276231169700623, 0.5365670919418335 |
| per-example clean distances | 0.5464549660682678, 0.5491783618927002 |
| max abs diff, adapted-clean vs adapted-corrupt pooled | 0.23897740244865417 |

**These numbers have no performance interpretation whatsoever.** They are the
loss of an untrained adapter at initialisation, under diagnostic weights. A
lower or higher value would mean nothing about Stage-1's viability. What they
establish is only that the components are **finite** and **arithmetically
consistent**.

Three consistency checks, recomputed from the per-example distances:

* `mean(align distances) = 0.5320951` — matches `loss_align`, confirming
  **mean over the batch, not sum**;
* `mean(clean distances) = 0.5478167` — matches `loss_clean`;
* `1.0 · loss_align + 1.0 · loss_clean = 1.0799117` — matches `loss` exactly.

All four distances lie in `[0, 2]`, the valid range for a cosine distance.

The nonzero `clean_corrupt_max_abs_diff` matters more than the loss: the two
adapted branches share one base grid and differ **only** in their orthography
channels, so a nonzero difference proves the channels actually reach the
representation. Had they been ignored, the two branches would have coincided.

## Gradient routing

| | |
|---|---|
| Backward calls | **1** |
| Encoder gradient tensors | **0** |
| Encoder nonzero gradients | **0** |
| Adapter groups with finite, nonzero gradients | **8 / 8** |

| Adapter parameter | `abs_sum` |
|---|---|
| `fusion.weight` | 1593.4522705078125 |
| `gate.weight` | 511.51104736328125 |
| `fusion.bias` | 4.49508810043335 |
| `layer_norm.bias` | 0.7126071453094482 |
| `tone_embedding.weight` | 0.6079995036125183 |
| `letter_embedding.weight` | 0.5789470672607422 |
| `layer_norm.weight` | 0.4559824466705322 |
| `gate.bias` | 0.45198360085487366 |

Both embedding tables received nonzero gradients, which is the end-to-end
confirmation that the orthography channels are connected to the loss through
the frozen encoder.

## Zero-update proof

```
adapter_sha256_before  8cdd8c7e14e681076282e9743db8cacea23534d2248c9d773fc37b7402cd76d7
adapter_sha256_after   8cdd8c7e14e681076282e9743db8cacea23534d2248c9d773fc37b7402cd76d7
encoder_sha256_before  85965b16464681d45da4ed02c5370879a2a855071e84db6def3d429137fe52cb
encoder_sha256_after   85965b16464681d45da4ed02c5370879a2a855071e84db6def3d429137fe52cb
optimizer_created            false
parameter_update_performed   false
training_performed           false
```

**A backward pass is not an optimizer step, and an optimizer step is not
training.** Gradients were computed and then discarded; the parameter
fingerprints are byte-identical before and after. Nothing learned anything.

## Position and freezing contracts

The run matched the verified position profile — `vinai/phobert-base` /
`roberta` / `RobertaModel` / `roberta_input_ids_offset`, evidence
`D-B4B-002` — so arbitrary RoBERTa-family support was **not** reopened.
The encoder stayed in `eval` while the adapter stayed in `train`, and the
encoder's trainable parameter count was **0** against the adapter's exactly
`3,551,232 = 6d² + 16d` at `d = 768`.

## Artifacts inspected

`stage1-real-phobert-diagnostic-results.zip`
(SHA-256 `beeef65c391a964100731b9d33a9cd498b4471cf5bd5b4a6369929fb38ad3450`),
containing `20260820T093520Z/`: `config.json`, `summary.json`, `losses.json`,
`gradients.json`, `provenance.json`, `examples.json`, `report.md`,
`repo-head.txt`, `scientific-status.txt`.

**The ZIP is deliberately not committed.** `results/**` is git-ignored except for
`.gitkeep`, and the B4B run directory was likewise never committed; the durable
record is this file. Every number above was read from those artifacts, not from
a summary.

## Reproducibility limitations

Recorded honestly rather than left implicit:

1. **No syllable-inventory provenance in the artifact.** The run used the pinned
   Vietnamese syllable inventory — a fresh Colab runtime fetched it through the
   repository's checksum-verifying fetcher — but `provenance.json` records only
   model, tokenizer and position-profile provenance. The inventory decides which
   spans are eligible and therefore every corruption denominator, so a scientific
   run must persist it. See [D-S1A-008](../spec/decisions.md).

2. **The driver is not committed.** At `6eb053f` the Stage-1 *library*
   (`unmark/stage1/*`) is committed and the working tree was clean, so the code
   that computed the result is pinned exactly. But unlike every previous phase —
   `b3b0`, `b3b1`, `b3b2`, `b4b` all committed their probe scripts — there is no
   `scripts/stage1_*` driver. The inputs are recoverable from the artifact
   (sample ids, texts, seed, visit), but the assembly step is not. See
   [D-S1A-008](../spec/decisions.md).

Neither invalidates the diagnostic: the resource was checksum-verified before the
run, and the library code is pinned. Both must be closed before scientific
training.

## What this does not establish

Not Stage-1 performance. Not a good loss value. Not optimal `lambda` values. Not
optimal corruption settings. Not an optimal checkpoint. Not successful training.
Not downstream robustness. Not the final backbone choice.

**No OPEN scientific value was resolved by this run.**
