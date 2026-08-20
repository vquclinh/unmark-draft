# B4B — PhoBERT adapter integration

**Final status: `B4B_PHOBERT_ADAPTER_INTEGRATION_COMPLETE` — 27 of 27 checks
passed** on the repaired rerun.

Two runs are recorded, in order. **The first run's failure is not erased**: it
reported 21/22 and is kept as INCOMPLETE, because that is what it reported.

| Run | Id | Result |
|---|---|---|
| First | — | **INCOMPLETE**, 21/22 — provenance *reporter* defect |
| Final | `20260820T081554Z` | **COMPLETE**, 27/27, return code 0 |

---

# Run 1 — first real-model run

**Status: `B4B_PHOBERT_ADAPTER_INTEGRATION_INCOMPLETE` — 21 of 22 checks passed.**

The single failing check was a defect in the probe's own provenance *reporter*,
not in the model, the weights, or the adapter — but the run reported INCOMPLETE
and is recorded that way. It is not retroactively rewritten.

## Provenance

| | |
|---|---|
| **Checkpoint** | `vinai/phobert-base` |
| **Requested revision** | `01daacda68afe13d83023d16ec647239e344a1e6` |
| **Model class** | `RobertaModel` |
| **Tokenizer class** | `PhobertTokenizer` |
| **transformers** | 4.57.6 |
| **torch** | 2.11.0+cu128 |
| **Hidden size** | 768 |
| **Model vocab size** | 64001 |
| **Model weights loaded** | **true** |
| **Training performed** | **false** |

The revision is a **probe** revision, not the final backbone decision.
[D-B3B0-002](../spec/decisions.md#d-b3b0-002) remains **OPEN**.

## The single failure, and its root cause

```
revision verified (tokenizer and model)   FAIL

resolved_tokenizer_revision = 01daacda68afe13d83023d16ec647239e344a1e6
resolved_model_revision     = null
```

The collector looked for a Hugging Face cache snapshot path on the loaded object.
That works for a tokenizer, which keeps real resolved file paths on the instance.
It cannot work for a model: `model.name_or_path` is the **repo id**,
`"vinai/phobert-base"`, and never a path. The check could not have passed for any
model, at any revision.

**This was a reporting defect, not a loading defect.** Recorded as
[D-B4B-006](../spec/decisions.md#d-b4b-006--modelname_or_path-is-not-revision-evidence).

## Independent offline provenance diagnostic

Run on the same Colab runtime after the probe, against the cache as it stood:

```
HF_HOME     /content/unmark-draft/.hf-cache
hub cache   /content/unmark-draft/.hf-cache/hub
snapshot    .../models--vinai--phobert-base/snapshots/01daacda68afe13d83023d16ec647239e344a1e6
```

| Check | Result |
|---|---|
| `exact_snapshot_exists` | true |
| `direct_config_in_requested_snapshot` | true |
| `direct_weight_in_requested_snapshot` | true |
| `main_ref_matches_requested` | true |
| `transformers_cached_config_matches` | true |
| `transformers_cached_weight_matches` | true |
| `autoconfig_commit_matches` | true |

Status: **`MODEL_REVISION_CACHE_PROVENANCE_CONFIRMED`**.

`config.json`, `pytorch_model.bin`, `tokenizer.json`, `vocab.txt` and `bpe.codes`
all resolved under the requested snapshot; `AutoConfig` reported `_commit_hash =
01daacda68afe13d83023d16ec647239e344a1e6`; `cached_file` and
`huggingface_hub.try_to_load_from_cache` independently returned the same raw
snapshot paths.

**The weight file** `snapshots/01daacda…/pytorch_model.bin` symlinks to the
content-addressed blob
`blobs/a0b0f0912c710147fbaac015b0a4011216a0061a56c03b840b639e40d3bb49cc`. The
**raw** snapshot path carries the revision; the blob path does not, and resolving
the symlink first destroys the evidence. `main_ref_matches_requested` is recorded
as context but is **not** a required condition — the project pins an exact commit
and upstream `main` may legitimately move.

## Position ids — D-B4B-002 answered

The pre-committed rule: if **any** required case differs, explicit
`position_ids` are required. Observed on the real model:

| Case | Implicit `inputs_embeds` vs authoritative `input_ids` |
|---|---|
| 1 — one sentence, no padding | **identical** |
| 2 — right-padded batch | **DIFFERENT** |
| 3 — unequal-length batch | **DIFFERENT** |
| 4 — real special-token batch | **DIFFERENT** |

For a right-padded sequence:

```
authoritative (input_ids)      2, 3, 4, 5, 6, 1, 1, 1
implicit (inputs_embeds)       2, 3, 4, 5, 6, 7, 8, 9
```

Padding positions take the padding index on the authoritative path; the implicit
path numbers straight through them. **Case 1 matching is what makes this
dangerous** — a single unpadded sentence looks correct, so the bug would not
surface until batching.

**D-B4B-002 is CLOSED: explicit authoritative `position_ids` are required for the
PhoBERT adapted `inputs_embeds` path.**

**Scope of that result.** It was established for the **`vinai/phobert-base`
integration profile** — checkpoint, model type `roberta`, class `RobertaModel`,
transformers 4.57.6. It is **not** an experimental result about arbitrary
`roberta` checkpoints: `roberta-base`, `xlm-roberta-base` and
`vinai/phobert-large` were never measured, and the integration layer rejects them
until they are. The RoBERTa position arithmetic itself is not unique to PhoBERT;
what is PhoBERT-specific is the empirical permission to rely on it here.

## Frozen-model numerical control

With explicit authoritative `position_ids` supplied:

```
max_abs_diff           = 0.0
mean_abs_diff          = 0.0
max_abs_diff_content   = 0.0
mean_abs_diff_content  = 0.0
max_abs_diff_padding   = 0.0
```

between `model(input_ids=…)` and `model(inputs_embeds=Emb(input_ids),
position_ids=…, attention_mask=…)`. **Exact**, not within a tolerance.

## Adapter evidence

| Measure | Value |
|---|---|
| Hidden size (read from model) | 768 |
| Adapter trainable parameters | **3,551,232** |
| Expected `6d² + 16d` | **3,551,232** |
| Encoder trainable parameters | **0** |
| Initial gate min/max | 0.009999998845160007 |
| Initialised `z` max abs diff from base | 0.038273051381111145 |

The gate is `0.01` to floating-point precision, and the initialised adapter is
**not** identity — both as B4A locked them.

## Gradient routing

```
gradient_loss_source              encoder_final_hidden_state
gradient_path_includes_encoder    true
encoder_output_requires_grad      true
encoder_grad_count                0
encoder_parameters_with_grad_tensor  0
```

Gradient tensors existed for `gate.weight`, `gate.bias`, `fusion.weight`,
`fusion.bias`, `layer_norm.weight`, `layer_norm.bias`, `tone_embedding.weight`
and `letter_embedding.weight`. All observations finite; at least one nonzero.

This validates the Audit-015 repair on the real model: the diagnostic scalar
comes from the encoder's final hidden state, so the backward pass genuinely
traverses the frozen encoder into `A_φ`.

## Module modes

| Step | wrapper | encoder | adapter |
|---|---|---|---|
| constructed | — | **false** | true |
| `wrapper.train()` | true | **false** | true |
| `wrapper.eval()` | false | **false** | false |
| `wrapper.train()` again | true | **false** | true |

Encoder `requires_grad` remained false throughout. D-B4B-004 holds on the real
model.

## The 21 checks that passed

Hidden size read from model · special token ids recorded · position behaviour
determined · explicit ids recover the authoritative path · frozen `input_ids` vs
`inputs_embeds` equivalence · tone `NA` exact zero · empty letter channel exact
zero · initial gate `0.01` · adapter output shape · parameter formula · encoder
zero trainable parameters · forced `g=0` wiring identity · initialised adapter
non-identity · zero encoder gradients · gradient loss from real encoder output ·
gradient path through encoder · finite adapter gradients · frozen encoder remains
eval · adapter follows wrapper mode · encoder stays frozen · Stage-1 pooling
shape.

## Rerun required

The rerun must execute the **full** integration suite, not merely skip the
repaired check. It is not COMPLETE because the old failing check was removed —
the repaired verifier must genuinely prove **both** tokenizer and model resolve
to `01daacda68afe13d83023d16ec647239e344a1e6`, and every previously passing
neural check must still pass. Any regression makes the rerun INCOMPLETE.

The rerun must also confirm the two hardening repairs made after this run: that
the wrapper matched a **verified position profile** rather than a model family,
and that it **rejects a wrong caller-supplied `position_ids`** instead of
honouring it.

---

# Run 2 — final rerun, COMPLETE

| | |
|---|---|
| **Run id** | `20260820T081554Z` |
| **Repository HEAD** | `7f6e26c80c0acfa3cdf9168a9b0e2981e6ae1491` |
| **Probe return code** | **0** |
| **Status** | **`B4B_PHOBERT_ADAPTER_INTEGRATION_COMPLETE`** |
| **Checks computed / passed / failed** | **27 / 27 / 0** |
| **Model weights loaded** | **true** |
| **Training performed** | **false** |

## Provenance — both tokenizer and model verified

| | |
|---|---|
| Checkpoint | `vinai/phobert-base` |
| Requested revision | `01daacda68afe13d83023d16ec647239e344a1e6` |
| **Resolved tokenizer revision** | `01daacda68afe13d83023d16ec647239e344a1e6` |
| **Resolved model revision** | `01daacda68afe13d83023d16ec647239e344a1e6` |
| `revision_verified` | **true** |
| Tokenizer class / `is_fast` | `PhobertTokenizer` / `false` |
| Model class | `RobertaModel` |
| transformers / torch / Python | 4.57.6 / 2.11.0+cu128 / 3.12.13 |
| Model dtype / device | `torch.float32` / `cpu` |
| Hidden size / vocab size | 768 / 64001 |
| Pad token id | 1 |

Special tokens and ids: `<s>`=0, `</s>`=2, `<unk>`=3, `<pad>`=1, `<mask>`=64000.

Structured evidence (`provenance.json`):

```
cached config raw path        -> snapshots/01daacda.../config.json
cached weight raw path        -> snapshots/01daacda.../pytorch_model.bin
model.config._commit_hash        01daacda...
config_in_requested_snapshot     true
weight_in_requested_snapshot     true
```

`refs/main` was recorded but **not required**. `model.name_or_path` was **not**
used as revision evidence. The resolved blob path was recorded as forensic
information only. The D-B4B-006 repair is confirmed against the real model.

## All 27 checks passed

Revision verified (tokenizer **and** model) · hidden size read from model ·
special token ids recorded · position-id behaviour determined · explicit
`position_ids` recover the authoritative path · derived ids match the model's
authoritative ids · padding index derived from the model · wrapper passes the
authoritative ids · wrapper rejects a wrong caller-supplied `position_ids` ·
backbone matched a **verified profile**, not just a family · `input_ids` vs
`inputs_embeds` control equivalent · tone `NA` exactly zero · empty letter
channel exactly zero · initial gate `0.01` · adapter output shape · parameter
count `6d²+16d` · encoder zero trainable parameters · forced `g=0` wiring
identity · initialised adapter not identity · no encoder gradients · gradient
loss from real encoder output · gradient graph reaches the adapter through the
frozen encoder · adapter gradients finite · encoder stays eval across every mode
transition · adapter follows the wrapper mode · encoder stays frozen · Stage-1
pooling returns `[B, d]`.

## Position ids — unchanged from the first run

| Case | Implicit vs authoritative |
|---|---|
| 1 — single sentence, no padding | **identical** |
| 2 — right padded | **DIFFERENT** |
| 3 — unequal lengths | **DIFFERENT** |
| 4 — real special-token batch | **DIFFERENT** |

```
authoritative   2, 3, 4, 5, 6, 1, 1, 1, ...
implicit        2, 3, 4, 5, 6, 7, 8, 9, ...
```

`explicit_position_ids_required = true`. The production helper derived
`padding_index = 1` **from the model**, not from a constant.

| Helper evidence | |
|---|---|
| `derived_matches_model` | **true** |
| `wrapper_padding_index` | 1 |
| `wrapper_passes_authoritative_ids` | **true** |
| `wrong_override_rejected` | **true** |

Matched profile: `vinai/phobert-base` / `roberta` / `RobertaModel` /
`roberta_input_ids_offset`.

**Empirical permission remains profile-specific.** This is not a result about
arbitrary RoBERTa checkpoints, and the integration layer rejects them.
[D-B3B0-002](../spec/decisions.md#d-b3b0-002) remains **OPEN**.

## Frozen-model equivalence

With authoritative explicit position ids, on `[2, 20, 768]`:

```
max_abs_diff          = 0.0        mean_abs_diff          = 0.0
max_abs_diff_content  = 0.0        mean_abs_diff_content  = 0.0
max_abs_diff_padding  = 0.0
```

**Exact**, including padding positions. Real-model evidence, not a tolerance.

## Adapter

| Measure | Value |
|---|---|
| Hidden size | 768 |
| Adapter trainable parameters | **3,551,232** |
| Expected `6d² + 16d` | **3,551,232** |
| Encoder trainable parameters | **0** |
| Wrapped trainable parameters | **3,551,232** |
| Initial gate min / max | 0.009999998845160007 / 0.009999998845160007 |
| Initialised `z` max abs diff from base | 0.038273051381111145 |

The gate value is **float32 precision, not drift**. `logit(0.01)` rounded to
float32 is `-4.595119953155518`, and its float32 sigmoid is exactly
`0.009999998845160007` — reproduced bit-for-bit offline. It sits 1.16e-09 from
`0.01`, well inside the probe's 1e-6 tolerance. The initialised adapter is
**not** identity, as designed.

## Gradient routing

```
gradient_loss_source                 encoder_final_hidden_state
gradient_path_includes_encoder       true
encoder_output_requires_grad         true
encoder_grad_count                   0
encoder_parameters_with_grad_tensor  0
adapter_grad_nonzero_somewhere       true
adapter_parameters_with_finite_grad  8
adapter_parameters_without_grad      []
```

Gradient tensors existed for all eight required components: `gate.weight`,
`gate.bias`, `fusion.weight`, `fusion.bias`, `layer_norm.weight`,
`layer_norm.bias`, `tone_embedding.weight`, `letter_embedding.weight`.

**A single diagnostic backward. No optimizer, no update, no training.**

## Module modes

| Step | wrapper | encoder | adapter |
|---|---|---|---|
| constructed | true | **false** | true |
| `wrapper.train()` | true | **false** | true |
| `wrapper.eval()` | false | **false** | false |
| `wrapper.train()` again | true | **false** | true |

`encoder requires_grad any = false` throughout. D-B4B-004 validated on real
PhoBERT.

## B3 → B4 channel interface

The probe built its tensors with the **actual deterministic pipeline**, on:

```
"Tôi đang học nghiên cứu tại Đại học Quốc gia 2026."
"Chào bạn."
```

exercising marked tones, `UNMARKED`, tone `NA`, `NONE`, `CIRCUMFLEX`, `HORN`,
`STROKE`, punctuation, digits and other non-applicable positions, multi-piece
content, real special tokens, and padding.

**This is integration evidence, not linguistic coverage.** Two sentences
establish that the deterministic and neural halves meet correctly at the
interface; they establish nothing about coverage of Vietnamese orthography, which
is B1A/B3A's separate evidence.
