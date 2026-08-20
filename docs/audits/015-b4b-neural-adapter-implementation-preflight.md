# Audit 015 — B4B neural adapter implementation preflight

| | |
|---|---|
| **Audit id** | 015 |
| **Date (UTC)** | 2026-08-20 |
| **Scope** | Implement `A_φ` in PyTorch; prepare the real-PhoBERT integration probe |
| **Repository state** | `HEAD = 3b1c66b`; this work uncommitted |
| **Predecessors** | [013](013-b3b1c-alignment-validation-and-channel-projection.md), [014](014-b4a-neural-adapter-contract-preflight.md) |
| **Phase** | Phase 1 / B4B |
| **Type** | **Implementation + probe preparation.** No training, no weights loaded locally |
| **Revised** | 2026-08-20 — two integration-safety gaps found in review and repaired **before** the Colab run: the gradient probe did not traverse the encoder (§M), and `wrapper.train()` could reactivate frozen-encoder dropout (§I). Verdict unchanged. |

---

## A. VERDICT

**PASS — REAL PHOBERT B4B COLAB PROBE REQUIRED**

The adapter is implemented exactly as B4A locked it, and everything checkable
without torch or model weights is checked: **1831 local tests pass, 16 skip**
cleanly because torch is absent by design.

**Real-model integration is NOT validated.** It cannot be, here. The position-id
question, the `input_ids` vs `inputs_embeds` control, the parameter count against
a real `hidden_size`, and gradient routing through a real frozen encoder all
require `scripts/b4b_phobert_adapter_probe.py` to run on Colab. Nothing in this
audit claims otherwise.

**Nothing was trained.** No optimizer, no `optimizer.step()`, no parameter
update, no dataset, no checkpoint saved. The local `.venv` remains ML-free:
torch, transformers, sentencepiece and datasets are all absent.

**Two integration-safety gaps were found in review and repaired before the real
run**, both of the same kind — a check that would have passed while the thing it
was supposed to protect was broken:

| Gap | Was | Now |
|---|---|---|
| **Gradient routing** | scalar taken from `z`, so `phi → z → loss` only | scalar from the **real encoder's final hidden state**, so the backward traverses `E_θ` into `A_φ` (D-B4B-005) |
| **Frozen-encoder mode** | `encoder.eval()` once at construction | `UnmarkEncoder.train()` overridden; encoder forced back to eval on **every** transition (D-B4B-004) |

The old `z.sum()` probe is **superseded and gone**; a test asserts it cannot
return.

---

## B. FILES CHANGED

| File | Change |
|---|---|
| `unmark/modeling/adapter.py` | **new** — `OrthographyInputAdapter`, `UnmarkEncoder` (with the `train()` override), `convex_combination`, freezing helpers |
| `unmark/modeling/pooling.py` | **new** — `masked_mean_non_special` (D-B4A-006) |
| `unmark/modeling/collate.py` | **new** — metadata → tensors; torch imported lazily |
| `unmark/modeling/__init__.py` | docstring: neural modules deliberately not re-exported |
| `scripts/b4b_phobert_adapter_probe.py` | **new** — Colab-only real-model probe |
| `tests/test_neural_adapter.py` | **new** — 58 local + 16 torch-gated |
| `tests/test_adapter_contract.py` | one B4A test rescoped (§S1) |
| `docs/spec/decisions.md` | D-B4B-001 … D-B4B-005 |
| `results/b4b/.gitkeep` | run directories stay out of git, per existing policy |

**No proposal change.** §4.5 is implemented as written; a model-API integration
detail does not alter the scientific description. PDF stale: **YES** (unchanged
from v1.4).

---

## C. LOCKED B4A CONTRACT

Implemented without deviation:

| Item | Implementation |
|---|---|
| Fusion | `Linear(3d → d, bias=True)` → `LayerNorm(d)`, **no activation between them** |
| Gate | `Linear(3d → d, bias=True)` → `sigmoid` — a projection, not a vector |
| Combination | `z = g⊙f + (1−g)⊙e` — **convex, not residual** |
| Tone table | exactly **7** rows, `NA` outside |
| Letter table | exactly **5** rows, `NA` outside |
| Gate init | `W_g = 0`, `c_g = logit(0.01)`, `g = 0.01` |
| Parameter count | `6d² + 16d` |
| Backbone | `hidden_size` from `model.config.hidden_size`; D-B3B0-002 **OPEN** |

---

## D. ADAPTER IMPLEMENTATION

`OrthographyInputAdapter.forward(base_embeddings, tone_ids, tone_mask,
letter_ids, letter_mask) → z: [B, L, d]`.

**Scope boundary, enforced by test.** The adapter consumes **tensors only**. It
imports nothing from `unmark.orthography`, `unmark.corruption`,
`unmark.linguistics` or `unmark.alignment` — a test asserts its import set is a
subset of `{__future__, typing, torch, unmark}`, where the only `unmark` imports
are the modeling contracts. The three-way separation is explicit:

```
text --B1A/B2/B3--> metadata --collate.py--> tensors --A_φ--> z
```

**Base embeddings.** `base_word_embeddings(encoder, input_ids)` calls
`encoder.get_input_embeddings()(input_ids)` — the **word** table only,
deliberately *not* `encoder.embeddings(...)`, which would also add position and
token-type embeddings and run the encoder's embedding LayerNorm and dropout.
Those belong downstream of `inputs_embeds` and must happen exactly once. No copy
is made, so the adapter never owns a trainable alias of a pretrained parameter.

---

## E. TONE ZERO-MASK IMPLEMENTATION

```python
mask = _as_bool_mask(tone_mask, "tone_mask")
_validate_ids(tone_ids, mask, TONE_TABLE_ROWS, "tone")
safe_ids = torch.where(mask, tone_ids, torch.zeros_like(tone_ids))
embedded = self.tone_embedding(safe_ids)
return embedded * mask.unsqueeze(-1).to(embedded.dtype)
```

**`-1` never reaches `nn.Embedding`.** It is replaced by row 0 before the lookup
and the result is then zeroed, so the substituted row cannot influence the
output. A test proves the substitution is invisible: `tone_ids = [-1, 2]` and
`[0, 2]` under the same mask give **identical** tensors.

**Validation is loud.** `_validate_ids` checks only *unmasked* positions — masked
positions are exactly where the sentinel lives. An unmasked id outside `[0, 7)`
raises `ChannelContractViolation`; silently accepting it would index the wrong
learned row and the model would train happily on it.

`tone_mask = True` implies a row in `0..6`; `tone_mask = False` **is** `NA`.
There is no eighth row, and a test asserts the cardinality comes from the
contract constant rather than a literal that could drift from it.

---

## F. LETTER MASKED-MEAN IMPLEMENTATION

```python
numerator = (embedded * weights).sum(dim=-2)   # [B, L, d]
count     = weights.sum(dim=-2)                # [B, L, 1]
pooled    = numerator / count.clamp(min=1.0)
has_any   = (count > 0).to(pooled.dtype)
return pooled * has_any
```

`NONE` is a learned row and participates in the mean; `NA` contributors are
excluded by the mask. Padded-dense `[B, L, K]` batching, with `K` the **batch's**
maximum — a test asserts `K` grows with batch content rather than being a global
constant.

**`0/0` is prevented, and the zero is by contract rather than by accident.** The
`clamp` alone would leave `sum(∅)/1 = 0` true only because an empty sum is zero;
the trailing `* has_any` is what *forces* it, and keeps forcing it if the
numerator ever stops being empty-safe. D-B4A-005 permits the clamp only on that
condition. A test asserts the result is finite — no `NaN`.

---

## G. GATE / FUSION / INITIALIZATION

**Gate — locked, explicit:**

```python
self.gate.weight.fill_(GATE_INIT_WEIGHT)   # 0.0
self.gate.bias.fill_(GATE_INIT_BIAS)       # logit(0.01) ≈ -4.59511985013459
```

`W_g = 0` makes the gate input-independent at step zero, so every token and every
hidden dimension starts at `sigmoid(c_g) = 0.01`. Verified numerically:
`gate_values(...)` returns `0.01` to `1e-7` on random input.

**This is not exact identity, and is not reported as such.** A test asserts the
initialised adapter's output is **not** `allclose` to `e`.

**Fusion and adapter LayerNorm — conventional PyTorch defaults, stated
explicitly.** `nn.Linear` uses Kaiming-uniform weights with a fan-in-scaled
uniform bias; `nn.LayerNorm` uses unit gain and zero bias; `nn.Embedding` uses
`N(0, 1)`. The proposal locks no separate initialisation for these, and inventing
one would be a new scientific hyperparameter. A test asserts no explicit
initialiser (`xavier_uniform_`, `kaiming_normal_`, `normal_`, `trunc_normal_`) is
called anywhere in the module — the defaults are the behaviour, not an accident.

**Forced gate-zero is a wiring test only.** `convex_combination(gate, fused,
base)` is a **free function** taking the gate as an argument, so the identity can
be tested at `g := 0` without the module carrying a flag. A test asserts no
`force_gate_zero` / `gate_zero_mode` / `use_gate_zero` / `zero_gate` appears in
any neural module **or the probe**, and that `AdapterConfig` has no such field.

---

## H. PARAMETER COUNT

`6d² + 16d`, derived in the test from the **declared module shapes** rather than
restated:

```
tone 7d + letter 5d + (3d² + d) + (3d² + d) + 2d  =  6d² + 16d
```

Verified at `d ∈ {64, 256, 768, 1024}` against `AdapterConfig`, and at runtime
against the built module at `d ∈ {16, 32}`.

**`hidden_size` is derived from `model.config.hidden_size`.** A test asserts the
literal `768` appears nowhere in the probe. If the real model reports 768 the
adapter will have **3,551,232** trainable parameters — the probe checks this
against the model, not against an assumption. D-B3B0-002 remains **OPEN**.

---

## I. FROZEN-ENCODER INTEGRATION

`freeze_encoder(encoder)` sets `requires_grad = False` on **every** encoder
parameter — word embeddings, positional embeddings, token-type embeddings, every
transformer block, every pretrained LayerNorm, and the pooler if present — and
calls `.eval()`. `UnmarkEncoder` applies it at construction and re-asserts
`requires_grad = True` on the adapter.

The encoder is held as a submodule, so `trainable_parameters(wrapper)` counts
only `A_φ`. The probe checks `trainable_parameters(model) == 0` and
`trainable_parameters(adapter) == 6d² + 16d` separately. **No downstream task
head exists in B4B.**

### Freezing and eval mode are different contracts (Gap 2, repaired)

`requires_grad = False` freezes *weights*. `eval()` disables *stochastic training
behaviour*. Freezing does not imply eval — and `nn.Module.train()` **recurses
into registered children**, so `wrapper.train()` would have flipped the
pretrained encoder into train mode and silently reactivated its dropout while
every encoder parameter stayed correctly frozen. Calling `encoder.eval()` once at
construction is not enough, and relying on the caller to re-invoke it is not a
contract.

**Why it matters here specifically.** §4.6 aligns the adapted branch to a
reference branch produced by *the same frozen encoder*. Encoder dropout during
adapter training would give the two branches different dropout draws of the same
weights, injecting avoidable stochasticity directly into the alignment objective.
The failure is silent: training proceeds and the loss decreases.

**The enforced invariant:**

| Call | `wrapper.training` | `encoder.training` | `adapter.training` |
|---|---|---|---|
| after construction | — | **False** | — |
| `wrapper.train()` | True | **False** | True |
| `wrapper.eval()` | False | **False** | False |
| `wrapper.train()` again | True | **False** | True |

`UnmarkEncoder.train(mode)` calls `super().train(mode)` for normal `nn.Module`
semantics — the adapter follows `mode`, `self.training` is set, `self` is
returned — then explicitly restores `self.encoder.eval()`. **`requires_grad` is
untouched**; this is module mode only, and a test asserts the override never
mentions `requires_grad`. The adapter is **not** pinned to eval.

**No escape hatch.** A test asserts `train()` takes `(self, mode)` and nothing
else. A frozen representation encoder running dropout is not something anyone
should be able to select by accident; changing it needs a logged decision
(D-B4B-004).

Runtime tests exercise construction, `train()`, `eval()`, and three repeated
`train()`/`eval()` cycles, asserting the invariant and that
`trainable_parameters(encoder) == 0` throughout.

---

## J. SPECIAL-TOKEN / PADDING CONTRACT

`build_example(input_ids, special_tokens_mask, projections)` interleaves content
projections with the model's own special tokens. **The special-token identity and
order come from the tokenizer** (`build_inputs_with_special_tokens` /
`get_special_tokens_mask`), never guessed.

Special tokens and padding get tone `NA` (sentinel + false mask), **zero**
applicable letter contributors, and no source range. A count mismatch between
non-special slots and projections **raises** — that would mean the alignment and
the encoder sequence disagree, which must never be papered over.

**No special-token bypass was added.** The locked architecture still passes
`[e_i ; 0 ; 0]` through `W_f`/`W_g`, and may transform those positions. That is a
consequence of the architecture, not a new exception. `attention_mask` remains
the tokenizer/model mask, untouched.

---

## K. POSITION-ID PROBE DESIGN

The question Audit 014 left open. **Not answered here — designed here.**

`PositionCapture` registers a **forward hook on the real position-embedding
module** (located by walking `named_modules()`, not assumed) and records the
actual index tensor it receives. Position ids are **not** inferred from library
source.

Four cases, each comparing path A (`input_ids`), path B (`inputs_embeds`), and
path C (`inputs_embeds` + explicit `position_ids`) when A and B differ:

| Case | Batch |
|---|---|
| 1 | one sentence, padding columns trimmed |
| 2 | batch with right padding |
| 3 | three examples of unequal length |
| 4 | real PhoBERT special tokens, with their ids recorded |

**The decision rule is pre-committed in D-B4B-002**, so the result cannot be read
selectively: if every required case matches, no explicit `position_ids` are
needed; if **any** differs, B4B must pass explicit `position_ids` reproducing the
authoritative `input_ids` path.

---

## L. INPUT_IDS VS INPUTS_EMBEDS CONTROL

The frozen-model control runs **before** the adapter is involved: same frozen
encoder, `model.eval()` under `torch.no_grad()` so dropout is disabled.

`tensor_diff` reports `max_abs_diff` and `mean_abs_diff` **split into attended
and padding positions**. That split is deliberate: if the two paths disagree only
at padding, the overall figure hides a result the content figure states plainly.
A mismatch confined to padding is a materially different finding from one at
content positions.

**No loose tolerance hides a mismatch.** The check is `max_abs_diff_content <
1e-4`, and the raw numbers are written to `equivalence.json` regardless of the
verdict. If material differences remain at content positions, B4B is **BLOCKED**.

---

## M. GRADIENT-ROUTING PROBE

**The first version was insufficient and is superseded.** It computed
`z.sum().backward()`, which validates only `phi → z → loss`. That would have
passed unchanged if the real integration path contained `z.detach()`, or ran the
encoder inside `torch.no_grad()` — leaving Stage-1 **unable to train `A_φ`
through the encoder** while the probe reported success. It was replaced before
the real run, and a test asserts `z_grad.sum().backward()` cannot come back.

**The repaired path** is the one future Stage-1 will actually use — through
`UnmarkEncoder`, not a hand-rolled forward:

```
phi → z = inputs_embeds → frozen E_θ → final hidden states → scalar → backward
```

The scalar is `masked_mean_non_special(hidden, attention_mask,
special_tokens_mask).sum()` — a finite diagnostic over attended, non-special
final hidden states. **It is not a scientific objective**; Stage-1's cosine loss
belongs to a later phase. One forward, one `backward()`, **no optimizer, no
`optimizer.step()`, no parameter update**; a test asserts the probe contains
exactly one `.backward()` call.

**Success conditions.** *Encoder*: `requires_grad == False` everywhere, and no
pretrained parameter carries a nonzero gradient — the artifact also reports how
many hold a `.grad` tensor at all. *Adapter*: gradient tensors must exist for
`W_g` weight **and** bias, `W_f` weight and bias, the adapter LayerNorm weight
and bias, and both embedding tables; every observed gradient finite; and **at
least one nonzero**, so a graph that is connected but numerically severed cannot
pass. Embedding rows the batch does not touch are **not** required to receive
gradients, and `W_g` starting at zero does not excuse a missing gradient
*tensor*. There are no `NA` rows, so zero-channel positions cannot create
gradients into nonexistent ones.

**Graph-break prohibition, correctly scoped.** No `detach()` and no `no_grad` may
sit on the adapted path used for gradient validation or future training. Tests
check both halves structurally: the adapter's forward path contains no `detach()`
and no `no_grad`; the probe's backward call and gradient-routing forward are not
inside any `no_grad` block. **`no_grad` is not banned from the probe** — the
`input_ids` vs `inputs_embeds` equivalence control is inference-only and
correctly uses `model.eval()` under `torch.no_grad()`. A test asserts that
control still uses it, so a later edit cannot over-correct by banning it
outright.

**The artifact proves which path ran**: `gradient_loss_source =
"encoder_final_hidden_state"` and `gradient_path_includes_encoder = true`, plus
`encoder_output_requires_grad`. **If the encoder-derived loss cannot
backpropagate into `A_φ`, B4B is INCOMPLETE.**

A local torch-gated test proves the wiring against a trivial stand-in encoder:
every adapter parameter receives a finite gradient, at least one is nonzero, and
the frozen encoder accumulates none. The real check runs on Colab.

---

## N. STAGE-1 POOLING UTILITY

`masked_mean_non_special(hidden_states, attention_mask, special_tokens_mask) →
[B, d]`, implementing D-B4A-006:

```
m_i = attention_mask_i AND NOT special_tokens_mask_i
h   = Σ_i m_i H_i / Σ_i m_i
```

Excludes padding **and** special tokens; works for arbitrary `L`; computed
independently per example, so the two branches may have different sequence
lengths. **Zero content positions raise `Stage1PoolingError` naming the offending
example indices** — no fallback to `<s>`, to an unmasked mean, or to a zero
vector.

**No Stage-1 loss and no training loop.** A test asserts the module defines
exactly `{content_mask, masked_mean_non_special}` and calls no
`backward` / `cosine_similarity` / `step` / `cross_entropy`.

---

## O. LOCAL TEST EVIDENCE

```
.venv/bin/python -m pytest -q -p no:cacheprovider
1831 passed, 16 skipped in 7.04s
```

Baseline before B4B: 1773. `tests/test_neural_adapter.py` holds **58 local + 16
torch-gated**. The skips are the torch numerics, skipped cleanly via
`pytest.mark.skipif` — **per test, not per module**.

**Added by this repair (11 local, 4 torch-gated).** Gap 1: the adapter's
`forward` and `adapted_embeddings` contain no `detach()`; no forward-path method
runs under `no_grad`; the probe's loss comes from `grad_outputs.last_hidden_state`
and not `z`; `z_grad.sum().backward()` cannot return; the gradient forward uses
the real wrapper; neither the backward call nor the gradient forward sits inside
a `no_grad` block; **and the equivalence control may still use `no_grad`** — that
last one exists so a future edit does not over-correct Gap 1 by banning `no_grad`
from the probe entirely. Gap 2: `train()` is overridden with `super().train(mode)`
plus `encoder.eval()` and returns `self`; the override never mentions
`requires_grad`; it takes no third argument; the probe records the mode sequence.
Torch-gated: the full mode-transition invariant including three repeat cycles;
`requires_grad` unchanged across transitions; gradients reaching every adapter
parameter through a stand-in encoder with the frozen encoder accumulating none;
and `W_g`'s gradient tensor existing despite zero initialisation.

Three tiers, deliberately:

* **Static (AST).** Table cardinality from contract constants not literals; both
  channels replacing the sentinel before lookup; the embedding tables called with
  `safe_ids` and never raw ids; no public gate-zero flag; no deterministic-pipeline
  imports in the adapter; no VnCoreNLP/transformers/unicodedata in the neural
  modules; no optimizer or checkpoint call; exactly one `.backward()` in the
  probe; no hardcoded `768`; no explicit weight initialiser.
* **Torch-free runtime.** The collator's metadata layer is genuinely exercised:
  `NA` → sentinel + false mask, `UNMARKED` a real row, projections landing in
  non-special slots, special tokens and padding `NA` in both channels, `NONE`
  included and `NA` excluded from contributors, `K` as a batch maximum, and loud
  failure on a projection-count mismatch.
* **Torch-gated.** Initial gate `0.01`; tone `NA` exactly zero; empty letter
  channel exactly zero and finite; the mean over `{NONE, CIRCUMFLEX}` equalling
  the arithmetic mean of those two rows; out-of-range unmasked ids raising; the
  masked sentinel provably invisible; the forced `g := 0` identity; the
  initialised adapter **not** being identity; the parameter formula; and pooling
  behaviour including loud failure.

**A note on test quality.** Four of my first-draft static tests matched raw
strings and flagged the modules' own documentation — a docstring saying "performs
no tokenization" tripped a `"tokenize" not in body` check. All four were rewritten
as structural AST checks over actual calls and imports. Raw-substring assertions
about source are brittle in exactly this way, and the fix is not a longer
denylist.

---

## P. REAL-MODEL COLAB PROBE

`scripts/b4b_phobert_adapter_probe.py`. **Not run locally**, and no claim is made
about its results.

Provenance recorded: requested and **resolved** revision for both tokenizer and
model (read back from the HF cache snapshot path, per audit 010 — passing
`revision=` does not prove the post-load state), tokenizer and model class,
transformers and torch versions, hidden size, vocab size, special token ids and
order, pad id, dtype, device. It **refuses with exit 3** if either resolves to a
revision other than the one requested.

Channel tensors come from the **real deterministic pipeline** — B1A decomposition,
B3A eligibility, B3B alignment and projection — not fabricated by hand. Dry-run
against a stub tokenizer confirms the two probe sentences exercise: marked tones
(`NANG`, `SAC`, `HUYEN`), `UNMARKED`, tone `NA` (digits and punctuation), letter
`NONE`, non-`NONE` labels (`CIRCUMFLEX`, `STROKE`, `HORN`), multi-piece tokens,
punctuation, special tokens, and padding via the shorter second sentence.

Writes `results/b4b/<run_id>/` with `config.json`, `summary.json`, `report.md`,
`position_ids.json`, `equivalence.json`, `gradients.json`, `module_modes.json`,
`channels.json`. Output
paths are resolved absolutely before anything can change the working directory (a
test asserts no `chdir`), so nothing depends on notebook state. Run directories
stay out of git per existing `.gitignore` policy; `results/b4b/.gitkeep` follows
the established pattern.

Status is computed from **22 checks** — the original 17 plus **five** added by
this repair — emitting `B4B_PHOBERT_ADAPTER_INTEGRATION_COMPLETE` or
`..._INCOMPLETE` with the failing check names listed. The count is **derived**:
`complete = all(checks.values())`, and the report prints
`len(summary['checks'])` rather than a literal, so code and documentation cannot
drift. A test asserts both.

The five added: gradient loss derived from real encoder output; gradient graph
reaches the adapter through the frozen encoder; encoder stays eval across every
mode transition; adapter follows the wrapper train/eval mode; encoder stays
frozen across mode transitions. The mode sequence is additionally written to
`module_modes.json` as evidence, which is an artifact rather than a check.

Mode invariants are exercised as a real sequence — constructed → `train()` →
`eval()` → `train()` again — with `wrapper.training`, `encoder.training`,
`adapter.training` and `encoder_requires_grad_any` recorded at each step. **A
failure makes the run INCOMPLETE.**

---

## Q. DECISION-LOG / PROPOSAL CONSISTENCY

Five entries: **D-B4B-001** (implementation matches the locked equation),
**D-B4B-002** (position-id rule pre-committed, result pending), **D-B4B-003**
(torch kept out of the package `__init__`), **D-B4B-004** (freezing weights and
disabling dropout are different contracts; the frozen encoder stays in eval
across every wrapper mode transition, the adapter follows the requested mode),
and **D-B4B-005** (gradient routing validated from the encoder's final hidden
state rather than from the adapter output, with the `detach`/`no_grad`
prohibition scoped to the trainable path so the inference-only equivalence
control keeps its `no_grad`).

**D-B4A-002 is untouched.** The position-id rule and its four required cases —
no padding, right padding, unequal batch lengths, real special tokens, actual
position ids, explicit `position_ids` if any A/B case differs — are unchanged by
this repair.

**No proposal change.** §4.5 is implemented as written, and a model-API
integration detail does not materially affect the scientific description. PDF
stale: **YES**, unchanged from v1.4.

---

## R. BLOCKING ISSUES

**None for this task.** The two gaps found in review are **repaired**, not
deferred: the gradient diagnostic now runs through the frozen encoder
(D-B4B-005), and the frozen encoder is proven to stay in eval across every
wrapper mode transition (D-B4B-004). Both were caught before the real run, which
is the point of a preflight.

One blocks *completion of B4B*, and by design:

**The real-model probe has not run.** Position-id behaviour, the frozen-model
equivalence control, the parameter count against a real `hidden_size`, and
gradient routing through a real frozen encoder are all unverified. They are
unverifiable locally and must not be reported as validated.

If the probe finds material content-position differences in the `input_ids` vs
`inputs_embeds` control that explicit `position_ids` do not resolve, **B4B is
BLOCKED** and the result must be reported rather than absorbed into a tolerance.
Likewise, if the encoder-derived diagnostic cannot backpropagate into `A_φ`, or
any mode invariant fails, the run is **INCOMPLETE** — both are now computed
checks, not prose.

---

## S. NON-BLOCKING ISSUES

1. **One B4A test was rescoped.** `test_no_nn_module_was_written` asserted no
   `nn.Module` subclass existed anywhere in `unmark/modeling/` — correct in B4A,
   when B4B's module did not exist. It is now
   `test_pure_data_contract_modules_define_no_nn_module`, scoped to `contracts.py`,
   `config.py` and `__init__.py`, which must never depend on torch.
   `tests/test_neural_adapter.py` guards the neural side. Recorded rather than
   quietly deleted.

2. **The collator is probe-shaped, not yet dataset-shaped.** `build_example` and
   `padded_batch` handle one example and one batch. A real Stage-1 data path will
   want bucketing and a `max_length` policy; neither is invented here.

3. **`UnmarkEncoder.forward` passes `position_ids` straight through when
   supplied**, and does not decide whether it *must* be supplied. That is
   D-B4B-002, deliberately left to the empirical result.

4. **The stand-in encoder in the local gradient test is trivial** — an embedding
   plus a linear layer. It proves the *wiring* (that a loss from the encoder's
   output reaches every adapter parameter and that the frozen encoder
   accumulates none); it says nothing about PhoBERT's real graph. That is the
   Colab probe's job.

5. **Two integration-safety gaps reached a written audit before being caught.**
   Both were of the same kind: a check that passes while the property it protects
   is broken. The `z.sum()` gradient probe would have passed against a detached
   graph; `encoder.eval()` at construction would have passed a one-shot mode
   assertion. Neither was found by a test — both were found by review. Worth
   recording as a limit on what the local tier can establish.

6. **Proposal PDF remains stale** (v1.4 source).

---

## T. GIT STATE

`HEAD = 3b1c66b`.

```
 M docs/spec/decisions.md
 M tests/test_adapter_contract.py
 M unmark/modeling/__init__.py
?? results/b4b/
?? scripts/b4b_phobert_adapter_probe.py
?? tests/test_neural_adapter.py
?? unmark/modeling/adapter.py
?? unmark/modeling/collate.py
?? unmark/modeling/pooling.py
```

Every change is left **unstaged**. No `add`, `commit`, `push`, `tag`, `stash`,
`reset`, `checkout` or `restore` was run. **Torch and transformers were not
installed locally; no model weights were downloaded or loaded; no network was
accessed; nothing was trained.** No Audit 016 was created — this file was revised
in place, with the superseded gradient-probe design recorded rather than erased.

```text
AUDIT FILE WRITTEN: docs/audits/015-b4b-neural-adapter-implementation-preflight.md
OTHER AUDIT MODIFICATIONS MADE: NO
COMMIT CREATED: NO
```
