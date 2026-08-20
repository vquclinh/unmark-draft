# Audit 017 — B4B real-PhoBERT final closure

| | |
|---|---|
| **Audit id** | 017 |
| **Date (UTC)** | 2026-08-20 |
| **Scope** | Persist the final real-model result; close B4B |
| **Repository state** | `HEAD = 7f6e26c`; this work uncommitted |
| **Predecessors** | [014](014-b4a-neural-adapter-contract-preflight.md), [015](015-b4b-neural-adapter-implementation-preflight.md), [016](016-b4b-real-model-provenance-and-position-repair.md) |
| **Phase** | Phase 1 / B4B closure |
| **Type** | **Result persistence and phase closure.** No code change, no training |

---

## A. VERDICT

**PASS — B4B COMPLETE; STAGE-1 IMPLEMENTATION MAY BEGIN**

The repaired real-PhoBERT probe rerun returned **0** with **27 of 27 checks
passing**, status `B4B_PHOBERT_ADAPTER_INTEGRATION_COMPLETE`. Both the
provenance verifier and the position-id enforcement added in Audit 016 are
confirmed against the real model.

**Stage-1 *implementation* may begin. Training may not.** Nothing has been
trained, and training requires the repository-wide **PRE-TRAIN audit** first.
`training_performed: false` on the run itself.

**1874 local tests pass, 32 skip** because torch is absent by design.

---

## B. SCOPE

This audit **persists a result and states a phase boundary**. It changes no
neural implementation, no adapter mathematics, and no scientific decision. The
only code touched is four narrow documentation-consistency assertions.

Deterministic B1/B2/B3 untouched.

---

## C. FINAL REAL RUN

| | |
|---|---|
| **Run id** | `20260820T081554Z` |
| **Repository HEAD (Colab)** | `7f6e26c80c0acfa3cdf9168a9b0e2981e6ae1491` |
| **Probe return code** | **0** |
| **Status** | **`B4B_PHOBERT_ADAPTER_INTEGRATION_COMPLETE`** |
| **Checks computed / passed / failed** | **27 / 27 / 0** |
| **Model weights loaded** | **true** |
| **Training performed** | **false** |

**Independently cross-checked before persisting**, rather than transcribed on
trust:

* The 27 check names supplied match the probe's own `checks` dictionary as an
  **exact set** — no name added, dropped or reworded.
* The local repository `HEAD` is `7f6e26c`, matching the Colab HEAD prefix.
* `6 · 768² + 16 · 768 = 3,551,232`, matching the reported adapter parameter
  count.
* `<mask>` = 64000 is the last index of a 64001-entry vocabulary; `pad_token_id`
  = 1 matches the derived `padding_index` = 1.

---

## D. PROVENANCE

| | |
|---|---|
| Checkpoint | `vinai/phobert-base` |
| Requested revision | `01daacda68afe13d83023d16ec647239e344a1e6` |
| **Resolved tokenizer revision** | `01daacda68afe13d83023d16ec647239e344a1e6` |
| **Resolved model revision** | `01daacda68afe13d83023d16ec647239e344a1e6` |
| `revision_verified` | **true** |
| Tokenizer / `is_fast` | `PhobertTokenizer` / `false` |
| Model class | `RobertaModel` |
| transformers / torch / Python | 4.57.6 / 2.11.0+cu128 / 3.12.13 |
| dtype / device | `torch.float32` / `cpu` |
| Hidden size / vocab | 768 / 64001 |
| Pad token id | 1 |
| Special tokens | `<s>`=0, `</s>`=2, `<unk>`=3, `<pad>`=1, `<mask>`=64000 |

Structured evidence: the cached **config** and cached **weight** raw paths both
lie under `snapshots/01daacda…/`, and `model.config._commit_hash` agrees.
`config_in_requested_snapshot` and `weight_in_requested_snapshot` are both true.

**The D-B4B-006 repair is confirmed on the real model.** `refs/main` was recorded
but **not required**; `model.name_or_path` was **not** used as revision evidence;
the resolved blob path was forensic only. The first run's sole failure is
therefore closed by evidence, not by removing the check — the check now passes
while verifying strictly more than it did before.

---

## E. 27/27 CHECKS

All true. Grouped by what they establish:

| Group | Checks |
|---|---|
| **Provenance** (1) | revision verified (tokenizer **and** model) |
| **Model facts** (2–3) | hidden size read from model; special token ids recorded |
| **Position ids** (4–10) | behaviour determined; explicit ids recover the authoritative path; derived ids match the model's; padding index derived from the model; wrapper passes the authoritative ids; wrapper **rejects** a wrong override; backbone matched a **verified profile**, not just a family |
| **Frozen control** (11) | `input_ids` vs `inputs_embeds` equivalent |
| **Channels** (12–13) | tone `NA` exactly zero; empty letter channel exactly zero |
| **Adapter** (14–19) | initial gate `0.01`; output shape; `6d²+16d`; encoder zero trainable; forced `g=0` wiring identity; initialised adapter **not** identity |
| **Gradients** (20–23) | no encoder gradients; loss from real encoder output; graph reaches the adapter through the frozen encoder; adapter gradients finite |
| **Module modes** (24–26) | encoder stays eval across every transition; adapter follows the wrapper mode; encoder stays frozen |
| **Stage-1 utility** (27) | pooling returns `[B, d]` |

---

## F. POSITION-ID RESULT

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

`explicit_position_ids_required = true`, unchanged from the first run.

| Helper evidence | |
|---|---|
| `derived_matches_model` | **true** |
| `wrapper_padding_index` | 1, **derived from the model** |
| `wrapper_passes_authoritative_ids` | **true** |
| `wrong_override_rejected` | **true** |

Matched profile: `vinai/phobert-base` / `roberta` / `RobertaModel` /
`roberta_input_ids_offset`.

**Empirical permission remains profile-specific.** This result is about
`vinai/phobert-base`, not arbitrary RoBERTa checkpoints; the integration layer
rejects them at construction. **[D-B3B0-002](../spec/decisions.md#d-b3b0-002)
remains OPEN.**

---

## G. FROZEN-MODEL EQUIVALENCE

With authoritative explicit position ids, on `[2, 20, 768]`:

```
max_abs_diff          = 0.0        mean_abs_diff          = 0.0
max_abs_diff_content  = 0.0        mean_abs_diff_content  = 0.0
max_abs_diff_padding  = 0.0
```

**Exact, including padding positions** — not a tolerance, and not a
content-only result with padding quietly excluded. Real-model evidence.

---

## H. ADAPTER / PARAMETER EVIDENCE

| Measure | Value |
|---|---|
| Hidden size | 768 |
| Adapter trainable parameters | **3,551,232** |
| Expected `6d² + 16d` | **3,551,232** |
| Encoder trainable parameters | **0** |
| Wrapped trainable parameters | **3,551,232** |
| Initial gate min / max | 0.009999998845160007 |
| Initialised `z` max abs diff from base | 0.038273051381111145 |

**The gate value is float32 precision, not drift, and I verified this rather
than assuming it.** `logit(0.01)` rounded to float32 is `-4.595119953155518`;
its float32 sigmoid is exactly `0.009999998845160007`, reproduced bit-for-bit
offline. That is 1.16e-09 from `0.01`, inside the probe's 1e-6 tolerance by three
orders of magnitude.

`wrapped == adapter` trainable count confirms the wrapper owns no trainable alias
of a pretrained parameter. The initialised adapter is **not** identity, as
designed — `g = 0.01` is close to the base-only pathway, not equal to it.

**No adapter mathematical decision changed.**

---

## I. GRADIENT ROUTING

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

All eight required components carried gradient tensors: `gate.weight`,
`gate.bias`, `fusion.weight`, `fusion.bias`, `layer_norm.weight`,
`layer_norm.bias`, `tone_embedding.weight`, `letter_embedding.weight`.

**This validates the Audit-015 repair on the real model.** The superseded
`z.sum()` design would have passed against a severed graph; this one could not,
because the scalar is downstream of the frozen encoder.

**One diagnostic backward. No optimizer, no `optimizer.step()`, no update, no
training.**

---

## J. MODULE MODES

| Step | wrapper | encoder | adapter |
|---|---|---|---|
| constructed | true | **false** | true |
| `wrapper.train()` | true | **false** | true |
| `wrapper.eval()` | false | **false** | false |
| `wrapper.train()` again | true | **false** | true |

`encoder requires_grad any = false` throughout. **D-B4B-004 validated on real
PhoBERT**: the `train()` override holds against a real transformer's submodule
recursion, which is where a plain `encoder.eval()` at construction would have
silently reactivated dropout.

---

## K. B3 → B4 CHANNEL INTERFACE

The probe built its tensors with the **actual deterministic pipeline** — B1A
decomposition, B3A eligibility, B3B alignment and projection — not by hand:

```
"Tôi đang học nghiên cứu tại Đại học Quốc gia 2026."
"Chào bạn."
```

Exercised: marked tones, `UNMARKED`, tone `NA`, `NONE`, `CIRCUMFLEX`, `HORN`,
`STROKE`, punctuation, digits and other non-applicable positions, multi-piece
content, real special tokens, and padding via the shorter sentence.

**This is integration evidence, not linguistic coverage.** Two sentences show the
deterministic and neural halves meet correctly at the interface. They say nothing
about coverage of Vietnamese orthography — that is B1A/B3A's separate evidence,
and claiming otherwise from two sentences would be a real overreach.

---

## L. DECISION CLOSURES

Existing entries were **annotated with real-model confirmation** rather than
duplicated — no new decision restates the run:

| Entry | Update |
|---|---|
| **D-B4B-002** | already CLOSED; now marked **re-confirmed** on the 27/27 rerun, including the wrapper passing authoritative ids and rejecting a wrong override |
| **D-B4B-004** | **confirmed on real PhoBERT** — encoder eval across all four transitions, `requires_grad` false throughout |
| **D-B4B-005** | **confirmed on real PhoBERT** — `encoder_output_requires_grad = true`, encoder grad count 0, eight components with finite gradients |
| **D-B4B-006** | **confirmed on real PhoBERT** — both tokenizer and model resolved to the requested revision |

One new entry, **D-B4B-007 — B4B is COMPLETE**, exists solely to state the phase
boundary and its limits (§O, §P). It restates no run detail the experiment record
already carries.

**[D-B3B0-002](../spec/decisions.md#d-b3b0-002) REMAINS OPEN.**

---

## M. EXPERIMENT RECORD

`docs/experiments/b4b-phobert-adapter-integration-result.md` now carries **both
runs, in order**:

| Run | Result |
|---|---|
| **Run 1** | `INCOMPLETE`, 21/22 — provenance *reporter* defect |
| **Run 2** | `COMPLETE`, 27/27, return code 0 |

**The first run's failure is not erased.** It reported INCOMPLETE and is kept
that way, with its root cause and the independent offline diagnostic intact. A
record that rewrites its own failures once they are fixed is not evidence of
anything. Tests assert both runs remain present.

---

## N. LOCAL REGRESSION TESTS

```
.venv/bin/python -m pytest -q -p no:cacheprovider
1874 passed, 32 skipped in 7.42s
```

Baseline before this task: 1870 passed, 32 skipped. **+4**, all
documentation-consistency assertions: the record states the final COMPLETE status
with the run id and HEAD; it preserves the 21/22 first run and both run headings;
it does not overclaim (the "integration evidence, not linguistic coverage" caveat
and the open backbone decision are both present); and D-B4B-007 scopes what
COMPLETE excludes.

**No local test attempts to reproduce the Colab run**, and none should — the
local environment is deliberately ML-free. The 32 skips are the torch numerics.

---

## O. WHAT B4B COMPLETE MEANS

**B4B NEURAL ADAPTER + REAL PHOBERT INTEGRATION: COMPLETE.**

* An actual PyTorch adapter exists.
* It implements the B4A-locked equation with no deviation.
* Real PhoBERT **weights were loaded**, at a **verified** revision for both
  tokenizer and model.
* Frozen-model `input_ids` vs `inputs_embeds` equivalence is **exact**.
* `inputs_embeds` position semantics were **measured** and are **enforced**.
* The parameter partition is verified: `6d² + 16d` trainable, **zero** in the
  encoder.
* Gradients route **through the frozen encoder** into `A_φ`.
* The frozen encoder's eval-mode invariant holds.
* The deterministic B3 → neural B4 interface was exercised with the real
  pipeline.
* **No scientific training has occurred.**

---

## P. WHAT REMAINS OPEN

* **The Stage-1 objective is not implemented.** §4.6's `L_align` / `L_clean` do
  not exist in code. Only the pooling utility does.
* **Nothing has been trained.** One diagnostic backward pass is not training.
* **[D-B3B0-002](../spec/decisions.md#d-b3b0-002) is OPEN** — the backbone is not
  selected, and the pinned revision remains a **probe** revision.
* **No downstream task, baseline or evaluation has run.** `RESTORE`, `ALIGN`,
  `FLOOR`, `UPPER` and the classification head are all untouched.
* Several §5 spec-lock items remain open, including the Stage-1 corpus and the
  classification-head concrete values.

**Stage-1 implementation may begin. Training may not** — that requires the
repository-wide **PRE-TRAIN audit**.

---

## Q. PROPOSAL / PDF STATUS

**Proposal updated: NO.** The run confirms the specification rather than changing
it; no scientific description mismatch was exposed.

**Compiled PDF stale: YES**, carried over from the earlier v1.4 source changes
(§4.6 Stage-1 pooling, §4.7/§8.2 letter cardinality).

---

## R. BLOCKING ISSUES

**None.** B4B is closed on real-model evidence, and the next phase is unblocked
for implementation.

---

## S. NON-BLOCKING ISSUES

1. **The device was `cpu`.** The run establishes correctness, not throughput; no
   GPU-specific numerical behaviour was exercised. Stage-1 will run on GPU, and
   nothing here rules out a device-dependent difference — though the frozen
   control being exactly 0.0 makes one unlikely to be structural.
2. **`VERIFIED_POSITION_PROFILES` still has exactly one member.** Any second
   backbone needs its own measurement; the wrapper refuses it until then.
3. **Two probe sentences are integration evidence only** (§K).
4. **The collator is probe-shaped, not dataset-shaped.** A real Stage-1 data path
   will want bucketing and a `max_length` policy; neither is invented yet.
5. **Proposal PDF remains stale.**

---

## T. GIT STATE

`HEAD = 7f6e26c`, matching the run's Colab HEAD
`7f6e26c80c0acfa3cdf9168a9b0e2981e6ae1491`.

```
 M docs/experiments/b4b-phobert-adapter-integration-result.md
 M docs/spec/decisions.md
 M tests/test_b4b_provenance_and_positions.py
?? docs/audits/017-b4b-real-phobert-final-closure.md
```

Every change is left **unstaged**. No `add`, `commit`, `push`, `tag`, `stash`,
`reset`, `checkout` or `restore` was run. **No ML packages were installed locally;
no model weights were downloaded or loaded locally; no network was accessed;
nothing was trained; Stage-1 was not implemented.**

```text
AUDIT FILE WRITTEN: docs/audits/017-b4b-real-phobert-final-closure.md
OTHER AUDIT MODIFICATIONS MADE: NO
COMMIT CREATED: NO
```
