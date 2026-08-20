# Stage-1 objective and data path — implementation preflight

**This is not a scientific result.** No model weights were loaded, no corpus was
read, and nothing was trained. It records what Stage-1 code now exists, what its
contracts are, and what remains scientifically OPEN.

| | |
|---|---|
| **Phase** | Stage-1A — objective and data path |
| **Predecessor** | [Audit 017](../audits/017-b4b-real-phobert-final-closure.md) — B4B COMPLETE |
| **Model weights loaded** | **no** (local environment is ML-free) |
| **Training performed** | **no** |
| **Optimizer implemented** | **no** |

## What now exists

| Module | Responsibility | torch |
|---|---|---|
| `unmark/stage1/contracts.py` | enums, `ObjectiveWeights`, `TruncationPolicy`, `CorruptionRatePolicy`, the OPEN/LOCKED registers | **no** |
| `unmark/stage1/data.py` | clean text → three prepared branches → padded batch | lazy, in `collate_stage1_batch` only |
| `unmark/stage1/objective.py` | cosine distance, `Stage1LossResult`, `Stage1Objective` | yes |
| `unmark/stage1/__init__.py` | re-exports the torch-free surface only | **no** |

`unmark.stage1` stays importable in the ML-free environment; the objective is
imported explicitly.

## The three branches

```
PATH R  x ──canon──► T(canon(x)) ─────────────► frozen E_θ ─► masked mean ─► h_ref
PATH C  x ──canon──► b(x) ─► T(b(x)) ─► clean channels ─► A_φ ─► frozen E_θ ─► h'(x)
PATH K  x ──canon──► C(·,p) ─► b(x̃) ─► T(b(x̃)) ─► corrupt channels ─► A_φ ─► frozen E_θ ─► h'(x̃)
```

* **PATH R uses no adapter, no channels and no `b(x)`.** It is the pretrained
  clean target, and runs under `no_grad`.
* **PATHS C and K share one base grid.** `b(C(x)) = b(x)` is *verified per
  example*, not assumed (D-S1A-002).
* The reference target for both adapted branches is `h(x)` from the **original
  clean** text. No restored text is fed anywhere and there is no restoration
  term.

## Loss

```
L_align = D( h'(x̃_p), h(x) )
L_clean = D( h'(x),    h(x) )
L       = λ_a · L_align + λ_c · L_clean
```

`D` is cosine distance over the feature dimension, per example, on pooled
`[B, d]` representations. Aggregation is the **mean** over the batch — a sum
would scale with batch size and silently change the effective learning rate.

**No token-level alignment.** The reference and base sequences have different
lengths in general; each branch is pooled independently and only the pooled
vectors are compared. That is precisely why D-B4A-006 locked pooled alignment.

## Tensor contract

Two independent padding domains, and deliberately unambiguous names:

| Field | Shape |
|---|---|
| `reference_input_ids` / `_attention_mask` / `_special_tokens_mask` | `[B, L_ref]` |
| `base_input_ids` / `_attention_mask` / `_special_tokens_mask` | `[B, L_base]` |
| `clean_tone_ids` / `corrupt_tone_ids` | `[B, L_base]` |
| `clean_tone_mask` / `corrupt_tone_mask` | `[B, L_base]` |
| `clean_letter_ids` / `corrupt_letter_ids` | `[B, L_base, K]` |
| `clean_letter_mask` / `corrupt_letter_mask` | `[B, L_base, K]` |
| pooled representation, each branch | `[B, d]` |

`L_ref ≠ L_base` is expected. `K` is the batch maximum. Channel semantics are
unchanged from B4B: tone `NA` is sentinel `-1` plus a false mask; `NONE`
participates in letter contributors; `NA` is excluded; special tokens and padding
are `NA` in both channels.

## Gradient and freezing

* Reference branch: `no_grad`, target only.
* Adapted branches: full graph, **nothing detached**, so gradients reach `A_φ`
  through the frozen encoder.
* One frozen `θ` serves all three branches — no second pretrained model is
  loaded for the reference.
* `Stage1Objective.train()` delegates to `UnmarkEncoder`, so the frozen encoder
  stays in eval (D-B4B-004).
* `position_ids` is omitted so the wrapper derives and enforces the authoritative
  values (D-B4B-002). Stage-1 does not reimplement that.

## What remains OPEN

`lambda_align`, `lambda_clean`, the Stage-1 corpus, `max_length` and overflow
behaviour, the corruption redraw schedule, the optional letter-dropout rate, the
Stage-1 seed, batch size, optimizer, learning rate, epochs/steps,
warmup/scheduler, gradient accumulation, checkpoint selection, and backbone
finalisation (D-B3B0-002).

The machine-readable register is `OPEN_STAGE1_VALUES`; `require_resolved(name)`
raises for any of them. **The existence of a config field does not mean a value
is decided.**

**No experiment-facing default exists for any of them.** `ObjectiveWeights`
requires both lambdas; `TruncationPolicy` requires both `max_length` and
`on_overflow` and has no no-argument form; `prepare_example` requires
`truncation` and `visit`. An **explicit** `TruncationPolicy.unbounded()` is a
caller statement, not a default.

Locked and implemented: `p ~ U(0,1)` per example continuous, cosine distance,
pooled-only alignment, masked-mean-over-non-special pooling, fully frozen encoder.

## Scientific values versus diagnostic-only values

`Stage1RunConfig` carries a required `Stage1Purpose` (D-S1A-006):

* **`DIAGNOSTIC`** — explicit values used only to exercise a forward/backward
  path, with no optimizer and no parameter update. The upcoming dry run may pass
  `lambda_align = 1.0`, `lambda_clean = 1.0` and an explicit `max_length` on this
  footing. **It resolves nothing.** `to_dict()` stamps `diagnostic_only: true`
  and `values_are_scientific: false` into the run artifact.
* **`SCIENTIFIC`** — **cannot be constructed** until every entry of
  `SCIENTIFIC_REQUIRED_VALUES` is named as resolved. It raises today.

So a diagnostic number cannot silently become a training default: the scientific
configuration does not exist as an object yet.

## Next required steps, in order

1. **Real-model Stage-1 dry run** on Colab — real PhoBERT, all three branches,
   one diagnostic backward permitted, **no optimizer and no parameter update**,
   `purpose = DIAGNOSTIC`. Confirms finite loss components, correct branch shapes
   with `L_ref ≠ L_base`, and gradient routing into `A_φ` with none into `θ`.
2. **Resolve the scientific values** needed to define the runner and its run
   configuration.
3. **Implement the Stage-1 training runner.** Optimizer, scheduler and
   checkpointing may exist *in code*; **no scientific training is run**.
4. **The repository-wide PRE-TRAIN audit**, which inspects the runner from step 3.
   It comes *after* the runner because its purpose is to check the **complete
   training path** — auditing a repository with no runner would inspect
   everything except the part that trains (D-S1A-007).
5. **Only if that audit PASSes:** the first scientific Stage-1 training run.

**The OPEN values and the PRE-TRAIN audit block scientific Stage-1 training. They
do not block step 1**, which is a real-model integration dry run with explicit
diagnostic-only values and no optimizer.
