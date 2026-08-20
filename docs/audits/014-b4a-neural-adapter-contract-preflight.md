# Audit 014 — B4A neural adapter contract preflight

| | |
|---|---|
| **Audit id** | 014 |
| **Date (UTC)** | 2026-08-20 |
| **Scope** | Persist B3B-2 evidence; close deterministic B3B; derive the `A_φ` tensor contract |
| **Repository state** | `HEAD = c09516e`; this work uncommitted |
| **Predecessors** | [011](011-b3b1a-input-path-and-alignment-preflight.md), [012](012-b3b1b-whitespace-chunk-alignment-repair.md), [013](013-b3b1c-alignment-validation-and-channel-projection.md) |
| **Phase** | Phase 0 → Phase 1 boundary / B4A |
| **Type** | **Specification / preflight.** No `nn.Module`, no torch, no weights, no training |
| **Revised** | 2026-08-20 — researcher resolved all six B4A decisions; verdict updated from CONDITIONAL PASS to PASS. Original findings preserved. |
| **Revised (2)** | 2026-08-20 — consistency cleanup: Sections F and N still described D-B4A-006 as open; `tone_mask` added to the tensor contract; `special_tokens_mask` scope clarified. No scientific decision changed. |

---

## A. VERDICT

**PASS — B4B NEURAL IMPLEMENTATION READY**

This audit first returned **CONDITIONAL PASS — RESEARCHER DECISIONS REQUIRED
BEFORE B4B**, on three blocking ambiguities the proposal left unspecified. **The
researcher has now resolved all six B4A items.** The original findings are
preserved below in full; nothing is rewritten to look as though the ambiguities
were never there.

**The locked contract, stated explicitly:**

| Item | Locked value |
|---|---|
| Deterministic B3B | **COMPLETE** (D-B3B2-001), unchanged by this revision |
| Fusion and gate | The **proposal** is authoritative: an **input-dependent sigmoid gate** `g_i = σ(W_g[e_i;t_i;l_i] + c_g)` and a **convex combination** `z_i = g_i ⊙ f_i + (1 − g_i) ⊙ e_i` — not a raw gate vector, not a residual addition |
| Tone `NA` | **Fixed zero vector, not a learned row** |
| Tone table | **Exactly 7 trainable rows**, for every H4 policy |
| Letter table | **Exactly 5 rows** — `n_λ = 5` |
| Zero applicable letter contributors | **Exact zero vector** |
| Gate initialisation | `W_g = 0`, `c_g = logit(0.01) ≈ −4.59511985013459`, **initial `g = 0.01`** |
| Forced `g := 0` | **A wiring test only** — not a mode, not a condition, not an identity claim |
| Stage-1 pooling | **Attention-masked mean over non-special content tokens** |
| Parameter formula | **`\|φ\| = 6d² + 16d`** |
| Backbone | **D-B3B0-002 remains OPEN** — `d` stays symbolic, `hidden_size` has no default |
| PhoBERT position ids under `inputs_embeds` | **Remains a required B4B empirical check** — not verified, not a B4A blocker |

**No `nn.Module` was implemented. No model weights were loaded. Nothing was
trained.** Torch and transformers remain uninstalled.

## B. FILES CHANGED

| File | Change |
|---|---|
| `docs/experiments/b3b2-channel-projection-result.md` | **new** — final B3B-2 scientific evidence |
| `docs/spec/neural-adapter.md` | **new** — the `A_φ` contract extraction; revised with the six resolutions |
| `docs/spec/decisions.md` | D-B3B2-001 (closure) + D-B4A-001 … D-B4A-007, each now carrying its resolution beside its original ambiguity |
| `unmark-proposal.md` | **v1.3 → v1.4**, two narrow corrections (listed below) |
| `unmark/modeling/contracts.py` | **new** — torch-free enums, label ids, initialisation plan, Stage-1 pooling contract |
| `unmark/modeling/config.py` | **new** — torch-free config, shapes, parameter accounting |
| `unmark/modeling/__init__.py` | **new** — exports |
| `tests/test_adapter_contract.py` | **new** — 131 tests |

**Proposal changes (narrow, both from researcher decisions):**

* **§4.6** now states the Stage-1 pooling — attention-masked mean over
  non-special content tokens, with the fail-loud rule. Previously it locked
  "pooled representations only" without saying which pool, which is a scientific
  choice, not a detail. (D-B4A-006.)
* **§4.7** letter-table budget line `~10 × d ≈ 8K` → `5 × d ≈ 4K`, and **§8.2**'s
  sketch `n_letter=10` → `n_letter=5`. (D-B4A-007.)

Nothing about the orthographic taxonomy, the fusion equation, the gate, or the
tone table was changed. A v1.4 changelog entry records both.

No existing module was modified. **No `nn.Module` was written.**

## C. B3B-2 FINAL SCIENTIFIC EVIDENCE

Run `20260820T041812Z`, HEAD `c09516e03300e670fc20ac10173d7c346106fd6a`, status
**`B3B2_CHANNEL_PROJECTION_COMPLETE`**. `vinai/phobert-base` at revision
`01daacda68afe13d83023d16ec647239e344a1e6`, `PhobertTokenizer`, `is_fast=false`,
**model weights loaded: false**, Python 3.12.13.

**Seven test cases** — `plain`, `tones`, `letters`, `mixed_en`, `punct`, `url`,
`digits` — across **six conditions**. These are seven *cases*, not seven
conditions; the grid is 7 × 6 = 42.

| Measure | Result |
|---|---|
| Token grid invariant across all six conditions | 7/7 |
| Piece ranges invariant | 7/7 |
| Sequence consistent under every condition | 7/7 |
| Monotonic marked-tone degradation | 7/7 |
| `STRIP_ALL` leaves no marked tone | 7/7 |
| Multi-candidate authoritative pieces | **0** |
| Token ids matched the per-case `FULL` grid | **42/42** |
| Piece ranges matched | **42/42** |
| Sequence consistency | **42/42** |

Tone-label totals and letter applicable-label totals are transcribed exactly into
the experiment record. **Three checks I derived from the supplied table**, each of
which it would have failed had the projection been leaking:

* **Every condition totals 75 labelled positions** — the authoritative position
  count does not move.
* **Vietnamese positions constant at 50; `NA` constant at 25.** Corruption never
  converts a Vietnamese position into a non-applicable one, nor invents one.
* **Marked tones 34 → 21 → 15 → 7 → 0 → 0, with `UNMARKED` rising 16 → 29 → 35 →
  43 → 50 → 50 to compensate exactly.**

`P100` and `STRIP_ALL` reach zero marked tones while Vietnamese positions remain
**`UNMARKED`, not `NA`**.

For the letter channel, the **applicable-contributor count is invariant at 165**
across all six conditions: `FULL`…`P100` share identical counts (143 `NONE`, 10
`CIRCUMFLEX`, 6 `HORN`, 4 `STROKE`, 2 `BREVE`), and `STRIP_ALL` shows 165 `NONE`.
The 22 letter-forming marks become `NONE`, **not** `NA` — `NONE` ≠ `NA` survives
the strongest corruption.

**Honesty note.** These counts are recorded as supplied and externally inspected.
The local ML-free environment cannot re-execute the probe, and I did not
re-derive them; what I verified is their internal consistency, above.

**Scope discipline.** The monotonic result is recorded as **observed under the
locked B2 protocol and these seven cases** — explicitly *not* a universal theorem
about arbitrary corruption processes.

---

## D. DETERMINISTIC B3B CLOSURE

**Deterministic B3B: COMPLETE** ([D-B3B2-001](../spec/decisions.md)).

```
b(C_c(x)) = b(x)      hence      T(b(C_c(x))) = T(b(x))
```

The raw-BPE character-range structure is identical across conditions; **only
channel values change**. Locked contract: `RAW_BASE`; authoritative `T(b(x))`;
slow-tokenizer authority with alignment metadata never defining token ids;
deterministic raw-BPE range reconstruction; B1A/B3A metadata overlay;
unique-candidate tone ownership; ≥2 candidates → tone `NA`; letter `NONE`
included, `NA` excluded; corruption-invariant grid and ranges.

**D-B3B0-001: CLOSED. D-B3B0-002: REMAINS OPEN** — the probe revision is
reproducibility evidence, not the backbone decision. A test asserts this was not
closed by accident.

**Unchanged by the B4A resolutions.** Deterministic B3B **remains COMPLETE**;
nothing decided for the adapter reopens it.

Special-token integration is a **B4/B5 integration test**, not an open B3B
scientific blocker.

---

## E. PROPOSAL ARCHITECTURE EXTRACTION

Read in full: `unmark-proposal.md` v1.3 (699 lines), `docs/spec/orthography.md`,
`docs/spec/decisions.md`, audits 010–013, both experiment records, and the
`orthography` / `linguistics` / `corruption` / `alignment` packages.

**The single most consequential finding is a discrepancy between the proposal and
the project's own summary of it (DISC-1, [D-B4A-001](../spec/decisions.md)).**

The B4A brief describes the expected architecture as "a 3d → d fusion projection;
LayerNorm; a per-dimension gate", offering `e_b + g ⊙ LN(W[e_b;e_τ;e_λ] + b)` as a
candidate form. **The proposal specifies neither.**

| | Project-history summary | Proposal §4.5 + §5.1 |
|---|---|---|
| Gate | reads as a raw trainable vector `g ∈ R^d` | **`σ(W_g[e;t;l] + c_g)`** — a *second* `3d → d` projection, input-dependent per position |
| Combination | residual `e + g ⊙ f` | **convex** `g ⊙ f + (1 − g) ⊙ e` |
| Gate parameters | `d` | **`3d² + d`** |

Decided in favour of the proposal on three independent grounds: §4.5 states it;
§5.1 locks "Gate: per-dimension, σ(W_g[·])"; and §4.7's budget bills a **"Gate
projection (3d → d) ≈ 1.8M"** as a line item, which a `d`-sized vector cannot
account for. "Per-dimension" describes the gate's **output shape**, not a
position-independent parameter.

The forms are not interchangeable: under the convex form `g → 1` *replaces* the
base embedding; under the residual form the base term is always at full strength.

Two further internal inconsistencies in the proposal, both resolved in favour of
the locked sections and recorded so B4B does not implement the superseded text:

* **DISC-3.** §4.5 writes `l_i = W_λ[λ_i]`, a single lookup; §4.4 step 4 and §5.1
  lock mean pooling in embedding space. §4.5's form is shorthand.
* **DISC-4.** §8.2's sketch returns one `letter_id` per token, which is
  *incompatible* with embedding-space pooling — pooling needs the character
  labels *and* `W_λ`, which lives inside the module. The sketch is illustrative;
  §4.4/§5.1 are locked.

---

## F. TENSOR SHAPES

`B` batch · `L` authoritative sequence length · `d` encoder hidden size · `K`
per-token letter-contributor count.

| Tensor | Shape |
|---|---|
| `input_ids` | `[B, L]` |
| `attention_mask` | `[B, L]` |
| `special_tokens_mask` | `[B, L]` — scope below |
| base embeddings `e` | `[B, L, d]` |
| `tone_ids` | `[B, L]` — valid rows `0..6`; `NA` carried as sentinel `-1` |
| `tone_mask` | `[B, L]` — **true iff the position indexes one of the seven learned tone rows; false for `NA`** |
| tone embeddings `t` | `[B, L, d]` — zero wherever `tone_mask` is false |
| letter contributor ids | `[B, L, K]` **ragged** |
| letter contributor mask | `[B, L, K]` |
| pooled letter `l` | `[B, L, d]` — zero when no contributor is applicable |
| concatenated `[e;t;l]` | `[B, L, 3d]` |
| `W_f`, `W_g` | `[d, 3d]` each |
| fusion output `f` (post-LN) | `[B, L, d]` |
| gate `g` | `[B, L, d]` |
| `z` = `inputs_embeds` | `[B, L, d]` |
| encoder hidden states | `[B, L, d]` |
| Stage-1 pooled | `[B, d]` — **attention-masked mean over non-special content tokens** (D-B4A-006, RESOLVED) |

**`tone_mask` is load-bearing, not decoration.** D-B4A-002 puts `NA` outside the
embedding table, so the sentinel `-1` in `tone_ids` is not indexable. **The
sentinel must never reach `nn.Embedding` in B4B** — the mask selects between a
safe placeholder lookup and exact zero.

**Scope of `special_tokens_mask`.** It is not uniformly optional:

| Consumer | Requirement |
|---|---|
| Adapter fusion | **Not inherently needed** — if tone/letter `NA` metadata is already supplied, the channels are already zero at special tokens |
| **Stage-1 pooling** | **Required** — a special-token mask, or an exactly equivalent tokenizer-derived mask |
| B4B real-model integration | **Must construct and verify it** for real PhoBERT special tokens |

The two masks do different jobs and neither substitutes for the other:
`attention_mask` excludes **padding**; `special_tokens_mask` excludes **model
special tokens**. **Padding is never counted as content.**

**No maximum `K` was invented.** `K` is ragged and stays symbolic. Three
batching options (padded-dense with a per-batch `K_max`; flat + segment reduce;
precomputed sparse averaging matrix) are recorded as **equivalent by linearity of
the mean** — so the choice is an implementation decision, not a scientific one,
and B4B may make it without a researcher decision.

---

## G. BASE EMBEDDING CONTRACT

`e_i = Emb_θ(b_i)` — the **frozen encoder's input word embedding table**, at the
authoritative `T(b(x))` ids, with `RAW_BASE` (no post-strip segmenter).

`z` replaces the **word** embedding only. Position and token-type embeddings are
supplied by the encoder, exactly once, downstream of `inputs_embeds`. The
proposal calls double position encoding **"the single most likely implementation
bug in the project"** and notes the failure is silent: the model trains, the loss
decreases, every number is wrong.

The adapter's LayerNorm is its own parameter, distinct from the encoder's
embedding LayerNorm, which still runs on `z`.

**One risk I flag that the brief did not name.** PhoBERT is RoBERTa-family, and
RoBERTa derives position ids from `input_ids` through a **padding-aware offset**;
when only `inputs_embeds` is passed, the model falls back to sequential position
ids. Whether that fallback yields the *same* position ids the batch would have
received through `input_ids` is exactly the class of silent discrepancy §4.5
warns about. **B4B must check it directly** — I have not verified it, and no
local verification is possible.

---

## H. TONE CHANNEL CONTRACT

**Fixed.** Syllable-level, copied to every overlapping subword (§4.4 step 3).
Ownership by distinct-candidate count (D-B3B1C-001). Trainable (§4.7).
Dimensionality `d`. Lexical `NGANG` **absent** from the deployable path — genuine
*ngang* and stripped tone both map to `UNMARKED`; `UNMARKED` ≠ `NA`. Special
tokens and padding carry `NA`. Multi-candidate ambiguity enters as `NA` and
nothing else; the contributor list is audit metadata, not a model input.

**Deterministic mapping** (`OBSERVABLE`): `SAC 0, HUYEN 1, HOI 2, NGA 3, NANG 4,
UNMARKED 5` — five marked tones in the proposal's order, then policy slot A.

### The cardinality conflict — DISC-2, blocking

**Verified, not assumed, and the brief's expectation does not survive contact
with the proposal.** The brief says the deployable labels imply "7 observable
token slots if consistent with the proposal". The counts match; **the
compositions do not.**

The proposal's seven are `5 marked + slot A + slot B`. The repository's seven are
`5 marked + UNMARKED + NA`. **`NA` is absent from the proposal's tone table
entirely** — §4.3 and §4.4 require non-Vietnamese subwords to carry `N/A`, but no
slot is allocated to it.

| Reading | Consequence |
|---|---|
| **(a)** `NA` takes unused slot B | Works for `OBSERVABLE`/`FORCED-NGANG`; **breaks `ORACLE`**, which needs slot B for `MISSING` |
| **(b)** `NA` is an 8th row | Contradicts the §5.1 7-slot lock; gives `ORACLE` 8 rows against 7 — **defeats the H4 equalization**, whose stated purpose is to remove "any objection that the oracle was granted extra capacity" |
| **(c)** `NA` is not a row: zero vector or masked | Preserves 7 rows **and** H4 equalization. **The proposal never says it** |

Under (c) the position is fused with a zero tone contribution but is *not* inert —
`[e_i ; 0 ; l_i]` still passes through `W_f`/`W_g` and the biases. Under (a)/(b)
the model gets a **learned** "this is not Vietnamese" vector. Materially different
inductive biases.

### RESOLVED — D-B4A-002: reading (c)

**Tone `NA` is the fixed zero vector, not a learned row.** `t_i = 0 ∈ R^d` at
non-Vietnamese positions, special tokens, padding, and multi-candidate ambiguous
pieces. **The tone table stays at exactly 7 trainable rows for every H4 policy.**
`UNMARKED` remains a genuine learned observable row and stays distinct from `NA`.

`NA` costs zero parameters, so `|φ|` is identical across all three policies and
the **H4 equalization holds exactly**. Under `OBSERVABLE` the 7-row table is
still allocated in full even though slot B is never indexed — dropping the unused
row would itself break the equalization.

**Batching:** `tone_ids: [B, L]` with valid rows `0..6`, plus `tone_mask: [B, L]`;
`NA` carried as the out-of-table sentinel `-1`. **A torch implementation must
never feed that sentinel to `nn.Embedding`** — safe placeholder lookup plus
masking, or equivalent, with the result forced to exact zero.

The contract **rejects the alternatives by name** rather than merely documenting
them: `EXTRA_ROW` and `SLOT_B_ROW` both raise `LockedContractViolation`.

---

## I. LETTER CHANNEL CONTRACT

**Fixed.** Character-level labels; pooled **in embedding space**, arithmetic mean,
v1 (§4.4 step 4, §5.1); `NONE` **included**, `NA` **excluded**; zero applicable
contributors → token letter channel `NA` (D-B3B1C-001). Learned attention pooling
is an ablation. Trainable, dimensionality `d`.

**Cardinality (D-B4A-007, non-blocking).** §4.7 bills `~10 × d` and §8.2 defaults
`n_letter=10`; §4.3 writes `{NONE, breve, circumflex, horn, stroke,
circumflex+…}`. B1A determined the applicable closed set is **5**. The
anticipated combination states do not arise: Vietnamese places **at most one**
letter-forming mark per character (`ă â ê ô ơ ư`, `đ` stroke). `~10` is a budget
estimate, not a lock — §5.1 says only "closed set". Cost difference `(10−5)·d ≈
3.8K` against ≈3.6M. The §4.7 table should be corrected when `n_λ` is finalised.

**Empty-contributor vector (D-B4A-005, was blocking).** D-B3B1C-001 locks the
*label-space* semantics but not the *vector*. Options: fixed zero; learned `NA`
row; masked out — the last changes the concatenation **width** and is therefore
incompatible with a fixed `W_f ∈ R^(d×3d)`. A masked mean over zero contributors
is `0/0`; unguarded it yields `NaN` and silently poisons the batch. **B3B-2
recorded 25 `NA` positions per condition, so real data exercises this on every
batch** — not a rare edge case.

### RESOLVED — D-B4A-005 and D-B4A-007

**The letter table is exactly 5 rows: `n_λ = 5`** — `NONE`, `BREVE`,
`CIRCUMFLEX`, `HORN`, `STROKE`. `NA` is not a trainable row.

**Zero applicable contributors → the exact zero vector:**

```
|A_i| > 0   →   l_i = (1 / |A_i|) · Σ_{j ∈ A_i} W_λ[label_ij]
|A_i| = 0   →   l_i = 0 ∈ R^d
```

`NONE` included in the mean, `NA` excluded — D-B3B1C-001 unchanged.

**The implementation must explicitly prevent `0/0`.** A torch implementation may
clamp the denominator for vectorisation, **but only if the zero-contributor
output is then explicitly forced to exact zero**. Clamping alone leaves
`sum(∅)/1 = 0` true by accident rather than by contract, and the accident stops
holding the moment the numerator stops being empty-safe.

Proposal §4.7 and §8.2 corrected in v1.4. **The orthographic taxonomy is
unchanged.**

---

## J. EXACT FUSION EQUATION

Verbatim from §4.5, not paraphrased:

```
e_i = Emb_θ(b_i)          t_i = W_τ[τ_i]          l_i = W_λ[λ_i]

f_i = LN( W_f [ e_i ; t_i ; l_i ] + c_f )

g_i = σ( W_g [ e_i ; t_i ; l_i ] + c_g ) ∈ (0,1)^d

z_i = g_i ⊙ f_i + (1 − g_i) ⊙ e_i
```

§5.1 locks the ordering independently: *"LayerNorm after fusion, before the gate
combination"*. Resolved order, no ambiguity remaining:

```
concat [e;t;l] → W_f · + c_f → LayerNorm → f
concat [e;t;l] → W_g · + c_g → σ         → g
z = g ⊙ f + (1 − g) ⊙ e
```

Gate transform: **σ, locked**. Gate **initialisation was unspecified —
D-B4A-003.** `c_g = 0` starts at `g ≈ 0.5` (adapter perturbs every embedding from
step 0); strongly negative starts near base-only; strongly positive starts at full
fusion. G1's criterion is "force the gate towards identity … pass within ≈1
point", so the choice materially affects whether G1 measures the architecture or
the initialisation.

Note on phrasing: "initialized at zero" and "initialized so its **effect** is
zero" are different requests, and under `σ` **only the first is achievable** — as
a bias value, never as a gate output.

### RESOLVED — D-B4A-003

```
W_g = 0
c_g = logit(0.01) = ln(0.01 / 0.99) ≈ −4.59511985013459
```

for every output dimension, so **at initialisation `g_i = 0.01`** for every token
and every dimension, before any learning. The locked sigmoid-gate architecture is
unchanged.

Three reasons: start close to the pretrained base-only pathway; avoid `c_g = 0 →
g = 0.5`, which would inject a **randomly initialised** fusion branch at half
weight on step zero; and retain a usable sigmoid derivative — `g(1−g) ≈ 0.0099` —
so the gate projection can still learn. An initialisation driving `g` to machine
zero would drive the derivative there too, and the gate could never open.

`W_g = 0` makes the gate input-independent at step zero: the concatenated
channels drop out and every position starts at `σ(c_g)`.

**This is the main initialisation and is config-logged**, surfaced as
`AdapterConfig.initialisation_plan()`. **No exact base-only equality is
claimed** — `g = 0.01` is close to that pathway, not equal to it.

---

## K. GATE-ZERO RECOVERY

**Confirmed against the proposal.** `g → 0` recovers the **base-only pathway**,
not the clean-text pathway. §4.5 explicitly corrects an earlier draft that
claimed otherwise: because `e_i = Emb_θ(b_i)` is computed from the *stripped*
stream, `E_θ(T(x))` is unreachable at any gate value.

**Base-only pathway means precisely:** the same authoritative `T(b(x))` ids; the
same special tokens; the same attention mask; the same frozen encoder; the
encoder's own word-embedding lookup; and no tone or letter contribution.

**The exact numerical condition — and why it cannot be met.** `z_i = e_i`
requires, per position and dimension:

```
g_i[k] · ( f_i[k] − e_i[k] ) = 0
```

so `g_i[k] = 0` **or** `f_i[k] = e_i[k]`. **Neither is attainable:**

1. `g = σ(·) ∈ (0,1)^d`, an **open** interval. `σ(u) = 0` only as `u → −∞`.
2. `f = LN(·)` is normalised per position; `e_i = Emb_θ(b_i)` is not. Equality
   for all `i` would need the LN affine parameters to invert normalisation for
   every token simultaneously.

**This is not a proposal inconsistency, and I am not reporting it as one.** §4.5
writes `g_i → 0` — a limit — and never claims exact recovery. The brief asked
whether the parameterization can achieve exact recovery; **it cannot**, and the
consequence is confined to the *test plan*: the "gate-zero numerical equivalence"
check cannot be an exact equality test on the trained parameterization.

Options (D-B4A-004): a forced `g := 0` override — exact, but tests wiring rather
than the parameterization; a tolerance test with `c_g` driven very negative; or
reparameterizing the gate to attain zero, which changes the locked architecture.

### RESOLVED — D-B4A-004

**Forced `g := 0` is a wiring test only.** Under an explicit test override,
`z == e` must hold **exactly**, up to ordinary floating-point arithmetic — the
identity `g·f + (1−g)·e = e` at `g = 0`, for any `f`.

It is **not** a trainable parameterization, **not** an experiment condition,
**not** a claim that `σ` attains zero, and **not** evidence that the initialised
module is identity.

**No casual production "gate zero mode" is exposed**, because such a mode could
silently enter an experiment. B4B should test the fusion-combination primitive or
internal path directly rather than adding a public flag; a test asserts no
gate-zero flag exists on the config.

**Separately, B4B must measure the real initialised gate at `g = 0.01` against
the base-only pathway and report the difference — expected to be nonzero.**
Reporting zero there would mean something is wrong.

---

## L. FROZEN / TRAINABLE PARAMETER PARTITION

| Frozen (`θ`) | Trainable (`φ`) |
|---|---|
| Word / input embedding table `Emb_θ` | Tone table `W_τ` |
| Positional embeddings | Letter table `W_λ` |
| Token-type embeddings | Fusion `W_f`, bias `c_f` |
| Encoder embedding LayerNorm | Adapter LayerNorm (gain, bias) |
| All transformer blocks | Gate `W_g`, bias `c_g` |
| All pretrained LayerNorms | |
| Pooler, if present | |

§5.1: "fully frozen; no layer unfrozen without a logged decision" — the config
raises rather than accepting `encoder_frozen=False`, so unfreezing requires a
decision-log entry, not a flag.

**The Stage-2 task head is excluded from `φ`**: §8.3 trains it in a separate
stage with the module frozen, and §4.7's budget lists no head. A test asserts no
head term appears in the count.

---

## M. INPUTS_EMBEDS / SPECIAL-TOKEN CONTRACT

B3B-2 operated on ordinary tokenizer positions and observed **no model-added
special tokens**. This is **not** a B3B blocker; it is deferred integration
testing.

**None of the following is empirically validated. B4B must verify:** exact
PhoBERT special-token ids and order; final `L` after special-token construction;
base embedding lookup for special tokens; tone `NA` at special tokens; letter
`NA` at special tokens; no fabricated source range; padding `NA`; `attention_mask`
unchanged; **position embeddings applied exactly once**, including the
RoBERTa-family position-id question in §G; `inputs_embeds` forward shape equal to
`input_ids` forward shape; and forced g := 0 wiring equivalence per D-B4A-004, clearly separated from the normal initialized g = 0.01 pathway.

---

## N. STAGE-1 INTERFACE

Interface only; no Stage-1 training implemented.

```
L_align = D( h′(x̃ₚ), h(x) )
L_clean = D( h′(x),   h(x) )
L       = λ_a · L_align + λ_c · L_clean
```

Reference branch `h(x)`: same frozen encoder, **without** the module. Adapted
branch `h′`: adapter + frozen encoder, corruption applied to the **tone channel**
with the base grid unchanged. Corruption `p ~ U(0,1)` per example, continuous.
Gradients to `φ` only; all of `θ` frozen. Distance: cosine, pooled.

### Pooling representation — HISTORICAL ambiguity, now RESOLVED

**Original ambiguity (as first found by this audit).** §4.6 locked "pooled
representations only" but did not select the pool, and §5.2/§13 left head pooling
open. `CLS` vs mean vs attention-masked mean differ materially under padding: an
unmasked mean averages `<pad>` embeddings in, a silent systematic bias that
varies with batch composition. It was marked OPEN rather than chosen.

**Researcher resolution — D-B4A-006: attention-masked mean over non-special
content tokens.** From the final encoder hidden state, computed **independently
per branch**. Per example:

```
content_mask = attention_mask AND NOT special_tokens_mask

pooled = sum(content_mask * H) / sum(content_mask)
```

Excluded: `<s>`, `</s>`, `<pad>`, and every other tokenizer or model special
token. **Padding is never counted as content.**

**Zero content positions after masking: FAIL LOUD.** No silent fallback to `<s>`,
to an unmasked mean, or to a zero vector — each would hand the cosine objective a
value representing nothing.

**The reference and adapted branches may have different sequence lengths.**
`h(x)` runs the encoder's own tokenization of clean text; `h′` runs the base
grid. Mean pooling maps both to `R^d` independently and **no per-token
correspondence is assumed** — which is precisely why §4.6 defers per-token
alignment. Any implementation assuming a shared `L` across branches is wrong.

Persisted into proposal §4.6 in v1.4, because this is a scientific decision
rather than an implementation detail.

---

## O. PARAMETER COUNT

| Component | Parameters |
|---|---|
| Tone table | `n_τ · d` |
| Letter table | `n_λ · d` |
| Fusion `W_f` | `3d²` |
| Fusion bias `c_f` | `d` |
| Gate `W_g` | `3d²` |
| Gate bias `c_g` | `d` |
| Adapter LayerNorm | `2d` |

```
|φ| = 6d² + (4 + n_τ + n_λ) · d
```

**With `n_τ = 7` (D-B4A-002) and `n_λ = 5` (D-B4A-007):**

```
|φ| = 6d² + 16d
```

**Arithmetic sanity check.** At `d = 768` this gives **3,551,232** — reproducing
§4.7's ≈3.6M within its own rounding, and **only** under DISC-1's reading, since
the budget can be matched at all only if the gate is a `3d → d` projection
(fusion 1.77M, gate 1.77M, tone ≈5K, letter ≈4K, LN ≈2K). **This is independent
corroboration of DISC-1.**

**`d = 768` is not locked.** It is used only to check arithmetic — D-B3B0-002 is
OPEN, `hidden_size` has no default, and a test asserts the contract modules
contain no hardcoded `768`.

**H4 consistency (H4 itself not implemented).** All three policies share one
`n_τ = 7` table, so `|φ|` is **identical** across `OBSERVABLE`, `FORCED-NGANG`
and `ORACLE` — the point of the equalization. **D-B4A-002 keeps `NA` outside the
table precisely so this stays true.** `h4_equalized()` holds under the locked
contract, and the extra-row alternative is now **rejected** rather than merely
shown to break it.

---

## P. SPECIFICATION QUESTIONS — ALL RESOLVED

This audit originally found six unspecified items, three of them blocking. **All
six are now resolved by researcher decision.**

| Id | Original question | Was blocking? | Resolution |
|---|---|---|---|
| **D-B4A-002** | `NA` has no row in the 7-slot tone table | **YES** | Fixed **zero vector**, outside the table; table stays at **7 rows** for every policy |
| **D-B4A-003** | Gate initialisation | **YES** | `W_g = 0`, `c_g = logit(0.01)`, **initial `g = 0.01`** |
| **D-B4A-005** | Empty letter-channel vector | **YES** | **Exact zero vector**; `0/0` explicitly prevented |
| D-B4A-004 | Gate-zero test form | No | Forced `g := 0` is a **wiring test only** |
| D-B4A-006 | Stage-1 pooled representation | No | **Attention-masked mean over non-special content**, fail loud on zero content |
| D-B4A-007 | `n_λ` final value | No | **`n_λ = 5`**; formula becomes `6d² + 16d` |

Each is logged in `docs/spec/decisions.md` with **the original ambiguity
preserved beside the resolution** — the ambiguity, the decision, the reason, the
mathematical consequence, what it affects, and proposal-update status. The
evidence chain is not erased: a contract that shows why a choice was needed is
more useful than one presenting it as obvious.

**The rejected alternatives are retained in the enums and rejected by name.**
`ToneNaTreatment.EXTRA_ROW`, `ToneNaTreatment.SLOT_B_ROW`,
`LetterEmptyTreatment.LEARNED_NA_ROW`, `LetterEmptyTreatment.MASKED_OUT` and
`GateInit.ZERO_BIAS` all raise `LockedContractViolation` naming their decision
id. Deleting them would let a later reader rediscover them as plausible options
rather than seeing them refused.

**Still open, and not a B4A item:** [D-B3B0-002](../spec/decisions.md#d-b3b0-002),
the backbone checkpoint. `d` stays symbolic; a test asserts it was not closed by
accident.

---

## Q. TEST EVIDENCE

```
.venv/bin/python -m pytest -q -p no:cacheprovider
1773 passed in 6.90s
```

`tests/test_adapter_contract.py` now holds **131 tests**. Baseline before B4A
began: 1642; after the preflight: 1692; after the resolutions: 1739; after this
consistency cleanup: **1773**. **All previously green tests remain green** — no
test was deleted to accommodate a resolution; the ones that asserted "this item
is UNDECIDED" were rewritten to assert the locked value and to reject the
alternatives.

Covering, beyond the original set: tone table exactly 7 trainable rows for every
policy; tone `NA` outside the table and negative-sentinel; tone `NA` → exact zero
vector; `UNMARKED` a learned row while `NA` is not; the sentinel documented as
never indexing a table; unused policy slot still costing a row; H4 equalization
holding; a learned `NA` row **rejected**; slot-B reuse **rejected**; any other
tone-table size **rejected**; letter table exactly 5 rows; `NONE` a valid learned
row; letter `NA` outside the table; empty contributor set → exact zero vector;
`0/0` explicitly forbidden in the contract text; `LEARNED_NA_ROW` and `MASKED_OUT`
**rejected**; dropping `NONE` or admitting `NA` to the pool **rejected**; the
`6d² + 16d` formula across six dimensions; the formula string matching the
arithmetic; `d = 768` → 3,551,232 while `AdapterConfig()` still raises;
`n_τ = 7` and `n_λ = 5` entering the count; `W_g = 0`; `c_g = logit(0.01)` to
1e-12; planned initial gate exactly 0.01; zero weight making the initial gate
input-independent; a usable initial derivative; the rejected `g = 0.5`
alternative; initialisation **not** claiming base-only equality; forced `g := 0`
distinct from initialisation; the convex-combination wiring identity; no public
gate-zero flag; Stage-1 masked mean; special tokens excluded; padding excluded;
unequal branch lengths permitted; zero content **failing loud**; no silent
fallback; mismatched mask lengths rejected; the masked-mean formula on a scalar
stand-in; zero channels still passing through fusion; no special-token bypass.

**Added in the consistency cleanup**: `tone_mask` present in the tensor contract
and sharing `tone_ids`' shape; its sentinel semantics documented;
`special_tokens_mask` present; the two masks shown **not** to substitute for each
other; padding never counted as content; the audit verdict being `PASS` while
still recording the historical `CONDITIONAL PASS`; the audit still stating each
locked value; the position-id question still presented as unverified B4B work;
and the Stage-1 formula present verbatim.

**A regression guard on this class of defect.** Two sentences survived the first
revision claiming a resolved item was still open — one in Section F, one in
Section N. `test_no_stale_current_state_claims_in_b4a_docs` now scans all three
B4A documents for seven stale current-state phrases. It caught a third instance
in `decisions.md` on its first run, which is the reason it exists: the guard is
on the class, not on the two instances I happened to find by eye.

**Environment**: no packages installed; `torch`, `transformers`, `sentencepiece`,
`datasets`, `py_vncorenlp` all absent from `.venv`; no network; no model
downloads. AST tests assert the contract modules import none of them and define
no `nn.Module` subclass or `forward` method.

---

## R. BLOCKING ISSUES

**None.**

The three that blocked when this audit was first written — D-B4A-002, D-B4A-003,
D-B4A-005 — are resolved. **B4B may begin the `nn.Module`.**

Two constraints carry into B4B and are requirements rather than blockers:

1. **The `NA` sentinel must never reach `nn.Embedding`.** Safe placeholder lookup
   plus masking, or equivalent, with the result forced to exact zero.
2. **`0/0` must be explicitly prevented** in the letter mean. Clamping the
   denominator is allowed **only** if the zero-contributor output is then forced
   to exact zero.

---

## S. NON-BLOCKING ISSUES

1. **PhoBERT position ids under `inputs_embeds` — remains a required B4B
   empirical check.** RoBERTa-family models derive position ids from `input_ids`
   via a padding-aware offset and fall back to sequential ids when only
   `inputs_embeds` is passed. **This is not verified and must not be "fixed" in
   pure-data B4A code.** B4B must compare the authoritative `input_ids` path
   against the `inputs_embeds` path under **no padding**, **right padding**,
   **unequal sequence lengths within one batch**, and **real PhoBERT special
   tokens**, inspecting the **actual position ids used** rather than inferring
   them. If they differ, B4B must supply explicit `position_ids` derived from the
   `input_ids` behaviour. §4.5 calls double position encoding "the single most
   likely implementation bug in the project".

2. **Zero channels do not make the adapter inactive.** Tone `NA = 0` and letter
   empty `= 0` mean those *channels* contribute nothing; the fusion still
   receives the full-width `[e_i ; 0 ; 0]` and `W_f`/`W_g` may still change
   `z_i`. **No special-token bypass was added**, because the proposal specifies
   none. Whether the adapter perturbs special-token embeddings is a consequence
   of the locked architecture, not a new exception.

3. **DISC-4** — §8.2's interface sketch is superseded by §4.4/§5.1 and must not
   be implemented as written. Its `n_letter` default is corrected in v1.4, but
   its one-`letter_id`-per-token signature remains incompatible with
   embedding-space pooling.

4. **Compiled proposal PDF is stale.** The editable source moved v1.3 → v1.4.

---

## T. GIT STATE

`HEAD = c09516e`, matching the B3B-2 scientific run's HEAD
`c09516e03300e670fc20ac10173d7c346106fd6a`.

```
 M docs/spec/decisions.md
 M unmark-proposal.md
?? docs/audits/014-b4a-neural-adapter-contract-preflight.md
?? docs/experiments/b3b2-channel-projection-result.md
?? docs/spec/neural-adapter.md
?? tests/test_adapter_contract.py
?? unmark/modeling/
```

Every change is left **unstaged**. No `add`, `commit`, `push`, `tag`, `stash`,
`reset`, `checkout` or `restore` was run. **No torch or transformers was
installed; no `nn.Module` was implemented; no model weights were loaded; nothing
was trained.** No Audit 015 was created — this file was revised in place, as
instructed, with the original findings preserved.

```text
AUDIT FILE WRITTEN: docs/audits/014-b4a-neural-adapter-contract-preflight.md
OTHER AUDIT MODIFICATIONS MADE: NO
COMMIT CREATED: NO
```
