# Neural adapter contract — `A_φ`

**Status: SPECIFICATION ONLY — COMPLETE.** No `nn.Module` exists. No torch is
installed. This document extracts the adapter contract from
`unmark-proposal.md` v1.4 and the decision log.

Audit 014 first found six items the proposal left unspecified, three of them
blocking. **All six are now resolved by researcher decision** (D-B4A-002 …
D-B4A-007). The original ambiguities are preserved below rather than overwritten,
because a contract that shows why a choice was needed is more useful than one
that presents it as obvious. **B4B is unblocked.**

Every claim below is tagged:

| Tag | Meaning |
|---|---|
| **[P]** | Fixed by the proposal |
| **[D]** | Fixed by a prior recorded decision |
| **[I]** | Implementation consequence — follows mechanically, no scientific choice |
| **[R]** | The proposal left it unspecified; **resolved by researcher decision** |

---

## 1. Notation

`B` batch size · `L` authoritative encoder sequence length · `d` pretrained
encoder hidden / input-embedding dimension · `n_τ` tone table rows · `n_λ` letter
table rows.

`d` is **not** substituted with 768 anywhere in this document.
[D-B3B0-002](decisions.md#d-b3b0-002) (backbone checkpoint) is **OPEN**, so the
contract stays backbone-parameterized.

---

## 2. The fusion equation, verbatim

**[P]** §4.5, reproduced exactly rather than paraphrased:

```
e_i = Emb_θ(b_i)          t_i = W_τ[τ_i]          l_i = W_λ[λ_i]

f_i = LN( W_f [ e_i ; t_i ; l_i ] + c_f )

g_i = σ( W_g [ e_i ; t_i ; l_i ] + c_g ) ∈ (0,1)^d

z_i = g_i ⊙ f_i + (1 − g_i) ⊙ e_i
```

**[P]** §5.1 locks the ordering independently: *"Normalisation: LayerNorm after
fusion, before the gate combination"*, and *"Gate: per-dimension, σ(W_g[·])"*.

Resolved order, with no remaining ambiguity:

```
concat [e;t;l]  →  W_f · + c_f  →  LayerNorm  →  f
concat [e;t;l]  →  W_g · + c_g  →  σ          →  g
z = g ⊙ f + (1 − g) ⊙ e
```

### 2.1 Discrepancy with project history — DISC-1

The B4A task brief describes the expected architecture as *"a 3d → d fusion
projection; LayerNorm; a per-dimension gate"*, and offers
`e_b + g ⊙ LN(W[e_b;e_τ;e_λ] + b)` as a candidate form. **Both differ materially
from the proposal, and the proposal wins.** Reporting rather than selecting:

| | Project-history summary | Proposal §4.5 + §5.1 |
|---|---|---|
| Gate | reads as a raw trainable vector `g ∈ R^d` | **`σ(W_g[e;t;l] + c_g)`** — a *second* `3d → d` projection, input-dependent, per position |
| Combination | residual `e + g ⊙ f` | **convex** `g ⊙ f + (1 − g) ⊙ e` |
| Gate parameters | `d` | **`3d² + d`** |

"Per-dimension" in §5.1 describes the gate's **output shape** — one scalar per
hidden dimension per position — not a position-independent parameter vector.

§4.7's budget confirms the reading: it bills a *"Gate projection (3d → d) ≈
1.8M"* as a line item beside the fusion projection. A raw `g ∈ R^d` vector would
cost `d`, not 1.8M. **The proposal is internally consistent; the summary is not
a faithful compression of it.**

The two forms are not interchangeable. Under the convex form the base term is
attenuated by `(1 − g)`; under the residual form the base term is always present
at full strength. They differ in what `g → 1` means: the convex form *replaces*
the base embedding, the residual form *adds* to it.

### 2.2 Letter-channel notation — DISC-3

**[P]** §4.5 writes `l_i = W_λ[λ_i]`, a single lookup. **[P]** §4.4 step 4 writes
`l_i = Pool({ W_λ[λ_c] : c ∈ span(token_i) })` with mean pooling, and **[P]**
§5.1 locks *"character level, mean-pooled in embedding space"*.

These are inconsistent as written. §4.4 and §5.1 are the reasoned, locked
statements — §4.4 argues at length that collapsing to one categorical label
"would reintroduce, at subword level, exactly the information loss that moving
the channel to character level was meant to remove". **§4.5's `W_λ[λ_i]` is
shorthand for the pooled vector.** Resolved in favour of pooling; recorded so a
later reader does not implement the shorthand.

---

## 3. Tensor contract

`K` denotes the per-token letter-contributor count. **It is ragged and is not a
fixed constant** — see §6.2. No maximum `K` is invented here.

| Tensor | Shape | dtype | Source |
|---|---|---|---|
| `input_ids` | `[B, L]` | int | **[D]** `T(b(x))`, slow tokenizer, authoritative |
| `attention_mask` | `[B, L]` | int/bool | tokenizer; **[P]** passed to the encoder unchanged |
| `special_tokens_mask` | `[B, L]` | bool | **[R]** **required for Stage-1 pooling**; scope below |
| `tone_ids` | `[B, L]` | int `∈ [0, 7)`, or `-1` for `NA` | **[D]** unique-candidate ownership |
| `tone_mask` | `[B, L]` | bool | **[R]** true iff the position indexes one of the seven learned rows; false for `NA` — §5.1 |
| `e` base embeddings | `[B, L, d]` | float | **[P]** `Emb_θ(input_ids)`, frozen table |
| `t` tone embeddings | `[B, L, d]` | float | **[P]** `W_τ[tone_ids]`; **[R]** exactly zero where `tone_mask` is false |
| `letter_ids` contributors | `[B, L, K]` ragged | int `∈ [0, 5)` | **[D]** per-character labels |
| `letter_mask` | `[B, L, K]` | bool | **[I]** true where a contributor is applicable |
| `l` pooled letter | `[B, L, d]` | float | **[P]** mean over contributors *in embedding space*; **[R]** exactly zero when none is applicable |
| `[e ; t ; l]` | `[B, L, 3d]` | float | **[P]** §4.5 concatenation |
| `W_f` | `[d, 3d]` | float | **[P]** single linear projection |
| `c_f` | `[d]` | float | **[P]** explicit in §4.5 |
| `f` fusion output | `[B, L, d]` | float | after `W_f` **and** LayerNorm |
| `W_g` | `[d, 3d]` | float | **[P]** §4.5, §4.7 |
| `c_g` | `[d]` | float | **[P]** explicit in §4.5 |
| `g` gate | `[B, L, d]` | float `∈ (0,1)` | **[P]** per-dimension, per-position |
| `z` = `inputs_embeds` | `[B, L, d]` | float | **[P]** word embeddings **only** |
| encoder hidden states | `[B, L, d]` | float | frozen `E_θ(inputs_embeds=z, attention_mask=…)` |
| Stage-1 pooled rep | `[B, d]` | float | **[R]** masked mean over non-special content — §11 |

**[I]** Invariant, from **[P]** §4.4: the three label sequences have equal
length, and that length is `|T(b(x))| = L`. Asserted before any training run.

**[R] Scope of `special_tokens_mask`.** It is not uniformly optional:

| Consumer | Requirement |
|---|---|
| Adapter fusion | **Not inherently needed** — if tone/letter `NA` metadata is already supplied, the channels are already zero at special tokens |
| **Stage-1 pooling** | **Required** — a special-token mask, or an exactly equivalent tokenizer-derived mask |
| B4B real-model integration | **Must construct and verify it** for real PhoBERT special tokens |

The two masks do different jobs and neither substitutes for the other:
`attention_mask` excludes **padding**; `special_tokens_mask` excludes **model
special tokens**. **Padding is never counted as content.**

**[R] The `NA` sentinel is not indexable.** `tone_ids` carries `-1` at
non-applicable positions and `tone_mask` is false there. **The sentinel must
never reach `nn.Embedding` in B4B** — the mask selects between a safe placeholder
lookup and exact zero.

---

## 4. Base embedding contract

**[P]** §4.5: `e_i = Emb_θ(b_i)` — the frozen encoder's **input word embedding
table**, looked up at the authoritative `T(b(x))` ids. **[D]** D-B3B1A-001 locks
`RAW_BASE`: no word segmenter between `b` and `T`.

**[P]** §4.5, stated as strongly as anything in the proposal:

> `z_i` substitutes for the *word* embedding only. Position embeddings and
> token-type embeddings are supplied by the encoder as usual — in practice, by
> passing `z` through the model's `inputs_embeds` interface. Adding position
> encodings inside the module would double-count them, and the failure is
> silent: the model trains, the loss decreases, and every number is wrong.
> **This is the single most likely implementation bug in the project.**

**[I]** Consequences for B4B:

| Component | Treatment |
|---|---|
| Word embeddings | replaced by `z`; the lookup for `e` uses the same frozen table |
| Positional embeddings | added by the encoder, **exactly once**, downstream of `inputs_embeds` |
| Token-type embeddings | same — encoder-supplied |
| Embedding LayerNorm + dropout | encoder-internal, applied to `z` after position/token-type addition |
| Special tokens | ordinary rows of `Emb_θ`; `e` is a real embedding, channels are `NA` |
| Padding | ordinary `Emb_θ[pad]`; excluded by `attention_mask` |

**[I]** The adapter's `LN` is **its own** parameter, distinct from the encoder's
embedding LayerNorm. `z` is *not* pre-normalised to the encoder's expectations —
the encoder's own embedding LayerNorm still runs.

**[OPEN — verification, not specification]** In HuggingFace, `inputs_embeds`
enters the embedding module which then adds position and token-type embeddings.
This is the documented contract, but it is **not empirically verified for the
pinned PhoBERT checkpoint in this repository**. B4B must verify it against the
real model (§9). A note specific to RoBERTa-family models, which PhoBERT is:
position ids are derived from `input_ids` via a padding-aware offset
(`create_position_ids_from_input_ids`), and when only `inputs_embeds` is passed
the model falls back to sequential position ids. **Whether that fallback matches
the ids the same batch would have received through `input_ids` is exactly the
kind of silent discrepancy §4.5 warns about, and B4B must check it directly.**

---

## 5. Tone channel contract

### 5.1 Cardinality — DISC-2, resolved

**[P]** §4.3 and §5.1: **7 slots = 5 marked tones + 2 policy slots**, and §8.2's
sketch says `n_tone=7  # 5 marked + 2 policy slots, identical for all three
policies`.

**[P]** §4.3's policy table:

| Policy | Slot A | Slot B |
|---|---|---|
| `OBSERVABLE` (UNMARK) | `UNMARKED` | *unused* |
| `FORCED-NGANG` | *ngang* | *unused* |
| `ORACLE` | *ngang* | `MISSING` |

**[D]** The repository's deployable taxonomy `TokenToneLabel` has **7 members**:
`UNMARKED, SAC, HUYEN, HOI, NGA, NANG, NA`.

**The original ambiguity.** Both are 7, and they are not the same 7. The
proposal's seven are `5 marked + slot A + slot B`; the repository's seven are
`5 marked + UNMARKED + NA`. §4.3 and §4.4 require non-Vietnamese subwords to
carry `N/A` in both channels, but **no row was allocated for it**. Three readings
were possible: `NA` takes the unused slot B (breaks `ORACLE`, which needs it for
`MISSING`); `NA` becomes an eighth row (breaks the §5.1 lock and the H4
equalization); or `NA` is not a row at all.

**[R] RESOLVED — D-B4A-002. `NA` is not a table row.**

The tone table is **exactly 7 trainable rows for every H4 policy**. `NA` is
structural non-applicability, not an observable tone state, and maps to the
**fixed zero vector**:

```
t_i = 0 ∈ R^d
```

at non-Vietnamese positions, special tokens, padding, and multi-candidate
ambiguous pieces. `UNMARKED` remains a genuine learned observable row and stays
distinct from `NA`.

This preserves the H4 equalization exactly: `NA` costs no rows, so it cannot give
one policy more capacity than another. Under `OBSERVABLE` the 7-row table is
still allocated in full even though slot B is never indexed — dropping the unused
row would itself break the equalization.

**[I] Batching contract:**

| Tensor | Shape | Note |
|---|---|---|
| `tone_ids` | `[B, L]` | valid policy rows are `0..6` |
| `tone_mask` | `[B, L]` | false where the tone channel is non-applicable |

`NA` may be carried as an out-of-table sentinel (`-1`) in the pure-data contract.
**A torch implementation must never feed that sentinel to `nn.Embedding`.** It
must use a safe placeholder lookup plus masking, or an equivalent safe mechanism,
and the resulting vector must be **exactly zero**. Feeding `-1` to an embedding
table wraps to the last row under some backends and raises under others; neither
is the contract.

### 5.2 What is fixed

**[P]** Tone is a **syllable** property, copied to every subword overlapping the
syllable's character span (§4.4 step 3).
**[D]** D-B3B1C-001: ownership by distinct-candidate count — 0 → `NA`; exactly 1
→ that candidate's observed tone even beside punctuation; **≥ 2 → `NA`**, all
contributors recorded, never resolved by length, position or averaging.
**[P]** Tone embeddings `W_τ` are **trainable** (§4.7 bills them).
**[P]** Dimensionality is `d` — `t_i` is concatenated with `e_i ∈ R^d`.
**[P] [D]** Lexical `NGANG` is **absent** from the deployable path: §4.3's
"Locked decision: no separate `MISSING` state at inference"; genuine *ngang* and
stripped tone both map to `UNMARKED`. `UNMARKED` ≠ `NA`.
**[I]** Special tokens and padding carry `NA` — they are not Vietnamese
syllables. Padding is additionally excluded by `attention_mask`.

**[I]** Multi-candidate ambiguity enters the tone input as `NA` and nothing else.
The recorded contributor list is **metadata for audit, not a model input**; no
tensor above consumes it. B3B-2 observed **0** multi-candidate pieces on the real
grid, so this path is currently unexercised in practice — which is a reason to
keep it explicit, not to drop it.

### 5.3 Label → integer mapping

**[I]** A deterministic mapping is required for reproducibility; the *choice* of
integers is arbitrary provided it is stable and recorded. Implemented in
`unmark/modeling/contracts.py` (torch-free), ordered `5 marked → policy slot A →
policy slot B`, matching the proposal's own description of the table:

```
SAC 0   HUYEN 1   HOI 2   NGA 3   NANG 4   slot A 5   slot B 6
```

Under `OBSERVABLE`, slot A is `UNMARKED` and slot B is allocated but never
indexed. **`NA` has no id in this table** — it is the out-of-table sentinel
`-1` plus `tone_mask`, resolving to the zero vector (§5.1).

---

## 6. Letter channel contract

### 6.1 What is fixed

**[P]** §4.3, §4.4 step 4, §5.1: labels are **per character**; a subword pools
its characters **in embedding space**, arithmetic mean, for v1. Learned attention
pooling is an ablation.
**[D]** D-B3B1C-001 / `LETTER_POOLING_RULE`: **`NONE` is included** in the mean;
**`NA` is excluded**; zero applicable contributors → token-level letter channel
`NA`.

**[I]** `NONE` ≠ `NA` is load-bearing and survives `STRIP_ALL`: B3B-2 showed the
applicable-contributor count constant at 165 across all six conditions, with the
22 letter-forming marks becoming `NONE` rather than `NA`.

### 6.2 Cardinality and the empty channel — DISC-5, resolved

**The original ambiguity.** §4.7 billed *"~10 × d, per character"*; §8.2 defaulted
`n_letter=10`; §4.3 wrote the set as `{NONE, breve, circumflex, horn, stroke,
circumflex+…}`. B1A had determined the applicable closed set is 5. Separately,
D-B3B1C-001 locked the *label-space* semantics of an empty channel but not the
*vector*.

**[R] RESOLVED — D-B4A-007. `n_λ = 5`.**

The trainable letter table contains exactly the five applicable character-level
states: `NONE`, `BREVE`, `CIRCUMFLEX`, `HORN`, `STROKE`. `NA` is **not** a
trainable row. The proposal's `~10` was a budget estimate and `n_letter=10` a
stale sketch: Vietnamese places **at most one** letter-forming mark on a
character (`ă â ê ô ơ ư`, and the `đ` stroke), so the anticipated `circumflex+…`
combination states do not arise. Proposal §4.7 and §8.2 are corrected in v1.4.
The orthographic taxonomy is unchanged.

**[R] RESOLVED — D-B4A-005. The empty channel is the exact zero vector.**

For a token with applicable contributor set `A_i`:

```
|A_i| > 0   →   l_i = (1 / |A_i|) · Σ_{j ∈ A_i} W_λ[label_ij]
|A_i| = 0   →   l_i = 0 ∈ R^d
```

`NONE` is included in the arithmetic mean; `NA` contributors are excluded.

**[I]** The implementation must explicitly prevent `0/0`. A torch implementation
**may** clamp the denominator for vectorisation, but only if the zero-contributor
output is then **explicitly forced to exact zero**. Clamping alone leaves
`sum(∅)/1 = 0` true by accident rather than by contract, and the accident stops
holding the moment the numerator stops being empty-safe. B3B-2 recorded 25 `NA`
positions per condition, so real batches exercise this path on every step.

### 6.3 Ragged contributors — DISC-4

**[P]** §8.2's sketch returns `letter_ids: list[int]` — one id per token — from
`propagate(...)`. That is **incompatible** with embedding-space pooling: pooling
needs the character labels *and* `W_λ`, which lives inside the module, so a
single pre-pooled id per token cannot be produced by `propagate` without moving
the embedding table out of the module.

§8.2 is an illustrative interface sketch; §4.4 step 4 and §5.1 are the locked
specification. **The sketch is superseded**, recorded so B4B does not implement
it.

**[I]** Contributors are genuinely ragged: one subword may cover 1..n characters.
Batching options, none of which changes the pooling semantics:

| Option | Shape | Notes |
|---|---|---|
| Padded dense | `[B, L, K_max]` + bool mask | simplest; `K_max` is a *batch* property, computed per batch, **not** a locked hyperparameter |
| Flat + segment reduce | `[N_contrib]` ids + `[N_contrib]` segment index | no padding waste; needs a segment-mean primitive |
| Precomputed sparse averaging matrix | `[B, L, C]` × `[B, C, d]` | one matmul; equivalent by linearity of the mean |

**[I]** All three are equivalent because the mean is linear and `W_λ` is a
lookup. The choice is an implementation/performance decision, **not** a
scientific one, and can be made in B4B without a researcher decision — provided
`NA` contributors are excluded from both numerator and denominator, and the
zero-applicable case is handled per §10.3.

**[I]** A masked mean must guard division by zero at zero-applicable positions.
Naively computing `sum / count` yields `NaN` there and silently poisons the
batch — the same class of silent failure as double position encoding.

---

## 6A. Zero channels do not make the adapter inactive

**[I]** A consequence worth stating, because the opposite reading is natural and
wrong. Tone `NA = 0` and letter empty `= 0` mean those *channels* contribute
nothing. They do **not** mean the adapter leaves the position alone. The fusion
still receives

```
[ e_i ; 0 ; 0 ]
```

a full-width input, and `W_f`, `c_f`, `W_g` and `c_g` may still change `z_i`.

Special tokens therefore carry: tone contribution **zero**, letter contribution
**zero**, source range **none** — and whatever `z_i` the locked architecture
produces from `[e_i ; 0 ; 0]`. **No special-token bypass is added**, because the
proposal does not specify one. Whether the adapter perturbs special-token
embeddings is a consequence of the locked architecture, not a new exception to
it.

---

## 7. Gate-zero recovery

**[P]** §4.5 is explicit that an earlier claim here was false, and states the
corrected one:

> Since `e_i = Emb_θ(b_i)` is computed from the **stripped** base stream,
> `g_i → 0` yields `E_θ(T(b(x)))`, not `E_θ(T(x))`. The gate recovers the
> **base-only pathway**, not the original model.

**Confirmed against the proposal.** The task brief's recorded constraint is
correct.

**[I]** "Base-only pathway" means precisely: the same authoritative `T(b(x))`
token ids; the same special tokens; the same attention mask; the same frozen
encoder `E_θ`; the encoder's own word-embedding lookup; and **no tone or letter
contribution**. It is *not* `E_θ(T(x))` — clean original text is a different
input pathway (§4.5, §6.6).

### 7.1 The exact numerical condition, and why it is unattainable

From `z_i = g_i ⊙ f_i + (1 − g_i) ⊙ e_i`, exact recovery `z_i = e_i` requires,
per position `i` and dimension `k`:

```
g_i[k] · ( f_i[k] − e_i[k] ) = 0
```

so for every `(i, k)`, **either** `g_i[k] = 0` **or** `f_i[k] = e_i[k]`.

Neither is attainable under the locked parameterization:

1. **`g_i[k] = 0` is unreachable.** `g = σ(·)` and §4.5 states the codomain as
   the *open* interval `(0,1)^d`. `σ(u) = 0` only as `u → −∞`. Exact zero is a
   limit, never a value.
2. **`f_i = e_i` is not generally attainable either.** `f = LN(·)`, whose output
   is normalised per position; `e_i = Emb_θ(b_i)` is an arbitrary pretrained
   vector with no such constraint. Making them equal for all `i` would require
   the LayerNorm affine parameters to invert the normalisation for *every* token
   simultaneously.

**The proposal does not claim exact recovery.** It writes `g_i → 0`, a limit, and
says the gate "recovers the base-only pathway" in that limit. There is therefore
**no internal inconsistency in the proposal** — but there is a real consequence
for B4B's test plan, which the task brief asks to be exact.

**[R] RESOLVED — D-B4A-004. A forced `g := 0` is a wiring test only.**

Under an explicit test override, `z == e` must hold **exactly**, up to ordinary
floating-point arithmetic. That override is:

* **not** a trainable parameterization;
* **not** an experiment condition;
* **not** a claim that `σ` attains zero;
* **not** evidence that the initialised module is identity.

**No casual production "gate zero mode" may be exposed**, because such a mode
could silently enter an experiment. B4B should test the fusion-combination
primitive or the internal path directly rather than adding a public flag.

Separately, **B4B must measure the real initialised gate at `g = 0.01` against
the base-only pathway and report the difference, which is expected to be
nonzero.** Reporting it as zero would mean something is wrong.

---

## 7A. Gate initialisation

**The original ambiguity.** The gate *transform* is locked (`σ`); its
initialisation appeared nowhere in the proposal, and `c_g = 0` (giving `g = 0.5`)
is the language-default that would have arrived by accident.

**[R] RESOLVED — D-B4A-003.**

```
W_g = 0
c_g = logit(0.01) = ln(0.01 / 0.99) ≈ −4.59511985013459
```

for every output dimension, so at initialisation

```
g_i = 0.01
```

for every token and every dimension, before any learning.

Three reasons: it starts training close to the pretrained base-only pathway; it
avoids `c_g = 0 → g = 0.5`, which would inject a **randomly initialised** fusion
branch at half weight on step zero; and it retains a usable sigmoid derivative
(`g(1−g) ≈ 0.0099`) so the gate projection can still learn. An initialisation
driving `g` to machine zero would drive the derivative there too.

`W_g = 0` makes the gate input-independent on step zero — the concatenated
channels drop out and every position starts at `σ(c_g)`.

**This is the main initialisation and must be explicit and config-logged.** It
does **not** claim exact base-only equality: `g = 0.01` is close to the base-only
pathway, not equal to it.

---

## 8. Frozen / trainable partition

**[P]** §4.5 *"All transformer layers above remain frozen"*; §5.1 *"Encoder: fully
frozen; no layer unfrozen without a logged decision"*; §8.3 Stage 1 *"Encoder
frozen"*.

| Frozen (`θ`) | Trainable (`φ`) |
|---|---|
| Word / input embedding table `Emb_θ` | Tone embedding table `W_τ` |
| Positional embeddings | Letter embedding table `W_λ` |
| Token-type embeddings | Fusion projection `W_f`, bias `c_f` |
| Encoder embedding LayerNorm | Adapter LayerNorm (gain, bias) |
| All transformer blocks | Gate projection `W_g`, bias `c_g` |
| All pretrained LayerNorms | |
| Pooler, if present | |

**[P]** §8.3: the Stage-2 task head is trained in a **separate stage** with the
module frozen. **It is not part of `φ`** and must not enter Stage-1 adapter
parameter counts — §4.7's budget lists no head.

---

## 9. What the B4B real-model probe must verify

**None of the following is empirically validated yet.** B3B-2 operated on
ordinary tokenizer positions and observed no model-added special tokens; that is
**not** a B3B blocker, it is deferred integration testing.

1. Exact PhoBERT special-token ids and their order.
2. Final sequence length `L` after special-token construction.
3. Base embedding lookup succeeds for special tokens.
4. Tone channel `NA` at every special token.
5. Letter channel `NA` at every special token.
6. No fabricated source range for special tokens.
7. Padding carries `NA` in both channels.
8. `attention_mask` passes through unchanged.
9. **Position embeddings applied exactly once** — and, for this RoBERTa-family
   model, that `inputs_embeds` produces the *same* position ids as `input_ids`
   would (§4). This check remains **open for real-model verification**; it is not
   a B4A blocker and must not be "fixed" in pure-data code. B4B must compare the
   authoritative `input_ids` path against the `inputs_embeds` path under **no
   padding**, **right padding**, **unequal sequence lengths within one batch**,
   and **real PhoBERT special tokens**, and must inspect the **actual position
   ids used** rather than inferring them. If the `inputs_embeds` fallback
   positions differ from the authoritative path, B4B must supply explicit
   `position_ids` derived from the `input_ids` behaviour.
10. `forward(inputs_embeds=…)` output shape equals `forward(input_ids=…)` output
    shape.
11. The forced `g := 0` wiring identity: `z == e` exactly (D-B4A-004).
12. The **real initialised** gate at `g = 0.01` against the base-only pathway,
    reporting the difference — **expected to be nonzero**.

---

## 10. Formerly open questions, now resolved

Audit 014 found six items the proposal left unspecified. **All are resolved**;
each is logged in [`decisions.md`](decisions.md) with the original ambiguity, the
decision, the reason, the mathematical consequence, and what it affects.

| # | Question | Resolution |
|---|---|---|
| 10.1 | Gate initialisation | **D-B4A-003** — `W_g = 0`, `c_g = logit(0.01)`, initial `g = 0.01` |
| 10.2 | Tone `NA` treatment | **D-B4A-002** — fixed zero vector, outside the 7-row table |
| 10.3 | Empty letter-channel vector | **D-B4A-005** — exact zero vector; `0/0` explicitly prevented |
| 10.4 | Stage-1 pooled representation | **D-B4A-006** — attention-masked mean over non-special content tokens |
| 10.5 | Gate-zero test form | **D-B4A-004** — forced `g := 0` as a wiring test only |
| 10.6 | `n_λ` final value | **D-B4A-007** — exactly 5 |

**One item remains open, and it is not a B4A item:**
[D-B3B0-002](decisions.md#d-b3b0-002), the backbone checkpoint. `d` therefore
stays symbolic throughout this document.

**One item is deferred to empirical verification rather than decision:** the
RoBERTa/PhoBERT position-id behaviour under `inputs_embeds` (§4, §9). It is not a
B4A blocker and must not be "fixed" in pure-data code; B4B verifies it against
the real model.

---

## 11. Stage-1 interface

**Interface only. Stage-1 training is not implemented in this task.**

**[P]** §4.6, §8.3:

```
L_align = D( h′(x̃ₚ), h(x) )        # corrupted through the adapter vs clean through the bare encoder
L_clean = D( h′(x),   h(x) )        # near-identity on uncorrupted input
L       = λ_a · L_align + λ_c · L_clean
```

| Element | Contract |
|---|---|
| Clean / reference branch | `h(x)` — the **same frozen encoder, without the module** |
| Corrupted / adapted branch | `h′(x̃ₚ)` — adapter + frozen encoder, corruption applied to the **tone channel**, base grid unchanged |
| Corruption sampling | **[P]** `p ~ U(0,1)` per example, continuous. An optional second rate governs letter dropout |
| Gradients | `φ` only: `W_τ`, `W_λ`, `W_f`, `c_f`, adapter LN, `W_g`, `c_g` |
| Frozen | all of `θ` (§8) |
| Objective input | pooled representation, cosine distance; pooling per **[R] D-B4A-006** below |
| Masks | `attention_mask` **and** `special_tokens_mask`, both required |

### 11.1 The pooled representation

**[R] RESOLVED — D-B4A-006.** Attention-masked **mean over non-special content
tokens**, from the final encoder hidden state. For `H ∈ R^(B×L×d)`:

```
m_i = attention_mask_i · (1 − special_tokens_mask_i)

h   = Σ_i m_i · H_i  /  Σ_i m_i
```

computed **independently for each branch**. Excluded: `<s>`, `</s>`, `<pad>`, and
every other tokenizer or model special token.

Reasons, in order of weight:

* **Mean pooling is defined across unequal branch lengths**, which is the
  situation §4.6 describes — and no per-token correspondence is assumed.
* **Excluding padding** prevents a bias that would otherwise vary with batch
  composition.
* **Excluding special tokens** prevents the alignment objective from receiving an
  artificially easy shared signal: those positions are near-invariant between
  branches, so a cosine objective including them is partly measuring agreement
  that was never in question.

**Zero content positions after masking is an error — fail loud.** No silent
fallback to `<s>`, to an unmasked mean, or to a zero vector: each would hand the
cosine objective a value that represents nothing.

Persisted into proposal §4.6 in v1.4, because this is a scientific decision
rather than an implementation detail.

**[I]** A structural asymmetry worth stating before it surprises someone: the
reference branch `h(x)` runs the encoder's **own tokenization of clean text**,
while `h′(·)` runs the base grid. Their sequence lengths differ. This is exactly
why §4.6 locks *pooled-only* alignment for v1 and defers per-token alignment —
"the clean and corrupted strings do not share a token grid, so per-token
alignment is itself an alignment problem". Any Stage-1 implementation that
assumes a shared `L` across the two branches is wrong.

---

## 12. Parameter count

Symbolic, backbone-parameterized.

| Component | Parameters |
|---|---|
| Tone embedding table | `n_τ · d` |
| Letter embedding table | `n_λ · d` |
| Fusion projection `W_f` | `3d²` |
| Fusion bias `c_f` | `d` |
| Gate projection `W_g` | `3d²` |
| Gate bias `c_g` | `d` |
| Adapter LayerNorm (gain + bias) | `2d` |

```
|φ| = 6d² + (4 + n_τ + n_λ) · d
```

With **`n_τ = 7`** (D-B4A-002 — `NA` is outside the table) and **`n_λ = 5`**
(D-B4A-007):

```
|φ| = 6d² + 16d
```

**Arithmetic sanity check only.** At `d = 768` the formula gives
**3,551,232**. The proposal's §4.7 budget totals ≈3.6M and is reproduced to
within its own rounding — and **only** under DISC-1's reading, since it can be
matched at all only if the gate is a `3d → d` projection (fusion 1.77M, gate
1.77M, tone ≈5K, letter ≈4K, LN ≈2K).

`d = 768` is used **here only** to check arithmetic. **It is not adopted and not
locked**: D-B3B0-002 is OPEN, and `hidden_size` has no default in the contract.

**[R]** H4 consistency: all three tone policies share one `n_τ = 7` table, so
`|φ|` is **identical** across `OBSERVABLE`, `FORCED-NGANG` and `ORACLE` — the
entire point of the 7-slot equalization. D-B4A-002 keeps `NA` **outside** the
table precisely so this stays true; a learned `NA` row would have given one
policy different capacity and broken the fairness argument. The contract rejects
that configuration by name rather than merely documenting it.

**[P]** The Stage-2 head is excluded (§8).
