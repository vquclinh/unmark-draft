# Audit 028 — Stage-1 scientific configuration review

| | |
|---|---|
| **Audit id** | 028 |
| **Created (UTC)** | 2026-08-22 |
| **Baseline HEAD** | `3bb0edbf5061154db3667dfc0cbf0432f038aa31` |
| **Scope** | Config-lock **review and proposal** before the Stage-1 runner exists. |
| **Predecessor** | [027](027-preg1-base-only-own-lr-sensitivity.md) — pre-G1 secondary sensitivity |
| **Type** | Review + proposal. **No runner, no optimizer step, no model load, no download, no training** |
| **NOT** | This is **not** the major PRE-TRAIN audit. That audit inspects the runner, which does not exist yet |
| **Revision 2** | **2026-08-22** — final config-lock clarification, **in place**. UVW three-shard source policy and load order; split-before-chunk with partition inheritance; the 11-run sequence stated unambiguously with the final three runs as the main adapters; the precommitted 20 k → 40 k budget rule; engineering defaults locked as **a-priori**; exact seed integers derived and recorded. |
| **Revision 1** | **2026-08-22** — researcher review, revised **in place**. Four scientific corrections (§D.1 contamination contradiction, §G.3 withdrawn "floor" claim, §G.1 withdrawn collapse claim, §G.4 undefined tie-break), one added requirement (§F.7 stream independence), the corpus decision (§D.2) and the chunking contract (§D.3). Approved values moved to **LOCKED**; engineering defaults **labelled as such**. |

---

## A. VERDICT

**REVIEW COMPLETE (Revision 1) — CONFIGURATION LARGELY LOCKED; ONE MECHANISM
STILL TO IMPLEMENT**

**The load-bearing finding, stated first:** under the **current default** Stage-1
corruption policy, the headline evaluation condition **STRIP-ALL has exactly
zero training support**. The letter-diacritic channel of the corrupted branch is
**bit-identical** to the clean branch in every prepared example. This is proved
from the data path in §F, not inferred from names.

The repair is now **decided**: a per-example scope mixture with
**`pi_strip = 0.25`**, plus the stream-independence property of §F.7
(**D-S1B-003**). **It is not yet implemented.** Stage-1 training must not begin
until `scope_for` exists with its ML-free tests — until then STRIP-ALL support is
still zero.

**Corrections made in Revision 1.** Four claims in the first draft were wrong or
unverifiable and have been withdrawn *in place*, each marked at the point of the
original error:

1. **§D.1** — requiring "zero overlap with UIT-VSFC validation **or test**" while
   keeping TEST sealed is **self-contradictory**; replaced by an explicit
   contamination contract with a **report-only** post-unsealing audit.
2. **§G.3** — "`L_clean` cannot reach 0" and "irreducible floor" are **not
   established**. Differing token grids do not bound the distance between
   *pooled* representations. Now: **initial clean-path distance**, plateau
   **observed**, not assumed.
3. **§G.1** — "`lambda_clean = 0` admits a collapsed representation" is
   **withdrawn**: `h(x)` varies per example, so a constant `h'` is not
   automatically a minimizer. The defensible claim is that `lambda_clean`
   *explicitly regularizes* clean-path preservation.
4. **§G.4** — "lower sample SD across seeds" was an **undefined** tie-break for a
   one-seed sweep; replaced by an explicit two-phase protocol.

Six decisions are now recorded in `docs/spec/decisions.md`. Everything else is
either an **EMPIRICALLY SELECTED LATER** pilot value or an explicitly labelled
**PROPOSED ENGINEERING DEFAULT** — never presented as evidence. Nothing was
written into production configuration, and no runner exists.

---

## B. WHAT WAS READ

`unmark-proposal.md` (§4.3, §4.5, §4.6, §5, §5.1, §6.3, §13), `docs/spec/decisions.md`
(all OPEN entries), `unmark/stage1/{contracts,data,objective}.py`,
`unmark/corruption/{corrupt,conditions,eligibility,deterministic}.py`,
`unmark/modeling/{config,contracts,adapter}.py`, `configs/corruption/default.yaml`,
Audits 019 (real-PhoBERT dry run), 022 (length profile), 024–027 (pre-G1).

Executed locally, ML-free and deterministic: the corruption operator and the
Stage-1 preparation path on sample text. **No model, no dataset, no network, no
optimizer.**

---

## C. PRE-G1 IS CLOSED

| | Macro-F1 | Accuracy |
|---|---|---|
| Vanilla (shared LR 0.01) | 0.745 601 988 142 145 4 | 0.901 326 595 072 646 9 |
| Base-only (shared LR 0.01) | 0.663 044 566 342 825 5 | 0.822 867 972 204 674 8 |
| **Burden (Vanilla − Base-only)** | **0.082 557 421 799 319 88** | **0.078 458 622 867 972 2** |

The secondary own-LR sensitivity executed at `3bb0edb`. Base-only **independently
selected LR = 0.01** — the same value the primary shared protocol had frozen,
from the same precommitted grid under the same tuning protocol. The secondary
burden is therefore **numerically identical** to the primary burden.

**What this does and does not mean.** It removes one specific alternative
explanation: the gap is not an artefact of Base-only having been denied its own
tuning. It is **not** a significance result, **not** a bound, and **not** a
claim that the gap would survive any other protocol change. No p-value, no
threshold and no test was computed at any point.

Pre-G1 is closed. It is not rerun, its grid is not widened, and its numbers are
inputs from here on. Recorded as **D-PREG1-015**.

---

## D. STAGE-1 DATA POLICY

### D.0 UIT-VSFC is excluded from Stage-1 selection

The pre-G1 diagnostic measured **downstream burden**. It did not, and cannot,
license using UIT-VSFC scores to choose Stage-1 hyperparameters.

Using them would break the study in two independent ways:

1. **Selection leakage into the headline number.** Official validation is the
   set the Stage-2 result is reported from. Choosing `lambda`, LR, adapter
   capacity or corruption policy by that score makes the reported number a
   report of the selection, exactly the failure D-PREG1-009 exists to prevent
   one level down.
2. **It inverts the scientific claim.** UNMARK's claim is that Stage-1
   self-supervised alignment produces diacritic robustness. Tuning Stage-1 on a
   labelled downstream task would make Stage-1 a supervised search over that
   task, and the claim would become circular.

**Locked:** Stage-1 model and configuration selection uses **Stage-1 held-out
unlabeled signals only**. Official UIT-VSFC TEST remains sealed and structurally
unreachable. Recorded as **D-S1B-001**.

### D.1 Contamination contract — corrected

An earlier draft of this review required the Stage-1 corpus to have **zero
overlap with UIT-VSFC validation *or test***, while simultaneously requiring
official TEST to stay sealed. **Those two requirements contradict each other:**
verifying non-overlap with TEST means reading TEST. The requirement is replaced
by the following contract.

| # | Rule |
|---|---|
| 1 | Official UIT-VSFC **TEST remains SEALED**. It is not opened for contamination screening, or for any other purpose, before final evaluation |
| 2 | Stage-1 corpus construction may screen overlap **only** against UIT-VSFC material the pre-G1 protocol has **already legitimately opened** — the derived TRAIN pool and official VALIDATION |
| 3 | Screening at that stage is an **exact/canonical duplicate** check: equality of `canon(x)` (and of its sha256), not a fuzzy or semantic similarity search |
| 4 | Any **fuzzy or semantic** near-duplicate analysis, if ever performed, is reported **separately** and never conflated with the exact check. It has a threshold, and thresholds are choices |
| 5 | **No claim of "zero TEST overlap" may be made before TEST is opened.** The honest statement is: *no exact overlap was found against the material legitimately available* |
| 6 | **After** the complete UNMARK configuration and model are frozen, and TEST is eventually unsealed for final evaluation, a contamination audit is run against TEST |
| 7 | That post-unsealing audit is **REPORT-ONLY**. It must not trigger retroactive corpus removal, retraining, model re-selection, configuration change, or any change to the reported result |

**Why rule 7 rather than "remove and retrain".** Acting on a contamination
finding *after* seeing TEST would make TEST a selection signal — the exact
failure the seal exists to prevent, arriving through the back door. The
scientifically honest response to post-hoc contamination is to **report it as a
measured limitation of the result**, not to launder it by retraining until the
number improves.

Recorded as part of **D-S1B-002**.

### D.2 Stage-1 corpus — RESEARCHER DECISION

**Main Stage-1 corpus: `undertheseanlp/UVW-2026`** (Vietnamese Wikipedia).
Researcher decision, 2026-08-22. **Not downloaded in this task.**

#### Source shards — all three, and they are NOT a split

**Use all three root Hugging Face parquet source shards:**

```
train.parquet   validation.parquet   test.parquet
```

**The upstream UVW `train` / `validation` / `test` labels carry no scientific
split meaning for UNMARK.** They are three source shards of **one unlabeled
Wikipedia corpus**. In particular the shard named `test.parquet` has nothing to
do with UIT-VSFC's sealed TEST, and treating the upstream labels as a split would
import a partition that was never designed for this study.

#### Pipeline order — load, screen, split, *then* chunk

At the pinned dataset revision, in exactly this order:

| # | Step |
|---|---|
| 1 | **Load and concatenate** the three shards in the fixed documented order `train.parquet → validation.parquet → test.parquet` |
| 2 | **Preserve article/document ids** through the concatenation |
| 3 | **Contamination screening** (§D.1): exact/canonical duplicates against **only** legitimately opened UIT-VSFC material |
| 4 | **Construct UNMARK's own document-level train/dev partition** |
| 5 | **Only after document partitioning**, perform deterministic chunking (§D.3) |
| 6 | **Every chunk inherits its parent document's partition** |

**The load order is part of the pin.** Concatenation order determines document
enumeration, and `sample_id` keys the corruption draw — so a different order is a
different corruption stream, even at an identical revision.

**Required structural property:** it must be **structurally impossible** for
chunks of one Wikipedia article to appear in both Stage-1 train and Stage-1 dev.
Steps 4→5→6 give that by construction: the partition is decided at document
level *before* any chunk exists, and chunks cannot be assigned independently of
their parent. Chunking before splitting would allow two chunks of one article to
land on opposite sides — near-duplicate leakage into the very signal used to
select `r` and the learning rate.

#### The pin — still OPEN, still blocking

**Before implementation or execution**, the exact Hugging Face **dataset
revision** and the sha256 of **all three parquet source files** must be pinned,
in the same form every other external artifact in this project is pinned
(`RESTORE` checkpoint, syllable inventory, backbone revision). An unpinned
corpus makes Stage-1 unreproducible, exactly as an unpinned backbone would.

Public `main` was **observed at review time** as
`a0a79294e4568137e25828bb3f2a4cde8546e1fb`. This is recorded as an observation
only. **`main` moves and must not be trusted silently**: execution must name an
explicit full revision and verify the three file hashes at load. This value has
**not** been verified by this audit — nothing was downloaded.

**Rationale, recorded:**

* clean Vietnamese text is appropriate for learning **orthographic equivalence**
  — the thing Stage-1 actually optimises;
* **document/article identity is available**, which the split (item 3) and the
  corruption key (`sample_id`, §5.3) both require;
* **reproducibility is simpler** than a massive web mixture;
* main Stage-1 **does not need labelled downstream-domain matching** — it is
  self-supervised, and domain matching to a labelled task is precisely the
  coupling §D.0 forbids;
* broader corpora such as **CulturaX-vi may be explored later as a corpus/domain
  ablation**, and **must not retroactively replace the main result**.

**Not claimed.** This corpus is **not** guaranteed free of sealed UIT-VSFC TEST
content. Wikipedia and student-feedback text are different domains, which makes
overlap unlikely, but "unlikely" is not "verified" and TEST is not opened to
check (§D.1 rules 1, 5). The post-unsealing audit (§D.1 rule 6) is where that
question is answered, **report-only**.

Only corpus contents allowed by the final pinned corpus protocol may be used.

### D.3 Chunking contract — replaces silent SKIP

An earlier draft proposed `on_overflow: SKIP` as the training policy. **That is
withdrawn**: silently dropping long documents biases the corpus toward short
examples, and it does so invisibly.

**Deterministic pre-chunking happens before Stage-1 preparation.** The contract:

| # | Requirement |
|---|---|
| 1 | **Preserve text order.** Chunks are contiguous and in document order; no shuffling, no reordering, no dropping of interior text |
| 2 | **No extra normalization.** Chunking must not restore, repair or normalize away orthographic information beyond the canonical pipeline already specified (`canon`, §5.3 / D-B2-004). It is a segmentation step, not a cleaning step |
| 3 | **Stable chunk ids**, derived as `f"{document_id}#{chunk_index}"`. `sample_id` keys the corruption draw, so an unstable id silently changes the corruption stream |
| 4 | **Fit `max_length = 256`** on the relevant clean **and** base tokenizer paths — both, because the reference and base branches have *different* lengths and separate padding domains |
| 5 | **Preserve the clean/base channel alignment contracts** — chunk boundaries must not split a syllable span, or the B3 projection desynchronises |
| 6 | **Chunking runs only after the document-level partition exists** (§D.2 step 5). It never sees, and never decides, which side a document is on |
| 7 | **Every chunk inherits its parent document's partition**, never an independently assigned one. Chunks of one article can therefore never straddle Stage-1 train and dev |

**Runtime overflow policy: `FAIL`.** After correct chunking nothing can overflow,
so `FAIL` is a **guard**, not a data policy. Any overflow at runtime means the
chunking or preparation contract is wrong, and the run must **stop** rather than
quietly bias the corpus. This is the same fail-closed posture as
`BaseInvarianceViolation`.

`max_length` stays **256** unless inspection finds a concrete contract reason to
change it — and a contract reason, not a convenience one.

### D.4 Seed roles — derived, domain-separated, and recorded before use

Root tag **`UNMARK-STAGE1-v1`**. Role seeds are derived with the repository's
established convention, `derive_seeds(tag, count)` in
`unmark/evaluation/profiling.py` — `sha256(tag)` read as consecutive 2-byte
big-endian integers. **No integer here was chosen; every one is recomputable from
its tag string alone**, which is the property that makes the seeds falsifiable.

Verification that this is the same convention already in use:
`derive_seeds("UNMARK-PREG1-TUNE-v1", 3)` reproduces the committed
`TUNING_SEEDS = (5509, 19422, 11800)` exactly.

**The exact integers, recorded here BEFORE the first scientific run:**

| Role | Namespace tag | Seed |
|---|---|---|
| Pilot / Phase-1 selection | `UNMARK-STAGE1-v1\|selection` | **21230** |
| Final main Stage-1, run 0 | `UNMARK-STAGE1-v1\|train\|0` | **36930** |
| Final main Stage-1, run 1 | `UNMARK-STAGE1-v1\|train\|1` | **7309** |
| Final main Stage-1, run 2 | `UNMARK-STAGE1-v1\|train\|2` | **5993** |
| Corruption stream | `UNMARK-STAGE1-v1\|corruption` | **35422** |

All five integers are **distinct** (verified). Seeds are 16-bit, `[0, 65535]`,
as throughout this project.

**Why separate namespaces.** Training, selection and corruption must not
accidentally share one integer: a training seed equal to the corruption seed
would couple parameter initialisation to the corruption stream, and a selection
seed equal to a training seed would make the Phase-1 run indistinguishable from a
final run. Domain separation makes that structurally impossible rather than
merely unlikely.

**This is separate from, and additional to, the `rate_for` / `scope_for` stream
separation of §F.7.** That separates two draws *within* the corruption seed; this
separates the seeds themselves.

---

## E. BACKBONE — D-B3B0-002 CLOSED

**Recommendation: LOCK** `vinai/phobert-base`, revision
`01daacda68afe13d83023d16ec647239e344a1e6`.

The revision is not a fresh choice; it is the one every downstream artifact was
already validated against:

| Evidence | Audit | Established on this exact revision |
|---|---|---|
| Input/tokenizer contract | 006, 010 | tokenizer revision verified |
| Alignment + channel projection | 013 | manual alignment validated |
| Adapter on the real model | 016, 017 | position-id semantics; `VERIFIED_POSITION_PROFILES` |
| Stage-1 three-branch graph | 019 | 31/31 checks, adapter `3,551,232` params, encoder `0` |
| Pre-G1 burden diagnostic | 024–027 | 30 real head runs, both pathways |

D-B3B0-002 was **OPEN — EMPIRICAL PROBE REQUIRED**. The probes are done and all
passed on this revision. Leaving it open now means the whole validated stack
rests on a revision the spec still calls provisional — "neither locked nor
tracked as open, which is the worst of the two states", in the decision's own
words.

**A second backbone is not adopted.** Proposal §6.1 mentions ViSoBERT, but a
second backbone is a **generalisation ablation** that must run *after* the main
result on a locked backbone, with its own position-id evidence
(`VERIFIED_POSITION_PROFILES` has exactly one entry and fails closed). Adding
one now because 90 GB of VRAM is available would be choosing an experiment by
resource availability. Recorded as **D-B3B0-007**, closing D-B3B0-002.

---

## F. LOAD-BEARING — STAGE-1 CORRUPTION SUPPORT REVIEW

### F.1 What the specification asks for

Proposal **§6.3**, verbatim: *"`STRIP-ALL` is the condition that matches how
people actually type and **should be reported as the headline number**."*

Proposal **§4.6**, verbatim: *"For each example, sample a corruption rate
`p ~ U(0,1)` and set **the tone channel** to `UNMARKED` for a random `p`-fraction
of syllables. […] **An optional second rate governs letter-diacritic dropout.**"*

So the proposal's *primary* Stage-1 corruption sentence is **tone-only**, and
letter-diacritic dropout is an explicitly anticipated **optional second rate**
that it never gives a value for. This is not a suggestion from discussion; it is
in the specification, and it is registered in code as
`OPEN_STAGE1_VALUES["letter_dropout_rate"]`.

### F.2 The actual mechanism, traced

`unmark/stage1/data.py::prepare_example` builds **one** condition per example:

```python
rate = corruption_policy.rate_for(example.sample_id, visit)
condition = CorruptionCondition(
    name=f"stage1-p{rate:.6f}",
    scope=CorruptionScope[corruption_policy.scope],   # <-- ONE scope, all examples
    probability=rate, ...)
```

`unmark/stage1/contracts.py::CorruptionRatePolicy`:

```python
seed: int
scope: str = "TONE"        # <-- a DEFAULT, and the only letter-diacritic switch
```

`unmark/corruption/corrupt.py::_apply` is where information is actually removed:

```python
if scope in (CorruptionScope.TONE, CorruptionScope.TONE_AND_LETTER):
    kept = [m for m in marks if m not in _TONE_MARK_SET]     # tone marks
    ...
if scope is CorruptionScope.TONE_AND_LETTER:                 # <-- ONLY here
    kept = [m for m in marks if m not in _LETTER_MARK_SET]   # ă â ê ô ơ ư
    ...
    if base in D_STROKE:                                     # đ -> d
        base = D_STROKE[base]
```

That `if` is the **only** place `_LETTER_MARK_SET` and `D_STROKE` are consulted.
Removal is per **selected syllable**, and a selected syllable loses tone and
letter diacritics **together**.

### F.3 Answers

**A. Can `tau` (tone) become unavailable?**
**YES.** Under either scope, a selected syllable loses its tone mark. Precisely:
the tone channel is not marked *N/A* — it takes the observed `UNMARKED` state,
one of the 7 rows of `ToneChannelContract`. That is the correct deployment
analogue: a person typing without an IME produces a genuinely unmarked syllable,
not an unknown one. (`NA ≠ NONE` is a real distinction here — N/A is reserved for
non-Vietnamese spans and is `exclude_na_from_pool=True`.)

**B. Can `lambda` (letter diacritic) become unavailable?**
**NO — not under the default.** `scope` defaults to `"TONE"`, and under `TONE`
the letter branch of `_apply` is never entered. It becomes possible **only** if a
caller explicitly passes `scope="TONE_AND_LETTER"`, which nothing currently does.

**C. Can `tau` and `lambda` be unavailable simultaneously?**
**Only under `TONE_AND_LETTER`, and then only jointly.** Because both removals
are gated on the *same* per-syllable Bernoulli draw, a selected syllable loses
both and an unselected syllable keeps both. The state *"tone absent, letter
present"* — which is exactly conditions **P25/P50/P75/P100** — then has
probability **zero**.

**D. Does the training distribution have support for STRIP-ALL?**

**NO.** And neither single scope can cover the headline evaluation set:

| Stage-1 `scope` | FULL | P25 / P50 / P75 | P100 | **STRIP-ALL** |
|---|---|---|---|---|
| **`TONE` (current default)** | yes | yes | yes | **NO — probability 0** |
| `TONE_AND_LETTER` | yes | **NO — probability 0** | **NO — probability 0** | yes (≈ `1/(N+1)` per example) |

### F.4 Proof from execution

Real corruption operator, real syllable inventory, ML-free:

```
clean                     : Tôi đã đọc quyển sách này rồi và thấy rất hay
scope=TONE,            p=1: Tôi đa đoc quyên sach nay rôi va thây rât hay   <- ô ơ ê â đ SURVIVE
scope=TONE_AND_LETTER, p=1: Toi da doc quyen sach nay roi va thay rat hay   <- STRIP-ALL
```

Over 8 examples drawn from the **default** policy (`scope="TONE"`, `p` from the
real keyed digest), letter diacritics were present in **8 / 8**.

At the channel-tensor level, through `prepare_example` (the actual Stage-1 data
path), over 18 prepared examples × 2 scopes:

| Stage-1 `scope` | tone channel differs (clean vs corrupt) | **letter channel differs** |
|---|---|---|
| **`TONE` (current default)** | 14 / 18 | **0 / 18** |
| `TONE_AND_LETTER` | 14 / 18 | 13 / 18 |

**Under the current default the corrupted branch's letter channel is
bit-identical to the clean branch's, in every example.** The letter embedding
table is therefore never trained in its degraded configuration, and `L_align`
never once has to survive missing letter-diacritic information — the precise
condition the project promises to report as its headline number.

### F.5 Why this was not caught earlier

Nothing is wrong with the corruption engine: `TONE_AND_LETTER` is implemented,
audited (Audits 003/004) and correct. The gap is that Stage-1 exposes **one**
scope for the whole run, defaulted to the proposal's primary sentence, while the
evaluation grid spans **both** regimes. The default is exactly the class of
value `unmark/stage1/contracts.py` warns about in its own module docstring —
*"an API default is a scientific decision if it can reach an experiment"* — and
`scope` is the one field in that file that carries a default anyway.

### F.6 Smallest coherent repair — PROPOSED

**Draw the scope per example, from the same keyed digest, instead of fixing it
for the run.** One new scalar, `pi_strip`:

```
with probability pi_strip      : scope = TONE_AND_LETTER,  p ~ U(0,1)
with probability 1 - pi_strip  : scope = TONE,             p ~ U(0,1)
```

Why this is the minimal option:

* **No change to the corruption engine.** `corrupt()`, `_apply()` and both
  scopes are already implemented and audited. Only `CorruptionRatePolicy` gains
  a `scope_for(sample_id, visit)` alongside the existing `rate_for`, using the
  same `blake2b` construction — so determinism, reproducibility and the "no
  global RNG" property are inherited, not re-established.
* **The locked distribution is untouched.** `p ~ U(0,1)` per example remains
  exactly as §4.6 and §5.1 lock it. `pi_strip` is the *"optional second rate"*
  §4.6 already anticipates, and nothing else.
* **It restores support for the whole evaluation grid**: the `TONE` component
  covers FULL→P100, the `TONE_AND_LETTER` component covers FULL→STRIP-ALL.
* **It performs no restoration.** Corruption still only ever *removes*
  information that is represented outside the base channel, so the invariant
  `strip_to_base(canon(x)) == strip_to_base(corrupt(x))` is untouched and
  `BaseInvarianceViolation` remains the guard it was.

**Rejected alternative — an independent per-syllable letter-dropout rate `q`.**
This is the reading the phrase "second rate" most literally suggests, but it
requires modifying `_apply()` to take two independent per-syllable decisions,
which changes an audited deterministic engine and creates the state *"letter
diacritic removed, tone kept"* — a state that is **not** in the evaluation grid
and does not correspond to any real typing behaviour (no IME removes `ơ` while
keeping `ớ`'s tone). It is strictly more machinery for strictly less relevant
support. **Not recommended.**

### F.7 The two draws must be independent — REQUIRED PROPERTY

`rate_for` and `scope_for` must use deterministic but **domain-separated**
streams. Both may derive from `(seed, sample_id, visit)`, but under **distinct
namespace tags**:

```
rate_for (sample_id, visit) : blake2b("stage1-rate"  | schema | seed | sample_id | visit)
scope_for(sample_id, visit) : blake2b("stage1-scope" | schema | seed | sample_id | visit)
```

**Forbidden:** reusing one scalar draw for both, deriving one from the other, or
making `scope` conditional on the sampled `p` (for example "strip letters only
when `p > 0.9`").

**Required property:**

```
P(p | scope = TONE)            = Uniform(0, 1)
P(p | scope = TONE_AND_LETTER) = Uniform(0, 1)
```

up to the deterministic finite sample.

**Why this matters scientifically.** If `scope` were conditioned on `p`, the
`TONE_AND_LETTER` component would occupy only part of the rate range, and the
letter-degraded regime would be confounded with corruption severity. The model
would then have seen "letters missing" only ever alongside "most tones missing",
and any measured STRIP-ALL behaviour could not be attributed to letter
information alone. Independence keeps the two axes separable — and keeps the
`p ~ U(0,1)` distribution locked by §4.6 and §5.1 **exactly** uniform within each
scope, rather than uniform only in the marginal.

A test asserting the two marginals and their independence is cheap, ML-free, and
must ship with the mechanism.

**The corruption engine itself remains unchanged.**

### F.8 Value

**`pi_strip = 0.25` — LOCKED** as an a-priori researcher decision (2026-08-22).

It is fixed **before** any Stage-1 result exists and is **not** tuned — not on
UIT-VSFC, not on any downstream score, and not on the Stage-1 held-out signal
either. Recorded as **D-S1B-003**.

---

## G. LAMBDA REVIEW

### G.1 The tradeoff

```
L = lambda_align * D(h'(corrupt), h(clean))  +  lambda_clean * D(h'(clean), h(clean))
```

`L_align` asks the adapted **corrupted** representation to match the clean
reference — this is the robustness the project is claiming. `L_clean` asks the
adapted **clean** representation to match it too.

**What `lambda_clean` does, stated defensibly.** `lambda_clean` **explicitly
regularizes preservation of the adapted clean pathway**. Without it, clean-path
preservation is simply **not directly optimized** — `h'(x)` is then constrained
only indirectly, through whatever it shares with `h'(x_p)`.

An earlier draft of this review claimed that `lambda_clean = 0` would leave
nothing to prevent a *collapsed*, constant representation. **That claim is
withdrawn**: `h(x)` varies across examples, so a constant `h'` is **not**
automatically a minimizer of `D(h'(x_p), h(x))` — a constant vector cannot be
simultaneously cosine-aligned with a set of differing targets. Collapse is a
hypothesis worth monitoring (§H item 23), not a proved consequence of
`lambda_clean = 0`.

With `lambda_align = 0` there is no Stage-1 objective at all.

**The ratio `r = lambda_clean / lambda_align` is the meaningful axis.**

### G.2 Absolute scale is *not* mathematically irrelevant

Under plain SGD, scaling `L` by `c` is equivalent to scaling the LR by `c`. Under
the **AdamW** actually proposed, that equivalence fails for three concrete
reasons:

1. **`eps` breaks scale invariance.** The update is `m̂ / (sqrt(v̂) + eps)`.
   Adam is scale-invariant only where `sqrt(v̂) >> eps`; scaling `L` down moves
   small-gradient parameters toward the `eps` floor and changes their effective
   step. With `eps = 1e-8` and a bounded cosine loss this is a live regime for
   the gate bias and the LayerNorm parameters.
2. **Decoupled weight decay does not scale with the loss.** AdamW applies
   `-lr * wd * theta` independently of the gradient. Doubling `L` therefore
   halves the *relative* strength of weight decay — a real change to the
   objective being optimised, not a reparameterisation.
3. **Gradient clipping, if enabled, is defined on an absolute norm.**

**Consequence:** the scale must be *held fixed* while `r` is varied, or the
comparison confounds the tradeoff with the optimiser. **Proposed
parameterisation**, one free parameter:

```
lambda_align = 2 / (1 + r)          lambda_clean = 2r / (1 + r)
=> lambda_align + lambda_clean = 2 for every r; r = 1 gives (1.0, 1.0)
```

`r = 1` reproduces exactly the `(1.0, 1.0)` scale already exercised end-to-end on
real PhoBERT in Audit 019, where `loss_align = 0.5321` and `loss_clean = 0.5478`
at initialisation — the two terms are already the same order of magnitude, so
this normalisation is not distorting anything.

### G.3 The initial clean-path gap — measured, not assumed

`h'(x)` runs the *base grid* `T(b(x))` through the adapter, while `h(x)` runs the
reference tokenization of `canon(x)` through the bare encoder. The two branches
**do not share a token grid**, which is exactly why §4.6 aligns pooled
representations rather than tokens. Audit 019 measured
`d_clean = 0.5478` at initialisation on real PhoBERT.

**Terminology correction.** An earlier draft of this review called that value an
**irreducible floor** and asserted that `L_clean` "cannot reach 0". **Both
statements are withdrawn — neither is established.** Differing token grids do
**not** imply that the two *pooled* representations cannot become equal or
arbitrarily close: pooling maps variable-length token sequences into one shared
`R^d`, and nothing proved here forbids the adapted branch from reaching the
reference point. No positive lower bound on `d_clean` has been derived, and none
is claimed.

The correct terms, used from here on:

* **initial clean-path distance** (equivalently *initial clean-path gap*) — the
  measured value at initialisation, `0.5478` in Audit 019;
* its **observed plateau** — whatever `d_clean` settles at during training.

**Required before any `r` is chosen:** measure the initial clean-path distance on
the held-out split, and track `d_clean` throughout training so the plateau is
**observed empirically rather than assumed**. The reason is unchanged even though
the claim is weaker: if `d_clean` plateaus well above zero, that plateau must be
read as an empirical property of this architecture and pooling, **not**
automatically as a failure of `lambda_clean` — and equally, it must not be
assumed unreachable before it has been measured.

### G.4 Selection procedure — unlabeled, small, precommitted

**Data:** the Stage-1 held-out dev split (§H item 3), disjoint at **document**
level. No labels. **No UIT-VSFC** (§D).

**Measure**, for each candidate config, on the held-out split, at a **fixed grid
of corruption conditions** rather than at random `p` — otherwise configs are
compared on different corruptions:

| Signal | Definition |
|---|---|
| `d_c` | mean `D(h'(x_c), h(x))` for `c ∈ {FULL, P50, P100, STRIP_ALL}` |
| `d_clean` | mean `D(h'(x), h(x))` — the preservation term |
| `gap` | `d_STRIP_ALL − d_clean` — the quantity Stage-1 exists to shrink |

**Primary criterion (precommit before running):** minimise the **worst case**

```
score(config) = max over c in {FULL, P50, P100, STRIP_ALL} of d_c
```

Worst-case rather than the mean, because a mean lets a config win by being
excellent at `FULL` and poor at `STRIP-ALL` — the reverse of the project's
headline claim.

**Two phases, and the tie-break is well-defined in each.**

An earlier draft listed "lower sample SD across seeds" as a tie-break for a
**one-seed** candidate sweep. **That is undefined** — a single run has no sample
SD — and it is corrected here.

*Phase 1 — select `r`.* One **precommitted selection seed**, 5 candidates:

```
r in {0.25, 0.5, 1, 2, 4}    (lambda_align, lambda_clean derived as in G.2)
```

| | Rule |
|---|---|
| Primary | lowest held-out worst-case condition distance, `score(config)` |
| Tie-break 1 | lower `d_clean` |
| Tie-break 2 | smaller `r` |

No seed-variance term appears, because at one seed there is none to compute.

*Phase 2 — the final main Stage-1 runs.* Re-run **only the selected `r`**, at the
selected LR, on the **3 precommitted Stage-1 seeds**. Report the mean and the
**sample SD (n−1)** **descriptively**.

**These three runs are simultaneously (a) the Phase-2 descriptive
characterisation of the selected configuration and (b) the FINAL MAIN Stage-1
trained adapters for the study. There is no further Stage-1 training round after
them.**

**Phase 2 may not reopen LR or `r` selection.** It characterises the selected
configuration; it does not re-rank either grid, and a disappointing SD is not
grounds to revisit `r`. Otherwise the grid would have been selected on the
statistic used to report it.

### G.5 The complete main Stage-1 sequence — exactly 11 runs

| Stage | Runs | Grid | LR | `r` | Seed |
|---|---|---|---|---|---|
| **LR pilot** | **3** | `{1e-4, 3e-4, 1e-3}` | swept | fixed `r = 1` | `selection` = **21230** |
| **`r` Phase 1** | **5** | `r ∈ {0.25, 0.5, 1, 2, 4}` | **frozen winner** from the pilot | swept | `selection` = **21230** |
| **FINAL MAIN Stage-1** | **3** | — | **selected** | **selected** | `train\|0,1,2` = **36930, 7309, 5993** |
| | **TOTAL = 11** | | | | |

**There is no additional main Stage-1 training round after the final three.**
This audit does not write "Phase 2 → Stage-1 run" anywhere, because that phrasing
would imply a twelfth-and-onward training stage that does not exist. The three
final runs *are* the main Stage-1 result.

LR is *not* co-searched with `r`: the pilot runs first at `r = 1`, its winner is
frozen, and only then does the `r` sweep run. No interaction search.

**Not a significance test.** No p-value, no threshold, no pass/fail.

### G.6 The update-budget rule — precommitted, one continuation, then stop

Approved: `batch_size = 128`, `eval_every_updates = 500`,
`initial_max_updates = 20 000`.

**Exactly one automatic continuation rule, precommitted here:**

| # | Rule |
|---|---|
| 1 | Train to update **20 000** |
| 2 | If the checkpoint chosen by the locked validation rule (§G.4 / H.17) is **at update 20 000**, **continue the SAME run** from its last checkpoint/state to update **40 000** |
| 3 | The continuation **preserves** adapter state, optimizer state, corruption `visit`/pass state, and every deterministic stream. **It does not restart from scratch** |
| 4 | Checkpoint selection then considers the **complete trajectory** (0 → 40 000), not just the continuation segment |
| 5 | If the selected checkpoint is **again the final update (40 000)**, **STOP** and mark that run/config **BUDGET-LIMITED** |
| 6 | **No 60 k, 80 k or further extension may be added after inspecting results** |

**Why this is a stopping rule and not a downstream decision.** The trigger is a
property of the *Stage-1 held-out* selection — "the best checkpoint is the last
one we computed", i.e. the budget bound rather than the optimum. It reads no
downstream score, and the ceiling is fixed **before** any run. An open-ended
"extend until it stops improving" would let training length be tuned on the
selection metric; a single precommitted doubling with an explicit
**BUDGET-LIMITED** label reports the limitation instead of optimising against it.

Rule 3 matters for reproducibility specifically: the corruption draw is keyed on
`(seed, sample_id, visit)`, so a continuation that reset `visit` would silently
re-serve the *same* corruptions rather than continuing the schedule.

---

## H. THE CONFIGURATION TABLE

**Status vocabulary.** Four statuses, kept distinct so that engineering
convenience is never presented as evidence:

| Status | Meaning |
|---|---|
| **LOCKED** | Researcher-approved and recorded in `decisions.md`. Backed by prior evidence or by an explicit a-priori decision |
| **LOCKED — A-PRIORI ENGINEERING** | Researcher-approved and recorded, **but with no Stage-1 evidence behind it**. Fixed *before* any Stage-1 result existed. Kept as its own tier so that an engineering choice is never later re-read as something Stage-1 discovered |
| **EMPIRICALLY SELECTED LATER** | Chosen by a precommitted pilot. **The winner is not known now and is not guessed here** |
| **OPEN** | Still needs a decision or a pin |

| # | Item | Existing status | Evidence | Value / policy | Rationale | Decision status |
|---|---|---|---|---|---|---|
| 1 | **Main backbone** | was OPEN (D-B3B0-002) | Audits 006, 010, 013, 016, 017, 019, 024–027 all on this revision | `vinai/phobert-base` @ `01daacda68afe13d83023d16ec647239e344a1e6`, `d = 768`, frozen | Every probe that could have rejected it passed on it | **LOCKED** (D-B3B0-007) |
| 2 | **Stage-1 corpus** | was OPEN (§5 table, §13.3) | §D.2 | **`undertheseanlp/UVW-2026`**, using **all three** root parquet shards `train` + `validation` + `test`, concatenated in that fixed order. The upstream labels are **source shards, not a split**, and carry no scientific meaning for UNMARK | Clean Vietnamese text suits orthographic-equivalence learning; article identity available; reproducible; no labelled-domain matching needed. Using one shard would discard corpus for no reason; honouring the upstream split would import a partition never designed for this study | **LOCKED** (D-S1B-002) |
| 2a | **Corpus revision + hashes** | not pinned | public `main` **observed** at review time as `a0a79294e4568137e25828bb3f2a4cde8546e1fb`, **unverified — nothing was downloaded** | Explicit full HF revision + sha256 of **all three** parquet files, verified at load | An unpinned corpus is as unreproducible as an unpinned backbone, and `main` moves. The observed value is recorded as an observation, never as the pin | **OPEN — MUST PIN BEFORE EXECUTION** |
| 2b | **Contamination contract** | contradictory in draft | §D.1 | Screen exact/canonical duplicates against **already-opened** UIT-VSFC material only; TEST stays sealed; post-unsealing audit is **report-only** | Verifying non-overlap with TEST would require opening TEST | **LOCKED** (D-S1B-002) |
| 3 | **Corpus train/dev split** | not specified | D-PREG1-014 precedent | Document/article-level split, seeded from `UNMARK-STAGE1-v1`; dev = fixed **5 000 documents**, not a percentage | Document-level prevents near-duplicate leakage; a fixed count keeps selection-signal variance stable as corpus size changes. **The 5 000 is an engineering choice** | **LOCKED — A-PRIORI ENGINEERING** |
| 4 | **`max_length`** | was OPEN (D-S1A-003) | D-PREG1-008b fixed 256 *by protocol*; Audit 022 measured zero overflow at 256 | **256** | Matches precedent and the validated tokenizer contract; not selected from data | **LOCKED** |
| 5 | **Chunking + overflow** | was OPEN (D-S1A-003) | §D.2 steps 4–6, §D.3 | **Split first, chunk second**: document-level partition, then deterministic chunking, and **every chunk inherits its parent document's partition**. Runtime `on_overflow = FAIL` as a **guard** | Chunking before splitting would let two chunks of one article land on opposite sides — near-duplicate leakage into the very signal that selects `r` and the LR. Silent `SKIP` would bias the corpus toward short documents invisibly | **LOCKED** |
| 6 | **`lambda_align`** | OPEN (D-S1A-005) | §4.6 "tuned on a development split" | `2 / (1 + r)` | Fixes absolute scale so only the tradeoff varies (§G.2) | **EMPIRICALLY SELECTED LATER** (via `r`) |
| 7 | **`lambda_clean`** | OPEN (D-S1A-005) | as above | `2r / (1 + r)` | as above | **EMPIRICALLY SELECTED LATER** (via `r`) |
| 7a | **`r` grid + rule** | not specified | §G.4 | `r ∈ {0.25, 0.5, 1, 2, 4}`; Phase 1 one seed, Phase 2 three seeds, descriptive only | Precommitted rule with a well-defined tie-break at one seed | **LOCKED** (procedure) / **EMPIRICALLY SELECTED LATER** (winner) |
| 8 | **Optimizer** | OPEN | `preg1_protocol` house style | **AdamW**, betas `(0.9, 0.999)`, eps `1e-8`, amsgrad `False` | Reuses the audited house style rather than inventing a second convention | **LOCKED** |
| 9 | **Learning rate** | OPEN | pre-G1's `0.01` is a *linear head on frozen features* — not transferable | 3-point pilot **`{1e-4, 3e-4, 1e-3}`** at `r = 1`, selected on §G.4's score | 3.55 M adapter under a frozen encoder; 3 runs, not a search | **EMPIRICALLY SELECTED LATER** — **the winner is not known now** |
| 10 | **Batch size** | OPEN | none for Stage-1; 90 GB VRAM; 3 encoder forwards/step; pre-G1 used 128 | **128** | Comfortable at `d=768`, len 256, 3 branches, and matches precedent. **This is engineering, not evidence** — no Stage-1 measurement supports 128 over 64 or 256 | **LOCKED — A-PRIORI ENGINEERING** |
| 11 | **Budget** | OPEN | none specified | **Update budget, not epochs.** `initial_max_updates = 20 000`; **exactly one** automatic continuation to **40 000** under the §G.6 rule; then STOP and mark **BUDGET-LIMITED** | The 20 000 is an a-priori engineering choice, not a finding. It is made safe by a **precommitted** stopping rule rather than by judgement: the continuation trigger reads only the Stage-1 held-out selection, the ceiling is fixed before any run, and no further extension may be added after seeing results | **LOCKED — A-PRIORI ENGINEERING** (rule) |
| 12 | **Warmup / schedule** | OPEN | `LR_SCHEDULE = CONSTANT`, no warmup, in the audited pre-G1 protocol | **Constant LR, no warmup** | Simplest defensible choice, matching precedent; warmup is an ablation | **LOCKED** |
| 13 | **Gradient accumulation** | OPEN | precedent `= 1`; VRAM is not a constraint | **1** | Accumulation simulates large batches on small GPUs; not needed, and it adds a silent effective-batch confound | **LOCKED** |
| 14 | **Gradient clipping** | OPEN | precedent `None`; cosine loss bounded in `[0, 2]` | **None initially**, grad-norm monitored (item 23). **Revisit trigger:** a non-finite loss, or grad norm exceeding its running median by >100× | Clipping an already-bounded objective adds a knob with no motivating failure. Any revisit is driven by the training diagnostic, **never** by a downstream score | **LOCKED** (initial, under the stated diagnostic trigger) |
| 15 | **Weight decay** | OPEN | `WEIGHT_DECAY_WEIGHT = 0.01`, `WEIGHT_DECAY_BIAS = 0.0` house style | **0.01** on fusion/gate weight matrices; **0.0** on biases, LayerNorm, **tone embeddings** and **letter embeddings** | Two strengths of claim, kept separate: the **0.01/0.0 split** is precedent-following engineering; the **embedding exclusion** has a real argument — decaying the tone/letter tables shrinks channel information toward zero, the opposite of Stage-1's purpose. Neither is Stage-1-validated | **LOCKED — A-PRIORI ENGINEERING** |
| 16 | **Stage-1 seeds** | OPEN | `derive_seeds` convention, verified to reproduce the committed pre-G1 seeds | Root tag `UNMARK-STAGE1-v1`, **domain-separated per role** (§D.4): selection **21230**; final main runs **36930 / 7309 / 5993**; corruption **35422** | Every integer is recomputable from its tag string alone, so none can have been chosen to flatter a result. Separate namespaces make it structurally impossible for training, selection and corruption to share an integer | **LOCKED** — recorded before first use |
| 17 | **Checkpoint selection** | OPEN | pre-G1 `CHECKPOINT_RULE` shape | **Lowest held-out worst-case condition distance**, then **lower `d_clean`**, then **earliest update** | Same shape as the audited rule; "earliest" breaks ties toward less training | **LOCKED** |
| 18 | **Corruption-rate distribution** | **LOCKED** (§4.6, §5.1) | `LOCKED_STAGE1_VALUES` | `p ~ U(0,1)` per example, continuous — **unchanged** | Already locked; untouched by this review | **LOCKED (restated)** |
| 19 | **Redraw schedule** | was OPEN (D-S1A-004) | mechanism resolved; `visit` already required | **Redraw per visit**, `visit = epoch/pass index` | Per-example-fixed `p` would give one corruption level per example for the whole run, wasting the continuous distribution §4.6 insists on | **LOCKED** |
| 20 | **Letter-diacritic policy** | was OPEN; §F found **zero STRIP-ALL support** | §F.2–F.4: letter channel bit-identical in **0/18** | **Per-example scope mixture**; `pi_strip = 0.25` | The headline evaluation condition must have training support; smallest repair, no audited engine code changed | **LOCKED** (D-S1B-003) — **mechanism still to implement** |
| 20a | **Stream separation** | not specified | §F.7 | `rate_for` and `scope_for` use **domain-separated** streams (`"stage1-rate"` / `"stage1-scope"`); `p` uniform **within each scope** | Conditioning scope on `p` would confound "letters missing" with corruption severity | **LOCKED** (D-S1B-003) |
| 21 | **Resume / persistence** | not specified | §G.6 rule 3 | **best + last** checkpoint; **optimizer state**; **corruption `visit`/pass state**; all deterministic streams. Artifacts record backbone revision, corpus revision + all three file sha256, seeds, `pi_strip`, `r`, LR | Not a preference but a **correctness constraint**: the corruption draw is keyed on `(seed, sample_id, visit)`, so a resume that dropped `visit` would silently re-serve the same corruptions instead of continuing the schedule | **LOCKED** |
| 22 | **Evaluation cadence** | not specified | none | Every **500 updates** on the held-out split at the fixed condition grid | Sets checkpoint-selection granularity. **No evidence sets 500** — finer costs more, coarser risks stepping over the best checkpoint | **LOCKED — A-PRIORI ENGINEERING** |
| 22a | **Validation condition grid** | not specified | §G.4 | **`{FULL, P50, P100, STRIP_ALL}`**, fixed | Configs must be compared on identical corruptions, not on random `p` | **LOCKED** |
| 23 | **Diagnostic signals** | not specified | Audit 019 established which are informative | `loss`, `loss_align`, `loss_clean` **separately**; `d_c` per condition; `d_clean` **trajectory** (§G.3); mean gate activation (saturation); grad-norm for all **8** adapter groups; **tone and letter embedding grad norms both nonzero**; encoder grad norm **exactly 0**; `BaseInvarianceViolation` count **0**; fraction of examples where channels differ; **representation-collapse watch** — per-batch SD of `h'` across examples | Each targets a specific failure this project has seen or guarded. The collapse watch replaces the withdrawn §G.1 claim with an observation | **LOCKED** |
| 24 | **D-B3B0-002** | was OPEN | §E | Closed; revision locked | see §E | **LOCKED** (D-B3B0-007) |
| 25a | **Eligibility policy** | was `UNRESOLVED` (GAP-2) | `active_eligibility_policy()` returns `VIETNAMESE_SYLLABLE_INVENTORY`; inventory loaded, sha256 pinned | Resolved — **but `configs/corruption/default.yaml` still says `policy: UNRESOLVED`, `provisional: true`** | Stale YAML contradicting runtime; must be corrected before it reaches a run artifact | **OPEN — STALE CONFIG, NON-BLOCKING** |
| 25b | **Corruption seed** | OPEN | distinct from `stage1_seed` | Derived from `UNMARK-STAGE1-v1`, recorded separately from the training seed | Two different seeds; conflating them makes the corruption stream irreproducible | **LOCKED** |
| 25c | **`VARIANT` condition** | KNOWN DEFERRED GAP (D-B2-005) | needs `TonePlacement.TRADITIONAL` | Stays deferred; **not** a Stage-1 training condition | Out of scope; noted so it is not silently forgotten | **DEFERRED (unchanged)** |
| 25d | **Stage-2 pooling** | OPEN (D-G1-005) | pre-G1 `FIRST_TOKEN` scoped to pre-G1 only | Unchanged — **still OPEN**, not a Stage-1 item | Recorded so this review cannot be read as closing it | **OPEN (unchanged, out of scope)** |
| 25e | **D-G1-004** | OPEN | full §6 grid head values | Unchanged — Stage-2 concern | Out of scope | **OPEN (unchanged, out of scope)** |
| 25f | **Adapter capacity** | locked architecture | `6d² + 16d` = **3 551 232** at `d=768`, confirmed in Audit 019 | **Unchanged.** A larger adapter is a later capacity ablation and must not retroactively replace the precommitted main architecture | Explicit researcher instruction | **LOCKED** |
| 25g | **Corpus/domain ablation** | not specified | §D.2 | CulturaX-vi or similar **may** be explored later; must not retroactively replace the main result | Same rule as the backbone ablation | **DEFERRED** |

---

## I. THE STAGE-1 CONFIGURATION

One internally consistent block. Every line carries its status:
`# LOCKED`, `# LOCKED-ENG` (researcher-approved **a-priori engineering** — no
Stage-1 evidence behind it), `# PILOT` (empirically selected later — **winner not
known now**), or `# PIN`.

```yaml
schema_version: stage1-config-v1

backbone:                                     # LOCKED (D-B3B0-007)
  checkpoint: vinai/phobert-base
  revision:   01daacda68afe13d83023d16ec647239e344a1e6
  hidden_size: 768
  frozen: true

adapter:                                      # LOCKED (architecture precommitted)
  fusion: linear                              # W_f in R^(d x 3d)
  layernorm: after_fusion_before_gate_combination
  trainable_parameters: 3551232               # 6d^2 + 16d at d=768

corpus:
  dataset: undertheseanlp/UVW-2026            # LOCKED (D-S1B-002)
  revision: <PIN>                             # PIN - observed main a0a79294...,
                                              #   UNVERIFIED, must not be trusted
  shards:                                     # LOCKED - all three, in this order
    - {file: train.parquet,      sha256: <PIN>}
    - {file: validation.parquet, sha256: <PIN>}
    - {file: test.parquet,       sha256: <PIN>}
  shard_labels_are_a_split: false             # LOCKED - source shards only
  pipeline_order:                             # LOCKED (D.2) - order is load-bearing
    - concatenate_three_shards_in_listed_order
    - preserve_document_ids
    - contamination_screen                    # exact/canonical, opened material only
    - document_level_partition                # SPLIT ...
    - deterministic_chunking                  # ... THEN CHUNK
    - chunks_inherit_parent_partition
  contamination:                              # LOCKED (D-S1B-002, see D.1)
    screen_against: [uitvsfc_derived_train, uitvsfc_official_validation]
    method: exact_canonical_duplicate         # canon(x) equality / sha256
    test_split: SEALED_NOT_OPENED
    post_unsealing_audit: REPORT_ONLY
  split:
    kind: document_level                      # LOCKED (leakage control)
    dev_documents: 5000                       # LOCKED-ENG
    invariant: no_article_spans_train_and_dev # LOCKED - structural

chunking:                                     # LOCKED contract (D.3)
  runs_after: document_level_partition
  partition: inherited_from_parent_document
  preserve_text_order: true
  extra_normalization: none                   # canon() only
  chunk_id: "{document_id}#{chunk_index}"
  must_fit: [reference_path, base_path]       # both branches
  never_split: syllable_span

sequence:
  max_length: 256                             # LOCKED
  on_overflow: FAIL                           # LOCKED - a guard, not a data policy
  truncation: not_offered                     # LOCKED - would desync B3 projection

corruption:
  rate_distribution: uniform_0_1_per_example  # LOCKED (proposal 4.6, 5.1)
  redraw: per_visit                           # LOCKED
  scope_policy: per_example_mixture           # LOCKED (D-S1B-003)
  pi_strip: 0.25                              # LOCKED - a-priori, never tuned
  streams:                                    # LOCKED (D-S1B-003, see F.7)
    rate_tag:  "stage1-rate"
    scope_tag: "stage1-scope"
    required_property: "P(p | scope) = U(0,1) for BOTH scopes"
  eligibility: VIETNAMESE_SYLLABLE_INVENTORY  # resolved; stale YAML to correct
  seed: 35422                                 # LOCKED - UNMARK-STAGE1-v1|corruption

objective:                                    # LOCKED form (proposal 4.6)
  distance: cosine
  level: pooled
  pooling: attention_masked_mean_non_special
  scale: lambda_align + lambda_clean = 2      # LOCKED (G.2)
  r: <PILOT from {0.25, 0.5, 1, 2, 4}>        # PILOT
  lambda_align: 2 / (1 + r)
  lambda_clean: 2r / (1 + r)

optimizer:
  name: adamw                                 # LOCKED
  betas: [0.9, 0.999]                         # LOCKED
  eps: 1.0e-8                                 # LOCKED
  amsgrad: false                              # LOCKED
  schedule: constant                          # LOCKED
  warmup: none                                # LOCKED
  gradient_accumulation_steps: 1              # LOCKED
  gradient_clipping: none                     # LOCKED (initial; trigger H.14)
  learning_rate: <PILOT from {1e-4, 3e-4, 1e-3}>   # PILOT
  weight_decay:                               # LOCKED-ENG
    weights: 0.01                             # fusion / gate matrices
    bias_layernorm_tone_letter_embeddings: 0.0

training:
  batch_size: 128                             # LOCKED-ENG
  eval_every_updates: 500                     # LOCKED-ENG
  initial_max_updates: 20000                  # LOCKED-ENG
  budget_rule:                                # LOCKED (G.6) - precommitted
    if_selected_checkpoint_at: 20000
    then: continue_same_run_to_40000          # NOT a restart
    preserve: [adapter_state, optimizer_state, corruption_visit_state, streams]
    selection_considers: full_trajectory_0_to_40000
    if_selected_checkpoint_at_40000: STOP_AND_MARK_BUDGET_LIMITED
    further_extension_allowed: false          # no 60k/80k after seeing results
  persistence:                                # LOCKED
    keep: [best_by_rule, last]
    include: [adapter_state, optimizer_state, corruption_visit_state]

seeds:                                        # LOCKED - derive_seeds(tag), D.4
  root_tag: UNMARK-STAGE1-v1
  selection:  21230                           # UNMARK-STAGE1-v1|selection
  train:     [36930, 7309, 5993]              # UNMARK-STAGE1-v1|train|{0,1,2}
  corruption: 35422                           # UNMARK-STAGE1-v1|corruption

run_plan:                                     # LOCKED (G.5) - exactly 11 runs
  - {stage: lr_pilot,          runs: 3, r: 1,        seed: selection}
  - {stage: r_phase1,          runs: 5, lr: frozen,  seed: selection}
  - {stage: final_main_stage1, runs: 3, seeds: train}
  total_runs: 11
  note: the final 3 runs ARE the main Stage-1 adapters; nothing follows them

selection:                                    # LOCKED (D-S1B-001)
  data: stage1_heldout_unlabeled_only
  forbidden: any UIT-VSFC or downstream score
  condition_grid: [FULL, P50, P100, STRIP_ALL]
  metric: max over condition_grid of mean cosine distance to h(x)
  tie_break: [lower d_clean, earliest update]
  r_phase1_tie_break: [lower d_clean, smaller r]
  phase2_may_reopen_selection: false

monitoring:                                   # LOCKED (H.23)
  d_clean_trajectory: true                    # plateau OBSERVED, not assumed
  representation_collapse_watch: true         # per-batch SD of h' across examples
  encoder_grad_norm_must_be: 0
  embedding_grad_norms_must_be_nonzero: [tone, letter]
  base_invariance_violations_must_be: 0
```

---

## J. WHAT REMAINS OPEN

Exactly three things, and only the first is not code:

1. **The corpus pin** (item 2a) — the exact `UVW-2026` revision and the sha256 of
   **all three** parquet shards. Public `main` was *observed* as
   `a0a79294e4568137e25828bb3f2a4cde8546e1fb`, **unverified**; `main` moves and
   is not the pin. **Execution may not begin without this.**
2. **`scope_for`** — decided (D-S1B-003) but **not implemented**. Until it exists
   with its ML-free tests, STRIP-ALL training support is **still zero**.
3. **The corpus/chunking pipeline** — loader, contamination screen,
   document-level partition, pre-chunker, all to the §D.2/§D.3 contracts.

**Empirical by design, not open questions:** the learning rate (from
`{1e-4, 3e-4, 1e-3}`) and `r` (from `{0.25, 0.5, 1, 2, 4}`). Both are resolved by
the precommitted pilots of §G.5. **Neither winner is known or guessed here.**

Deliberately untouched: Stage-2 pooling (D-G1-005), D-G1-004, `VARIANT`
(D-B2-005), the GRR degenerate-denominator policy, and the stale
`configs/corruption/default.yaml` (non-blocking). None is a Stage-1 value.

---

## K. SELF-AUDIT

| # | Check | Result |
|---|---|---|
| 1 | Audit 028 revised **in place**; no Audit 029 created | **yes** |
| 2 | No Stage-1 runner implemented | **yes** |
| 3 | No training, no optimizer step | **yes** |
| 4 | No model or dataset downloaded; `UVW-2026` **not** fetched | **yes** |
| 5 | Official UIT-VSFC TEST untouched and unopened | **yes** |
| 6 | No scientific value written into production config | **yes** — no config file modified |
| **Revision-1 corrections** | | |
| 7 | Contamination contradiction removed | **yes** — §D.1; TEST stays sealed, screening limited to already-opened material |
| 8 | Exact/canonical vs fuzzy checks distinguished | **yes** — §D.1 rules 3–4 |
| 9 | No claim of "zero TEST overlap" before unsealing | **yes** — §D.1 rule 5; §D.2 states it is *not* guaranteed |
| 10 | Post-unsealing audit is REPORT-ONLY | **yes** — §D.1 rules 6–7, with the reason |
| 11 | `decisions.md` checked for the same contradiction | **yes** — D-S1B-001 had no overlap clause; the contract is added to D-S1B-002 |
| 12 | "irreducible floor" / "cannot reach 0" withdrawn | **yes** — §G.3, marked as a correction at the point of the error |
| 13 | Plateau to be **observed**, not assumed | **yes** — §G.3; `d_clean` trajectory added to monitoring (§H.23) |
| 14 | Collapse claim corrected to the defensible statement | **yes** — §G.1; collapse demoted to a monitored hypothesis |
| 15 | Stage-1 objective unchanged | **yes** — no change to `objective.py`, or to the loss form |
| 16 | `r`-sweep tie-break well-defined at one seed | **yes** — §G.4; SD term removed from Phase 1 |
| 17 | Phase 2 cannot reopen the `r` grid | **yes** — stated explicitly, with the reason |
| 18 | No larger combinatorial HPO added | **yes** — still 3 + 5 + 3 = 11 runs |
| 19 | `pi_strip = 0.25` locked a-priori, never tuned | **yes** — §F.8, D-S1B-003 |
| 20 | `rate_for` / `scope_for` domain-separated | **yes** — §F.7, distinct namespace tags |
| 21 | Required uniformity property stated | **yes** — `P(p \| scope) = U(0,1)` for **both** scopes |
| 22 | Scope not conditional on `p`; no shared scalar draw | **yes** — explicitly forbidden, with the confounding reason |
| 23 | Corruption engine unchanged | **yes** — only `CorruptionRatePolicy` gains a method |
| 24 | Corpus decision recorded with rationale | **yes** — §D.2, D-S1B-002 |
| 25 | Corpus **not** downloaded; revision/hashes flagged as a pin | **yes** — item 2a, OPEN before execution |
| 26 | CulturaX-vi framed as a later ablation only | **yes** — item 25g |
| 27 | Silent SKIP removed as the normal policy | **yes** — §D.3; runtime is now `FAIL` as a guard |
| 28 | Chunking contract covers all five requirements | **yes** — order, no extra normalization, stable ids, fits both paths, no split syllable |
| 29 | `max_length = 256` retained | **yes** |
| **Honesty of status labels** | | |
| 30 | Four-tier status vocabulary used | **yes** — LOCKED / PROPOSED ENGINEERING DEFAULT / EMPIRICALLY SELECTED LATER / OPEN |
| 31 | batch size, budget, eval cadence, dev size, weight decay labelled as engineering | **yes** — items 3, 10, 11, 15, 22, each saying no Stage-1 evidence supports it |
| 32 | Update budget's arbitrariness stated, with a detection rule | **yes** — item 11: a selected checkpoint at the final update means the budget was too short |
| 33 | LR and `r` winners not guessed | **yes** — both marked `<PILOT>`; no value asserted |
| 34 | Only researcher-approved items locked in `decisions.md` | **yes** — no pilot value locked |
| 35 | pre-G1 not rerun, grid not widened, not reinterpreted as significance | **yes** — §C unchanged |
| 36 | Proposal PDF not regenerated | **yes** |
| 37 | One internally consistent configuration block | **yes** — §I, every line status-tagged |
| 38 | Everything unstaged; no prohibited git operation | **yes** |
| **Revision-2 finalisation** | | |
| 39 | All three UVW parquet shards used, in a fixed documented order | **yes** — §D.2, §I `shards` |
| 40 | Upstream train/validation/test stated to carry **no** split meaning | **yes** — §D.2; `shard_labels_are_a_split: false` |
| 41 | Pipeline order load → screen → **split** → **chunk** → inherit | **yes** — §D.2 steps 1–6, §I `pipeline_order` |
| 42 | No article's chunks can span Stage-1 train and dev | **yes** — structural: partition decided at document level before any chunk exists (§D.3 rules 6–7) |
| 43 | Observed `main` recorded as an observation, never as the pin | **yes** — item 2a says **unverified**; nothing was downloaded |
| 44 | Exactly 11 main Stage-1 runs, stated unambiguously | **yes** — §G.5 table, `total_runs: 11` |
| 45 | Final 3 runs identified as the **final main adapters** | **yes** — §G.4, §G.5; "nothing follows them" |
| 46 | No phrasing implying a training stage after Phase 2 | **yes** — the one remaining "→ Stage-1 run" in §L was found and rewritten |
| 47 | Phase 2 cannot reopen LR **or** `r` | **yes** — §G.4, `phase2_may_reopen_selection: false` |
| 48 | Budget rule precommitted: 20 k → one continuation → 40 k → STOP | **yes** — §G.6, six explicit rules |
| 49 | Continuation preserves adapter/optimizer/`visit`/stream state, no restart | **yes** — §G.6 rule 3, with the reproducibility reason |
| 50 | Selection considers the full 0→40 k trajectory | **yes** — §G.6 rule 4 |
| 51 | BUDGET-LIMITED marker; no 60 k/80 k after inspection | **yes** — §G.6 rules 5–6 |
| 52 | Budget rule reads no downstream score | **yes** — trigger is the Stage-1 held-out selection only |
| 53 | Engineering defaults locked **and** still documented as a-priori engineering | **yes** — the `LOCKED — A-PRIORI ENGINEERING` tier exists precisely for this |
| 54 | Seeds derived by the repository's convention, not invented | **yes** — `derive_seeds`; verified to reproduce `TUNING_SEEDS = (5509, 19422, 11800)` |
| 55 | Seed roles domain-separated; exact integers recorded before any run | **yes** — §D.4: 21230 / 36930 / 7309 / 5993 / 35422 |
| 56 | All five seed integers distinct | **yes** — verified |
| 57 | No code implemented, no data downloaded, no training run | **yes** |

---

## L. NEXT ACTION

1. **Pin** the `UVW-2026` dataset revision and the sha256 of **all three**
   parquet shards (item 2a). This is the last blocking OPEN item that is not
   code.
2. Correct the stale `configs/corruption/default.yaml` eligibility block.
3. Implement `scope_for` in `CorruptionRatePolicy` — **mechanism only** — with
   ML-free tests proving: STRIP-ALL support exists, P100 support survives, the
   two streams are independent, and `p` is uniform **within each scope**.
4. Implement the corpus loader (three shards, fixed order), the contamination
   screen, the document-level partition and the deterministic pre-chunker to the
   §D.2/§D.3 contracts, with tests — including one asserting that no article's
   chunks can span train and dev.
5. Implement the Stage-1 runner, including the §G.6 budget rule.
6. **PRE-TRAIN audit** of that runner — the major audit, which this one is not.
7. Execute the **11-run** main sequence of §G.5: LR pilot (3) → `r` Phase 1 (5) →
   **final main Stage-1 (3)**. The final three runs **are** the main Stage-1
   adapters; no training stage follows them.

---

**STATUS: CONFIG LOCK COMPLETE (REVISION 2) — READY FOR IMPLEMENTATION ON APPROVAL**
**CORPUS: `undertheseanlp/UVW-2026`, ALL THREE SHARDS, SPLIT BEFORE CHUNK, CHUNKS INHERIT PARTITION**
**RUN PLAN: EXACTLY 11 — 3 LR PILOT + 5 `r` PHASE 1 + 3 FINAL MAIN ADAPTERS; NOTHING FOLLOWS**
**BUDGET: 20 000 → ONE CONTINUATION → 40 000 → STOP, MARK BUDGET-LIMITED**
**SEEDS RECORDED BEFORE FIRST RUN: selection 21230; train 36930/7309/5993; corruption 35422**
**REMAINING OPEN: THE CORPUS REVISION + THREE FILE HASHES (PIN), AND IMPLEMENTATION**
**STRIP-ALL SUPPORT: DECIDED (`pi_strip = 0.25`) BUT NOT YET IMPLEMENTED — STILL BLOCKING**
**ENGINEERING DEFAULTS: LOCKED AS A-PRIORI CHOICES, NOT AS STAGE-1 FINDINGS**
**LOCKED: pre-G1 CLOSED; UIT-VSFC EXCLUDED; BACKBONE LOCKED (D-B3B0-002 CLOSED); CORPUS `undertheseanlp/UVW-2026`; SCOPE MIXTURE + STREAM SEPARATION**
**EMPIRICAL, NOT GUESSED: learning rate and `r`**
**ENGINEERING DEFAULTS LABELLED AS SUCH: batch size, update budget, eval cadence, dev size, weight decay**
**FOUR FIRST-DRAFT CLAIMS WITHDRAWN IN PLACE (§D.1, §G.1, §G.3, §G.4)**
**OFFICIAL TEST SEALED — NOT OPENED, NO OVERLAP CLAIM MADE**
**NO RUNNER, NO TRAINING, NO MODEL LOAD, NO DOWNLOAD**
**ALL CHANGES UNSTAGED**
