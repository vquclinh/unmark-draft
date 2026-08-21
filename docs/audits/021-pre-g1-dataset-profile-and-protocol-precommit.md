# Audit 021 — pre-G1 dataset profiler and protocol precommitment

| | |
|---|---|
| **Audit id** | 021 |
| **Created (UTC)** | 2026-08-20 |
| **Last revised (UTC)** | **2026-08-21** |
| **Scope** | Data-only dataset profiler; precommit the pre-G1 burden-diagnostic protocol |
| **Repository state** | `HEAD = e73637f`; this work uncommitted |
| **Predecessors** | [019](019-stage1-real-phobert-diagnostic-closure.md), [020](020-minimal-stage2-g1-evaluation-harness.md) |
| **Phase** | pre-G1 |
| **Type** | **Data-only infrastructure + precommitment.** No training, no optimizer, no dataset read |
| **Revision 1** | 2026-08-20 — **pre-result protocol lock**: SA-VLSP2016 superseded by **UIT-VSFC v1.0**; access model repaired; `max_length` fixed at 256; splits and full probe protocol locked. Verified before editing that **zero** downstream scores existed. |
| **Revision 2** | **2026-08-21** — **paired reproducibility lock**: explicit head initialisation, bit-identical paired starts, optimiser parameter groups, checkpoint eligibility. **No Stage-1 science or experimental result changed; the pre-G1 reproducibility specification was tightened before any real-data run.** Those four items *are* protocol decisions (D-PREG1-010), recorded as such. |

---

## A. VERDICT

**PASS — PROTOCOL LOCKED ON UIT-VSFC v1.0; NO DATASET PROFILED, NO DOWNSTREAM
TRAINING**

**2100 local tests pass, 56 skip.** The profiler, the repaired access model, the
deterministic splitter and the full probe protocol exist and are tested against
synthetic fixtures.

**No real dataset was read**, **no head was trained**, **no downstream score
exists**, and **the official test split was not used for any decision**.
**D-B3B0-002 remains OPEN.** The compiled PDF **remains stale**.

### The supersession, and why it is legitimate

An earlier revision of this audit selected **SA-VLSP2016**. That is **superseded
by UIT-VSFC v1.0** (D-PREG1-001b).

**Timing is what makes the change defensible, so I verified it before editing:**
no `results/preg1` directory existed, no pre-G1 experiment record existed, and no
`Delta_s` had ever been computed. **Zero real Vanilla-vs-Base-only scores
existed.** A dataset change *after* seeing a result would be indefensible;
before one, it is ordinary specification work — and the history is superseded
transparently rather than erased.

| Superseded | By |
|---|---|
| D-PREG1-001 — SA-VLSP2016 | **D-PREG1-001b — UIT-VSFC v1.0**, sentiment only |
| D-PREG1-002 — `authorisation_established` boolean | **D-PREG1-002b — four-state `DatasetAccess`** |
| D-PREG1-004 — 70/15/15 internal split | **D-PREG1-004b — official validation as measurement; train 80/20** |
| D-PREG1-008 — smallest of {64,128,256} at ≥99% | **D-PREG1-008b — fixed `max_length = 256`** |

**Three facts I recomputed rather than accepted:** the split seed 17486
reproduces exactly from `UNMARK-PREG1-SPLIT-UITVSFC-v1`; the tuning and
measurement seeds still reproduce from their tags; and the published class counts
sum **exactly** to 11,426 / 1,583 / 3,166.

## B. SCIENTIFIC PURPOSE

This task builds the **data-only** evidence needed before the *pre-G1
Vanilla-vs-Base-only clean-path burden diagnostic* can be run at all: is the
chosen dataset clean, duplicate-free and short enough to support it?

The distinction Audit 020 established is preserved throughout:

| | |
|---|---|
| **Stage-1 pooling** | masked mean, LOCKED §4.6, alignment objective only |
| **Stage-2 head pooling (full §6 grid)** | **OPEN** (D-G1-005) |
| **Pre-G1 burden diagnostic** | descriptive, no threshold — what is precommitted here |
| **Full G1** | §7: attach fusion layer, train briefly, gate towards identity — unchanged |

---

## C. DATASET DECISION AND LITERATURE RATIONALE

**Active: UIT-VSFC version 1.0**, Vietnamese Students' Feedback Corpus,
**sentiment task only**, labels `0 negative / 1 neutral / 2 positive`.

| Split | Size | negative | neutral | positive |
|---|---|---|---|---|
| train | 11,426 | 5,325 | 458 | 5,643 |
| validation | 1,583 | 705 | 73 | 805 |
| test | 3,166 | 1,409 | 167 | 1,590 |

Each row sums exactly to its split size — checked, not assumed. `neutral` is
about **4%** of train, which is why macro-F1 and per-class F1 are the reported
metrics rather than accuracy alone.

### Why SA-VLSP2016 was superseded

Not because of any result — none existed. Because the two datasets optimise
**different things**, and the earlier choice optimised the wrong one for *this*
measurement. The pre-G1 diagnostic wants the **cleanest identifiable
`x -> b(x)` manipulation**, not the most realistic noisy social-media benchmark.
Those goals pull in opposite directions.

### Why UIT-VSFC

* The original paper describes an explicit **normalization phase** — sentence
  segmentation, abbreviation expansion, misspelling correction, personal-name
  anonymisation — producing >16,000 normalized sentences.
* It has an **official train/validation/test structure**, so the measurement set
  need not be carved out of train.
* Its size makes a stable paired probe inexpensive.
* Its **official validation split can stay untouched** by head-protocol tuning —
  which is what lets protocol selection and measurement use genuinely different
  data.

### What is explicitly not claimed

**This does not claim the corpus is perfectly diacritized.** A paper's
normalization description is not evidence about orthographic exposure. The
profiler must measure that directly on the real data (§C1), and the audit trail
records the actual values after the Colab run.

### The dataset is LOCKED

Profiling is an **integrity and characterisation gate**, not a
downstream-score-based selection contest. If profiling reveals a catastrophic
integrity problem, **STOP and require a new explicit researcher decision** — do
not automatically switch again. SA-VLSP2016 remains eligible for the later full
benchmark.

### C1. Direct profiling is mandatory before head training

The profiler must measure, on the real selected dataset: base-equivalent rate;
orthographic exposure; tone and letter observed-unit densities; the changed-unit
distribution; ASCII/no-mark descriptives; URLs, mentions, hashtags, emoji;
repeated-character runs; digit-bearing tokens; duplicate groups;
conflicting-label groups; cross-split duplicates; Vanilla and Base-only
token-length distributions; and the **overflow rate at 256**.

## D. DATA ACCESS / LICENSING / PROVENANCE STATUS

UIT's official NLP dataset page lists **UIT-VSFC (version 1.0)** with a **direct
public download**. Unlike several other datasets on that same official page, it
is **not** presented with an instruction to email the group and sign a user
agreement.

**The SA-VLSP-specific assumption is therefore removed** (D-PREG1-002b). The old
boolean `authorisation_established`, with usability gated on an *authorised*
official copy, misclassified an officially and publicly distributed corpus as
unusable for a reason that does not apply to it.

**Repaired model** — four explicit states, no default:

| `DatasetAccess` | Usable for a scientific run |
|---|---|
| `OFFICIAL_PUBLIC_DISTRIBUTION` | **yes** |
| `OFFICIAL_AGREEMENT_AUTHORISED` | **yes** |
| `MIRROR` | no |
| `UNKNOWN` | no |

**License is kept strictly separate**, and this matters: *official public
distribution* and *an explicitly identified license* are **different facts**.
`license_status` defaults to `NOT_ESTABLISHED`, is **not** part of the usability
test, and **no license is invented** — a test asserts no license identifier
appears in the module. The profiler's report prints a distinct warning for each
condition, so a reader is never left inferring one from the other.

Raw dataset files are **never redistributed through git**; artifacts carry
provenance, hashes and counts, not corpus text — a test asserts no source text
survives into a serialised profile.

## E. PROFILER CONTRACT

`unmark/evaluation/profiling.py` — torch-free, network-free, no model.

Per split it reports: example count; label counts and proportions;
empty/invalid count; canonicalisation-change count; exact duplicates; canonical
duplicates; conflicting-label groups; base-equivalent count and rate; texts with
observed tone and with observed letter marks; the changed-orthographic-unit
distribution (min / p25 / p50 / p75 / p90 / p95 / p99 / max); character-length
distribution; and noise descriptives.

**It delegates to the authoritative implementations.** `canon` and `decompose`
come from `unmark.orthography`; no stripping rule is reimplemented, so a profile
cannot disagree with the pipeline it profiles. A test asserts both are called.

**Percentiles are nearest-rank, pure Python** — deterministic, and no numpy
dependency in an ML-free package.

---

## F. ORTHOGRAPHIC CLEANLINESS METRICS

**The vocabulary is the substance here.** A text with no observed mark is
**base-equivalent** — `canon(x) == b(canon(x))` — and that is **not** a
"missing-diacritic rate". §4.3 is explicit that genuine *ngang* and a stripped
mark are indistinguishable at inference; calling mark-free text "missing
diacritics" would assert knowledge the observation cannot supply.

Every field name says what was *observed*: `base_equivalent`,
`units_with_observed_tone`, `units_with_observed_letter`, `changed_units`. A test
scans the modules to ensure no "missing diacritics" claim appears, and that the
observation object exposes no such attribute.

**No restorer is run.** Cleanliness is never guessed by restoring the text and
comparing — a test asserts no restoration call or import exists.

ASCII-only and mark-free statistics may be reported **descriptively**; they are
never labelled as evidence of loss.

**Noise descriptives** — URLs, @mentions, hashtags, emoji/non-BMP symbols,
repeated-character runs, digit-bearing tokens — are **counts only**. Nothing is
normalised away and no "teencode corrector" exists; a test asserts no correction
routine is called.

---

## G. DUPLICATE / LEAKAGE SAFETY

`analyse_duplicates` works over canonical-text digests, so it catches duplicates
that differ only in NFC/NFD form or tone placement — a test proves two spellings
of `hòa` are found as one canonical group despite being distinct exact strings.

Three findings are separated because they need different responses:

| Finding | Meaning |
|---|---|
| canonical duplicate groups | the same text appears more than once |
| **conflicting-label groups** | the same canonical text carries **different labels** |
| **cross-split groups** | one canonical text appears in more than one split |

**Conflicting labels are a labelling problem, not a de-duplication problem.**
They are **reported with their sample ids and never silently dropped or
relabelled**: either action would change the label distribution before anyone
decided to. The handling decision stays **OPEN** for the researcher.

---

## H. PHOBERT TOKEN-LENGTH PROFILE CONTRACT

Behind the established lazy-import boundary in
`scripts/preg1_dataset_profile.py`. **Colab only** — the local `.venv` stays
ML-free, and `--data-only` runs everything else locally.

Uses the exact verified `vinai/phobert-base` @
`01daacda68afe13d83023d16ec647239e344a1e6`, **tokenizer only** (the tokenizer is
not the model; no weights are loaded).

For **TRAIN ONLY**, both pathways:

* Vanilla — `canon(x)` → tokenizer;
* Base-only — `canon(x)` → `b(x)` → the same tokenizer;

reporting min / p25 / p50 / p75 / p90 / p95 / p99 / max for each, the
base-minus-vanilla delta distribution, how many sequences change length,
UNK counts, and coverage at 64 / 128 / 256.

**Lengths include special tokens** via `build_inputs_with_special_tokens`, so
they match the convention the future evaluator uses rather than a bare piece
count — a length profile that ignored `<s>`/`</s>` would under-count exactly at
the threshold that matters.

**Train only, deliberately.** The sealed test split is never tokenized for a
protocol decision; a test asserts the script reads `records["train"]` for this.

---

## I. MAX_LENGTH — FIXED AT 256

**Superseded.** The earlier rule — smallest of `{64, 128, 256}` covering ≥99% of
train on both pathways — is replaced by a **fixed `max_length = 256`**
(D-PREG1-008b), with `truncation = true` and `padding = "max_length"`, for both
pathways.

**Why.** Pre-G1 aims to **minimise truncation**, not optimise inference
efficiency, and compute is not a constraint. PhoBERT's pretrained positional
capacity is 256 for a task sequence, so this is the maximum supported length.
Fixing it removes a **data-dependent protocol decision** from the measurement:
the earlier rule made the protocol a function of the corpus, which is one more
thing that could differ between a rerun and the original.

**The statistics are still reported, and still matter.** Length distributions,
coverage at 64/128/256, the **overflow rate at 256**, and the Vanilla/Base-only
delta. They now **characterise** the corpus and quantify truncation instead of
selecting a value. If records overflow 256, the exact aggregate rate is reported
and ordinary truncation applies; the verified backbone limit is never exceeded
automatically.

**The selection machinery was removed, not merely left unused.** A test asserts
`select_max_length`, `max_length_evidence`, `MaxLengthUnresolved` and the old
constants **no longer exist**, so a future caller cannot silently re-enable
data-driven selection — leaving dead code would have made that a one-line
regression.

---

## J. TEST-SET SEAL AND VALIDATION ROLE

| Split | Role |
|---|---|
| official **test** | **SEALED** — integrity, hash and duplicate checks only; **never** protocol decisions, **never** scores |
| official **validation** | **measurement-dev** — never used to select dataset, pooling, LR, epoch, or any head hyperparameter; the head is never tuned on it |
| official **train** | internally split 80/20 (§K) |

§5.4 permits test for "one final evaluation, for the tables that appear in the
paper" — and the pre-G1 diagnostic is **not** that evaluation.

Every run artifact stamps `official_test_sealed: true` and
`official_test_used_for_protocol_decisions: false`; tests assert both, and that
length profiling reads **train only**.

---

## K. INTERNAL TRAIN SPLIT — 80/20

**Superseded.** The earlier 70/15/15 division existed because SA-VLSP2016 offered
no separate development split. UIT-VSFC has one, so the internal division needs
only two parts (D-PREG1-004b):

| Part | Fraction | Role |
|---|---|---|
| protocol-train | **80%** | head training |
| protocol-dev | **20%** | shared head-protocol, LR and checkpoint selection |

**Why 20% dev.** `neutral` is about **4%** of train, so a larger dev share gives
a more stable macro-F1 tuning signal on the class that will dominate the metric's
variance — while still leaving over 9,000 training examples.

**Split seed, precommitted and verified.** Tag
`UNMARK-PREG1-SPLIT-UITVSFC-v1` → SHA-256 → first 2-byte big-endian word =
**17486**. Recomputed from the tag. A **third** tag, distinct from the tuning and
measurement tags, so split, tuning and measurement randomness are independent.

**The splitter is implemented** as a generic mechanism
(`profiling.stratified_group_split`) and is **deterministic**, **label-stratified**,
**group-aware by canonical text**, and **independent of any downstream score**. It
uses a keyed digest rather than `random`, so it is stable across processes — a
test asserts no `random` import.

Tests prove it: determinism, disjointness and completeness, seed-dependence,
class presence in both parts, and that an NFC/NFD duplicate pair **lands in the
same part**.

**It is not run on real data in this phase.** Conflicting-label canonical groups
must be inspected first, and how to handle them is a researcher decision.

**Duplicate contract.** Canonical duplicates stay in one group; official
train↔validation overlap is detected and reported before head training;
conflicting-label groups are reported with ids and counts, **never** silently
relabelled or dropped; and if such groups affect split integrity, **STOP** for
researcher review before downstream training.

---

## L. PRECOMMITTED PROBE PROTOCOL

**Recorded, not implemented. Nothing trains.**

**Pathways — unchanged from Audit 020.**

```
VANILLA    canon(x)          -> PhoBERT tokenizer -> frozen encoder
BASE_ONLY  canon(x) -> b(x)  -> the SAME tokenizer -> the SAME frozen encoder
```

**No word segmenter is introduced into either pathway.** The diagnostic must
differ in the *pathway transformation* only; adding VnCoreNLP would introduce a
second variable into a measurement built to isolate one. Consistent with the
locked `RAW_BASE` contract (D-B3B1A-001).

**The caveat is preserved, not "fixed".** Standard PhoBERT usage expects
word-segmented Vietnamese, and RAW_BASE is a deliberate project design choice and
a possible source of distribution shift. Silently segmenting would hide that.

**Pooling: `<s>` first token.** No mean pooling. Scoped to **this diagnostic
only** — it does not change Stage-1's masked mean (§4.6) and does not lock
pooling for the full grid (§5.2 stays OPEN). Tests assert both.

**Head: `Linear(d, 3, bias=True)`.** `d` from `model.config.hidden_size`. No
hidden layer, no dropout, no LayerNorm, no activation. **`768` appears nowhere as
a literal** — D-B3B0-002 is OPEN, and a test enforces it.

**Loss: ordinary multiclass cross-entropy.** No class weights, no focal loss, no
label smoothing. The imbalance is **exposed** through macro-F1 and per-class F1
rather than compensated by a second modelling intervention — which would itself
become a variable in a two-pathway comparison. Per-class F1 is reported as a
diagnostic, **`neutral` above all**, since at ~4% of train it is where the two
pathways are most likely to diverge.

**Encoder and numerics.** Frozen, `eval()`, extraction under `torch.no_grad()`,
**FP32** throughout with cached representations in FP32, **no AMP, no
BF16/FP16**. Checkpoint `vinai/phobert-base` @
`01daacda68afe13d83023d16ec647239e344a1e6` — a pin for **this probe's
reproducibility** that **does not close D-B3B0-002**.

**Optimisation, fully specified.** AdamW with `betas=(0.9, 0.999)`, `eps=1e-8`,
`amsgrad=false`; **CONSTANT** schedule; warmup **0**; **no** gradient clipping;
batch size **128**; **30 complete epochs**; early stopping **OFF**; shuffling on
and deterministic under the run seed; no encoder gradient or update;
`gradient_accumulation_steps = 1`; `drop_last = false`.

**Parameter groups.**

| Parameter | weight decay |
|---|---|
| head weight matrix | **0.01** |
| head bias | **0.0** |

The classifier intercept is **not** decayed. Shrinking it pulls the decision
boundary toward the origin, which on a corpus where `neutral` is ~4% of train
would penalise the minority class through a regularisation choice rather than
through the data — and macro-F1 is precisely the metric that would absorb it.

**Loss, spelled out.**
`CrossEntropyLoss(weight=None, label_smoothing=0.0, reduction="mean")`.

**Checkpoint eligibility.** Evaluate and select after each **complete** epoch;
epochs are numbered **1..30**; **epoch 0 — the untrained head — is not
eligible.** An untrained linear head on a 4%-neutral corpus can post a
deceptively reasonable accuracy by favouring a majority class, and letting it win
a checkpoint would report the initialisation rather than the pathway.

### L1. Paired reproducibility lock (D-PREG1-010)

**The gap this closes.** §L previously said the pathways share a run seed.
Sharing a seed **label** is not the same as starting from identical parameters:
if Vanilla ran first and Base-only then drew from the same advancing RNG stream,
the second head would start from **different** weights, and that difference would
land in `Delta_s` attributed to the pathway. For a paired measurement built to
isolate one variable, that is the failure most likely to pass unnoticed.

**Head initialisation, explicit:**

```
weight -> torch.nn.init.xavier_uniform_
bias   -> torch.nn.init.zeros_
```

`nn.Linear`'s implicit default is **not** relied upon — it is a Kaiming-uniform
variant whose exact form has changed across PyTorch versions, so depending on it
would make the comparison silently version-sensitive.

**Per-run sequence**, for every independent head-training run:

1. reset the run RNGs from the declared run seed **before** head construction;
2. construct the head;
3. **explicitly** apply xavier-uniform to the weight and zeros to the bias;
4. construct the deterministic shuffle generator from the same run seed.

**The paired guarantee.** For measurement seed `s`, `Vanilla(s)` and
`BaseOnly(s)` start from **bit-identical** classifier parameters, because each
re-seeds from `s` rather than inheriting RNG state from the other. The **data
order is paired too**: same example ids, same labels, same deterministic shuffle
schedule. Only the input pathway differs — which is what makes `Delta_s` a
genuinely paired measurement rather than two independent runs subtracted.

**Runtime options are not hyperparameters.** Implementation-level AdamW options
(`foreach`, `fused`, `capturable`) vary by PyTorch version. They must **not** be
tuned; the run artifact records the actual runtime version and the options in
force.

---

## M. PRIMARY VS SECONDARY TUNING DISTINCTION

**Primary LR search.** Grid `{1e-4, 3e-4, 1e-3, 3e-3, 1e-2}` — **five**
candidates. For each LR and each of the **3 tuning seeds**: train the linear head
for all 30 epochs, evaluate every epoch on protocol-dev, and select that run's
checkpoint by

1. highest protocol-dev macro-F1, 2. then higher accuracy, 3. then earliest epoch.

Aggregate across the three tuning seeds by

1. highest **mean** selected-checkpoint macro-F1, 2. then highest mean accuracy,
3. then **lowest sample SD of macro-F1**, 4. then **smaller** learning rate.

The SD tie-break is deliberate: between two LRs with equal mean, the more stable
one is the better protocol choice, and preferring the smaller LR last keeps the
rule fully deterministic.

**Vanilla only**, before any measurement-dev result is observed. The winner is
then **frozen and reused unchanged for both pathways**. Official validation is
never used in this selection, and the grid is not altered after viewing Base-only
results.

**The caveat travels with it.** Tuning on Vanilla does **not** make Vanilla an
upper bound. It makes the protocol shared and the comparison interpretable; it
does not establish that Base-only could not do better under its own tuning.

**Secondary sensitivity — precommitted, not run.** Each pathway may later select
its own LR under **exactly** the same grid, the same 3 tuning seeds, the same
protocol-dev, the same 30-epoch budget and the same checkpoint rule. It answers
*"best achievable head fit under equal tuning budget"* — a **different question**
— and **must not replace** the headline primary shared-LR result.

---

## N. PAIRED-SEED REPORTING

**Three precommitted, derivable seed sets**, from three different tags:

| Purpose | Tag | Seeds |
|---|---|---|
| Split | `UNMARK-PREG1-SPLIT-UITVSFC-v1` | **17486** |
| Tuning | `UNMARK-PREG1-TUNE-v1` | 5509, 19422, 11800 |
| Measurement | `UNMARK-PREG1-MEASURE-v1` | 53148, 59945, 42941, 720, 9428 |

**Derivation:** SHA-256 of the ASCII tag, read as successive 2-byte big-endian
words. **I recomputed all three sets and they match exactly** — which is the
point: a reader can verify none was chosen after seeing a result. Three separate
tags keep split, tuning and measurement randomness independent, and a test
asserts the sets are disjoint.

**Measurement.** For each of the 5 measurement seeds, train a Vanilla head and a
Base-only head sharing split, LR, optimizer, scheduler, loss, batch size, epoch
budget, seed, checkpoint criterion, architecture, `max_length` and precision.

**Each pathway trains its own head through its own clean pathway and may select
its own best epoch** under the same checkpoint rule. "Same protocol" does **not**
require an identical epoch number — demanding one would force a pathway onto a
checkpoint its own dev curve did not choose, which is a worse comparison, not a
fairer one.

Then freeze each selected head and evaluate on the **untouched official
validation** split. Report `Delta_s = Score_vanilla_s - Score_baseonly_s` for
**macro-F1** (primary) and **accuracy** (secondary): all five raw paired scores,
all five deltas, `mean(Delta)`, sample `std(Delta)`, and raw per-pathway
mean/std. Plus **per-class F1** for negative/neutral/positive.

Pairing matters: the two pathways share a seed, so the per-seed difference
removes seed-to-seed variance that would otherwise swamp the effect.

**No p-value is required for n = 5, and none is invented.** Five paired
observations do not support a significance claim.

**No pre-G1 threshold exists**, and full G1's "within approximately 1 point" is
**not borrowed**. **Neither result may be called an upper bound or ceiling on
UNMARK.**

---

## O. OPEN ITEMS BEFORE REAL PRE-G1 RUN

The protocol is now **fully locked**; what remains is data work and
implementation:

1. **Obtain UIT-VSFC v1.0** from the official public distribution and record its
   provenance (`access = OFFICIAL_PUBLIC_DISTRIBUTION`, per-file SHA-256).
2. **Run the data-only profile** and record the actual orthographic, duplicate
   and token-length values in the audit trail.
3. **Handling of conflicting-label canonical duplicates**, if the profile finds
   any — reported, never auto-resolved.
4. **Materialise the split** once (3) is known; the mechanism exists.
5. **The head trainer/driver**, which does not exist.
6. Still open beyond this diagnostic: §5.2's head values for the **full §6
   grid**; the full-G1 "≈1 point" interpretation; **D-B3B0-002**.

**No longer open** — locked by this revision: dataset, `max_length`, splits and
split seed, pooling, head, loss, optimizer, schedule, budget, LR grid, checkpoint
and aggregation rules, seeds, and reporting.

---

## P. TEST EVIDENCE

```
.venv/bin/python -m pytest -q -p no:cacheprovider
2100 passed, 56 skipped in 8.72s
```

Baselines: 2057 before revision 1, 2085 after it, **2100** after revision 2.
`tests/test_preg1_profiling.py` now holds **101 tests**, all ML-free and
network-free on synthetic fixtures.

**Added by revision 2 (15 tests):** explicit xavier-uniform weight and zero-bias
initialisation; `nn.Linear`'s default provably not relied upon; the four-step
per-run sequence re-seeding **before** construction; bit-identical paired starts;
no shared advancing RNG; paired deterministic data order; weight decay 0.01 on
the matrix and **0.0 on the bias**; mean-reduced unweighted CE with zero label
smoothing; `gradient_accumulation_steps = 1` and `drop_last = false`; exactly
epochs 1..30 eligible; **epoch 0 provably unable to win**; runtime options not
being hyperparameters; the reproducibility block serialising; the protocol
version recording the lock; and the whole contract remaining **ML-free** — the
initialiser is stored as a *name*, not imported.

**Added or rewritten by this repair:** UIT-VSFC as the active dataset with
version and task; SA-VLSP superseded and not active; labels exactly
negative/neutral/positive; **published counts summing to their split sizes**;
dataset locked and not score-selected; no word segmenter (structurally — no
import, no call); official validation measurement-only; test sealed; internal
split exactly 80/20; **split seed 17486 recomputed from its tag**; all three seed
sets disjoint; splitter determinism, disjointness, completeness,
seed-dependence, stratification, **NFC/NFD duplicates staying together**, no
global RNG, and bad-fraction rejection; `max_length` fixed at 256 with the
selection machinery **proven absent**; coverage reported descriptively and low
coverage **not** changing the value; official-public-distribution usable;
agreement-authorised usable; mirror and unknown not usable; **license separate
from access** and no license invented; the old boolean gone; `<s>` pooling scoped
with Stage-1 untouched; `Linear(d,3,bias=True)` with no hardcoded 768; plain CE
with no weights or smoothing; AdamW fields exact; constant schedule, zero warmup,
no clipping; 30 epochs, no early stopping; **five** LR candidates including
`1e-2`; ordered checkpoint and aggregation rules; primary LR Vanilla-only and
frozen for both; secondary clearly secondary; shared protocol with per-pathway
epoch freedom; five paired measurement seeds against official validation;
per-class F1 diagnostics; no significance claim; no threshold; and — across all
three modules — no torch, no optimizer, no training, no dataset import.

An end-to-end smoke run confirmed the repaired script reports
`OFFICIAL_PUBLIC_DISTRIBUTION` as usable while `license_established` stays
`false`, and prints `max_length : 256 (fixed)`.

**Two test defects of my own**, both fixed: a check banning the substring
`"vncorenlp"` matched my own prose stating it is *not* used, and a provenance
test still referenced the renamed `--source-type` flag. The first was rewritten
as a structural import/call check — the same prose-matching trap as in earlier
phases.

---

## Q. SCIENTIFIC NON-ACTIONS

* **No downstream training** — no head trainer exists.
* **No Stage-1 training.**
* **No optimizer** — none constructed; asserted structurally across all modules.
* **No HPO run** — the LR search is recorded, not executed.
* **No real Vanilla-vs-Base-only score.**
* **No official test evaluation** — test is sealed and untouched by any decision.
* **No model download locally**; no dataset download; no network access.
* **No dataset profiled** — only synthetic fixtures were read.
* **D-B3B0-002 OPEN.**
* **Compiled PDF stale** — not regenerated, not claimed synchronised.

---

## R. NEXT STEP

1. Obtain **UIT-VSFC v1.0** from the official public distribution.
2. Run the **data-only** profile in fresh Colab and record the actual
   orthographic, duplicate and token-length evidence — including the **overflow
   rate at 256**.
3. Inspect conflicting-label canonical duplicates; **STOP** for researcher review
   if they affect split integrity.
4. Materialise the 80/20 split with seed 17486.
5. Only then implement the pre-G1 head trainer/driver.

**It is still not Stage-1 HPO.** And when the burden diagnostic does run, it
remains **descriptive**: a PASS does not discharge G1, a FAIL does not equal a G1
FAIL, and neither is a ceiling on UNMARK.

---

## S. GIT STATE

`HEAD = e73637f`.

```
 M docs/spec/decisions.md
 M unmark/evaluation/__init__.py
?? docs/audits/021-pre-g1-dataset-profile-and-protocol-precommit.md
?? scripts/preg1_dataset_profile.py
?? tests/test_preg1_profiling.py
?? unmark/evaluation/preg1_protocol.py
?? unmark/evaluation/profiling.py
```

`git diff --check` is clean. Everything is left **unstaged**.

**No prohibited git operation was used.** No `add`, `commit`, `push`, `tag`,
`stash`, `reset`, `checkout` or `restore`. No unrelated researcher work was
touched.

---

## T. TASK-END SELF-AUDIT

Verified at the close of revision 2. Facts were **recomputed or inspected**, not
copied from the prompt.

### Dataset and process

| Check | Result |
|---|---|
| UIT-VSFC v1.0 is the only **ACTIVE** pre-G1 dataset | **OK** |
| SA-VLSP2016 exists only as superseded history / future candidate | **OK** — `SUPERSEDED_DATASET`, absent from comparators |
| **Zero** downstream scores existed before the switch | **OK** — no `results/preg1`, no experiment record, no `Delta_s` |
| Official validation is **measurement-only** | **OK** — role `measurement-dev`; never used for LR, epoch, pooling or any head value |
| Official test is **SEALED** | **OK** — integrity/hash/duplicate checks only |
| **No license invented** | **OK** — `NOT_ESTABLISHED` default, separate from access; no license identifier in the module |

### Data handling

| Check | Result |
|---|---|
| `max_length = 256` | **OK** — fixed; selection machinery **removed**, not merely unused |
| Internal split **80/20**, seed **17486** | **OK** — seed recomputed from `UNMARK-PREG1-SPLIT-UITVSFC-v1` |
| Canonical duplicates cannot cross the internal split | **OK** — group-aware splitter; NFC/NFD pair lands together |
| **No word-segmentation change** | **OK** — no segmenter imported or called; RAW_BASE caveat preserved |

### Probe protocol

| Check | Result |
|---|---|
| Pooling = `<s>` first token | **OK** — scoped to this diagnostic; Stage-1 untouched |
| Head = `Linear(d, 3, bias=True)` | **OK** — no hidden layer, dropout, LayerNorm or activation; **no hardcoded 768** |
| **Xavier-uniform weight / zero bias**, explicit | **OK** — `nn.Linear` default not relied upon |
| **Same-seed paired heads identical by contract** | **OK** — each run re-seeds before construction; no shared advancing RNG |
| **Deterministic paired shuffle** | **OK** — same ids, labels and schedule per seed |
| Weight decay **W = 0.01 / bias = 0.0** | **OK** |
| **Mean plain CE**, no class weights, no label smoothing | **OK** |
| `gradient_accumulation_steps = 1`, `drop_last = false` | **OK** |
| AdamW / LR grid / scheduler / warmup / budget unchanged | **OK** — 5 candidates incl. `1e-2`, constant, warmup 0, batch 128, 30 epochs, no early stopping |
| Checkpoint epochs **1..30 only**; epoch 0 cannot win | **OK** |
| Three tuning seeds unchanged | **OK** — 5509, 19422, 11800 |
| Five measurement seeds unchanged | **OK** — 53148, 59945, 42941, 720, 9428 |
| Primary shared LR unchanged | **OK** — Vanilla-only, then frozen for both |
| Own-best-epoch rule unchanged | **OK** — same criterion, epoch may differ |
| **No pre-G1 threshold** | **OK** — descriptive; §7's "≈1 point" not borrowed |

### Non-actions and hygiene

| Check | Result |
|---|---|
| No real profile, training or HPO performed | **OK** |
| Stage-1 unchanged | **OK** — 0 changed files under `unmark/stage1/` and `unmark/modeling/` |
| **D-B3B0-002 OPEN** | **OK** — the checkpoint pin explicitly does not close it |
| Compiled **PDF stale** | **OK** — not regenerated, not claimed synchronised |
| No prohibited git action | **OK** |
| All changes unstaged | **OK** — 0 staged |
| `git diff --check` | **OK** — clean |
| Full local suite | **OK** — **2100 passed, 56 skipped** |

### Bookkeeping (revision 3)

| Check | Result |
|---|---|
| Top-level verdict states **2100 passed, 56 skipped** | **OK** — was stale at 2085 after revision 2 |
| Historical **2085** kept only where labelled revision-1 history | **OK** — §P's "2057 before revision 1, 2085 after it, 2100 after revision 2" is unchanged |
| Revision 2 no longer claims "No science changed" | **OK** — it now states that no **Stage-1** science or experimental **result** changed, while naming the four items as protocol decisions (D-PREG1-010) |
| No locked scientific value changed | **OK** — dataset, splits, seeds, `max_length`, pooling, head, init, decay, loss, budget, LR grid, checkpoint rule all unchanged |
| No scientific code changed | **OK** — documentation only; `unmark/` and `tests/` untouched in this pass |

```text
AUDIT FILE REVISED IN PLACE:
docs/audits/021-pre-g1-dataset-profile-and-protocol-precommit.md

NEW AUDIT CREATED:
NO

ACTIVE PRE-G1 DATASET:
UIT-VSFC v1.0

TASK:
3-class sentiment

OFFICIAL VALIDATION ROLE:
MEASUREMENT-DEV

OFFICIAL TEST:
SEALED

MAX_LENGTH:
256

POOLING:
FIRST TOKEN <s>

HEAD:
LINEAR d->3

HEAD INITIALIZATION:
XAVIER_UNIFORM WEIGHT / ZERO BIAS

PAIRED INITIALIZATION:
IDENTICAL WITHIN MEASUREMENT SEED

WEIGHT DECAY:
WEIGHT 0.01 / BIAS 0.0

REAL DATASET PROFILE:
NOT RUN

REAL DOWNSTREAM TRAINING:
NOT RUN

REAL VANILLA-VS-BASE-ONLY RESULT:
NONE

STAGE-1 TRAINING:
NOT RUN

STAGE-1 HPO:
NOT RUN

D-B3B0-002:
OPEN

COMMIT CREATED:
NO
```
