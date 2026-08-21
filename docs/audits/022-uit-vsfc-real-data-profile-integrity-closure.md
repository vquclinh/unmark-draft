# Audit 022 — UIT-VSFC real-data profile integrity closure and profiler contract completion

| | |
|---|---|
| **Audit id** | 022 |
| **Created (UTC)** | 2026-08-21 |
| **Last revised (UTC)** | **2026-08-21** |
| **Scope** | Record the first real UIT-VSFC profiling pass; resolve the one annotation conflict it exposed; repair two profiler-contract gaps |
| **Repository state** | `HEAD = 7654ce1bba3eb93d55d7821fbcf10b1fb6741bf9`; this work uncommitted |
| **Predecessors** | [020](020-minimal-stage2-g1-evaluation-harness.md), [021](021-pre-g1-dataset-profile-and-protocol-precommit.md) |
| **Phase** | pre-G1 |
| **Type** | **Real-data integrity decision + profiler repair.** No training, no optimizer, no head, no downstream score |
| **Revision 1** | **2026-08-21** — **scientific wording repair, in place.** (a) The token-length prose conflated the **per-example** delta p99 (**+14**) with the shift between two **marginal** p99 quantiles (55 → 68, **+13**); both are now stated separately. (b) The exclusion rationale claimed a shared label error "does not cancel in `Delta_s`" — too categorical; it now says cancellation is **not guaranteed**. **No real metric, no locked value and no decision changed**; the exclusion policy is untouched. |

---

## A. VERDICT

**CONDITIONAL PASS / COLAB RERUN REQUIRED**

Three separate statuses, and they must not be collapsed into one:

| # | Status | Verdict |
|---|---|---|
| 1 | **Data integrity** | **PASS** — on the real evidence already observed. The one conflicting canonical group is resolved by explicit whole-group exclusion; the derived view has zero duplicates, zero conflicts, zero cross-split leakage. |
| 2 | **Tokenizer / truncation feasibility** | **PASS** — the first real pinned-tokenizer profile shows **zero** train overflow at the fixed `max_length = 256`, on **both** pathways. |
| 3 | **Final profile-contract closure** | **PENDING REAL COLAB RERUN** — two contract gaps were found *after* the run. The code is repaired and tested; the **real numbers the repair produces do not yet exist.** |

**2134 local tests pass, 56 skip** (was 2100/56 at Audit 021; +34 here).

**This audit does not close the pre-G1 profile.** Statuses 1 and 2 rest on
evidence observed before the repair; status 3 cannot be satisfied by any amount
of local work, because the missing quantities are measurements of a corpus this
environment will never hold.

**No head was trained. No optimizer existed. No model weights were loaded. No
downstream score exists.** The official test split was not consulted for any
decision. **D-B3B0-002 remains OPEN.** The compiled PDF **remains stale**.

---

## B. EVIDENCE STATUS — STATED PLAINLY

Every real number in this audit was **observed on Colab and supplied to this
session**. The Colab artifacts are **not present in this repository**. They were
**not independently opened, hashed or recomputed here**, and this audit does not
imply otherwise. The local environment is ML-free and holds no corpus.

Nothing below was produced by the repaired code. The repair changes what the
profiler *reports*; the pre-repair run could not have reported it.

Full evidence record:
[`docs/experiments/preg1-uit-vsfc-real-profile-result.md`](../experiments/preg1-uit-vsfc-real-profile-result.md).

| | |
|---|---|
| **Run HEAD** | `7654ce1bba3eb93d55d7821fbcf10b1fb6741bf9` |
| **Python / transformers** | 3.12.13 / 4.57.6 |
| **torch visible in `.venv-colab`** | **false** |
| **Tokenizer** | `vinai/phobert-base` @ `01daacda68afe13d83023d16ec647239e344a1e6` |
| **Model weights loaded** | **no** — tokenizer only |

The ten official file SHA-256s are recorded in the evidence file as
**externally observed Colab evidence**. No hash was fabricated for any file, and
no corpus file exists locally to hash.

---

## C. THE OFFICIAL CORPUS

| Split | Rows | `0` neg | `1` neu | `2` pos | Empty |
|---|---|---|---|---|---|
| train | 11 426 | 5 325 | 458 | 5 643 | 0 |
| validation | 1 583 | 705 | 73 | 805 | 0 |
| test | 3 166 | 1 409 | 167 | 1 590 | 0 |

Published counts matched **exactly**, and each split's classes sum to its row
count. Access is `OFFICIAL_PUBLIC_DISTRIBUTION`; the explicit license is
`NOT_ESTABLISHED`, and this audit does not upgrade the first into the second.

---

## D. THE INTEGRITY FINDING AND THE DECISION

`DUPLICATE_CONTRACT` requires that conflicting-label groups **STOP for
researcher review** rather than be silently repaired. The real data raised that
stop exactly once.

| | |
|---|---|
| Digest | `a193a8ff49cc5ab43da189f9126aea19a0a0e9df1e16acc0a710cf7e880d0daa` |
| Members | `train:11293`, `train:11417` |
| Identity | byte-identical, NFC-identical **and** NFD-identical |
| Sentiment | one `0`, one `2` |
| Topic | identical |

Identical under every normalisation the profiler applies: an annotation
conflict, not an orthographic artifact.

**Decision — `EXCLUDE_ENTIRE_CONFLICTING_CANONICAL_GROUP`**
([D-PREG1-011](../spec/decisions.md#d-preg1-011--conflicting-canonical-groups-are-excluded-whole)).
Both members dropped. **Neither relabelled. Neither preferred over the other.
The official corpus is unmodified.**

**Why the whole group.** Keeping one member requires asserting which annotation
is right, on no evidence. The contradictory supervision is **avoidable
annotation noise**, and pairing does **not guarantee that its effect cancels**:
Vanilla and Base-only use different representations and may respond differently
during optimization, checkpoint selection or evaluation, so a common noisy label
can still land asymmetrically in `Delta_s`. Whole-group exclusion removes the
ambiguity **symmetrically**, without asserting that either annotation is
correct. Two rows in 11 426 do not justify carrying that risk.

**Timing.** Decided **before any downstream score existed.** No Vanilla-vs-Base-
only result has ever been produced on this dataset, so no score could have
influenced the exclusion, and none did.

**Adapter view vs derived view.** The exclusion produces a **derived analysis
view**, not a modified dataset. The official distribution is untouched. Official
**validation** keeps its measurement-dev role, **unfiltered**. Official **test**
remains **SEALED**.

| | Official | Derived |
|---|---|---|
| Rows | 11 426 | **11 424** |
| `0` | 5 325 | 5 324 |
| `1` | 458 | **458 — unchanged** |
| `2` | 5 643 | 5 642 |

Derived TRAIN csv SHA-256
`a20c0f7760f32dc48263a79d73ddf5363526c17e9de2afc32d8346b23444d301` — **not
committed**; the digest is the reproducibility handle.

`neutral` is untouched, which matters: it is ~4% of train and the reason the
metric is macro-F1 rather than accuracy.

Re-profiling the derived view: **0** canonical duplicate groups, **0**
conflicting-label groups, **0** cross-split groups. **No leakage.**

---

## E. ORTHOGRAPHIC AND NOISE EVIDENCE

**`base_equivalent` = "no observed mark under `b(canon(x))`". It is not a
missing-diacritic rate**, and these figures do **not** prove the corpus is
correctly diacritized — unmarked Vietnamese is observationally ambiguous (§4.3).

| Split | N | `base_equivalent` | rate | tone | letter | `canon_changed` |
|---|---|---|---|---|---|---|
| derived train | 11 424 | 15 | 0.001313 | 11 398 | 11 318 | 217 |
| validation | 1 583 | 4 | 0.002527 | 1 576 | 1 556 | 29 |
| test | 3 166 | 4 | 0.001263 | 3 155 | 3 131 | 41 |

Changed units per example (derived train): min 0, p25 9, p50 13, p75 21, p90 32,
p95 41, p99 63, max 199.

Noise, descriptive only — **0** urls, **0** mentions, **0** hashtags, **0**
emoji/symbols across all three splits; repeated runs 3 / 0 / 1; digit-bearing
tokens 802 / 73 / 208 (train figures pre-exclusion). Nothing is normalised away.

---

## F. TOKENIZER PROFILE AND TRUNCATION FEASIBILITY

Derived train, N = 11 424, lengths including the evaluator's special-token
convention.

| | min | p25 | p50 | p75 | p90 | p95 | p99 | max | mean |
|---|---|---|---|---|---|---|---|---|---|
| Vanilla | 4 | 10 | 14 | 20 | 29 | 36 | 55 | 163 | 16.5804 |
| Base-only | 4 | 12 | 16 | 24 | 35 | 45 | 68 | 202 | 20.1119 |
| Base − Vanilla | −1 | 2 | 3 | 5 | 7 | 9 | 14 | 39 | 3.5314 |

Length changed: **10 642 / 11 424 = 0.93155**.

| Budget | Vanilla | Base-only | Joint |
|---|---|---|---|
| 64 | 0.994573 | 0.986432 | 0.986432 |
| 128 | 0.999912 | 0.999650 | 0.999650 |
| **256** | **1.000000** | **1.000000** | **1.000000** |

**Zero overflow at 256 on both pathways.**

`max_length` stays **fixed at 256**. This profile **confirms feasibility and
does not reopen the selection** — note that 128 would also have cleared a 99%
rule, which is precisely the data-dependent choice
[D-PREG1-008b](../spec/decisions.md#d-preg1-008b--max_length-fixed-at-256-not-selected-from-data)
exists to prevent.

Stripping marks lengthens sequences, and the two ways of saying so are **not
the same statistic**:

- the **per-example** Base-minus-Vanilla token-length delta has median **+3**
  and **p99 +14** pieces;
- separately, the **marginal** p99 sequence length shifts from **55**
  (Vanilla) to **68** (Base-only), a shift of **+13**.

A quantile of the per-example differences is not the difference of the two
marginal quantiles, and only the first describes what happens to a given
example. Reporting "p99 +13" as a delta would conflate them.

**This is a tokenization observation, not a downstream performance result**, and
it does **not** reopen `max_length` selection. It is a hypothesis about why the
base-only pathway might be burdened — not evidence that it is.

---

## G. GAP 1 — UNIT-LEVEL CHANNEL DENSITIES

**The defect.** Audit 021 precommitted tone and letter observed-**unit**
densities. The profiler delivered example-level `with_observed_tone` /
`with_observed_letter`, which answer only "does this sentence carry any mark at
all". A corpus where every sentence carries one mark and a corpus where every
sentence is fully marked are **indistinguishable** under that counter, yet they
are entirely different inputs to a tone/letter channel.

**Denominators were derived from §4.3 and the authoritative code, not from the
task prompt.** §4.3 fixes tone as a **syllable** property — one syllable carries
exactly one tone — and letter diacritics as a **character** property, several of
which may sit on different characters of one syllable.

| Channel | Denominator | Numerator |
|---|---|---|
| tone | syllables with `Eligibility.VIETNAMESE_CANDIDATE` | those whose `ObservedTone` is not `UNMARKED` |
| letter | character units whose `LetterDiacritic` is **not `NA`** | those whose `LetterDiacritic` is neither `NA` nor `NONE` |

**`NA` is not folded into `NONE`.** `NONE` = a letter that *could* carry a
Vietnamese letter diacritic and does not. `NA` = the channel does not apply at
all (digits, punctuation, whitespace, symbols). Counting `NA` in the denominator
would deflate every letter density by the corpus's punctuation and digit rate —
on this corpus, 802 digit-bearing tokens in train alone.

**Unresolved eligibility reports `null`, never `0`.** The tone denominator needs
the B3A syllable inventory; without it every syllable is `UNDECIDED`. The
profiler then returns `None`, serialised as JSON `null`, matching the
fail-visible discipline B2 applies through `EligibilityPolicy.UNRESOLVED`. **A
tone density of `0.0` is a finding; a missing inventory is a defect**, and the
artifact must not let the second impersonate the first. The letter density does
not depend on the inventory and stays defined.

**Aggregation** is `sum(numerators) / sum(denominators)`, not a mean of
per-example rates — the latter would weight a three-word sentence equally with a
forty-word one.

**No orthography was reimplemented.** `observe_orthography` delegates to
`canon` and `decompose`; the eligibility classifier is the B3A one, passed in.
The profiler cannot disagree with the pipeline it profiles.

**Example-level fields are retained**, so no earlier artifact is reinterpreted.

Recorded as
[D-PREG1-012](../spec/decisions.md#d-preg1-012--channel-densities-are-measured-per-unit-at-43-granularity).
Schema `preg1-profile-v1` → **`preg1-profile-v2`**.

---

## H. GAP 2 — UNK COUNT PATHWAY AMBIGUITY

**The defect, established by reading the code rather than assuming it.** In
`scripts/preg1_dataset_profile.py::tokenize_lengths`, a single `unk` accumulator
sat **inside** the loop that ran over both the canonical and the base-only
surface. The emitted `unk_token_count = 4` is therefore a **Vanilla + Base-only
sum with no attribution** — equally consistent with 4/0, 0/4, and every split
between.

**Why it matters here specifically.** The question a two-pathway token profile
exists to answer is whether stripping marks pushes text *out* of the tokenizer's
vocabulary. A summed counter cannot answer it in either direction.

**The repair.** `vanilla_unk_token_count` and `base_only_unk_token_count` are
reported separately; `total_unk_token_count` is retained as an explicitly named
aggregate. **Tokenization itself is unchanged** — a test pins that the tokenizer
still receives exactly the canonical surface followed by the base surface and
nothing else.

**The pre-patch `4` is not reinterpreted.** It cannot be attributed
retroactively; the rerun supplies the split.

Recorded as
[D-PREG1-013](../spec/decisions.md#d-preg1-013--unk-counts-are-attributed-to-a-pathway).

---

## I. WHAT WAS REPAIRED

| File | Change |
|---|---|
| `unmark/evaluation/profiling.py` | schema → `preg1-profile-v2`; `UNIT_DENSITY_SEMANTICS`; four counters + `eligibility_resolved` on `OrthographyObservation` and `SplitProfile`; `observed_tone_unit_density` / `observed_letter_unit_density` (`None` when unresolved); `classifier` parameter on `observe_orthography` and `profile_split` |
| `scripts/preg1_dataset_profile.py` | pathway-separated UNK counts; B3A classifier wired into `profile_split`; `eligibility_resolved` stamped into the summary; density table in the report |
| `unmark/evaluation/preg1_protocol.py` | `CONFLICTING_GROUP_POLICY`, `CONFLICTING_GROUP_EXCLUSION_SCOPE`, `OBSERVED_CONFLICTING_GROUPS`, `DERIVED_TRAIN_SIZE`, `DERIVED_TRAIN_LABEL_COUNTS`, `DERIVED_TRAIN_CSV_SHA256`; surfaced in the run manifest; version → `preg1-protocol-v4` |
| `tests/test_preg1_profiling.py` | +34 ML-free tests |
| `docs/spec/decisions.md` | D-PREG1-011, D-PREG1-012, D-PREG1-013 |
| `docs/experiments/preg1-uit-vsfc-real-profile-result.md` | new, raw-text-free evidence record |

**No raw UIT-VSFC text was added anywhere.** A structural test asserts the
protocol module contains no Vietnamese-marked characters at all.

**New real values from the patched code are still PENDING.** Every number in
sections C–F came from the pre-repair run.

---

## J. WHAT WAS NOT DONE

- **No head trainer implemented.**
- **No 80/20 split materialised.**
- **No Stage-1 training or HPO.** Stage-1 is untouched.
- **No model weights loaded**, locally or otherwise.
- **No dataset or model downloaded locally.** The `.venv` remains ML-free.
- **No tokenization behaviour changed.**
- **No locked Audit-021 value changed** — dataset, task, split roles, seal,
  `max_length`, padding, pooling, head shape and init, decay groups, optimiser,
  schedule, batch, epochs, checkpoint eligibility, LR grid, seeds, selection
  rule: all preserved.
- **No prohibited git operation.** Nothing staged, committed, tagged or pushed.

---

## K. TASK-END SELF-AUDIT

| # | Check | Result |
|---|---|---|
| 1 | Audit 021 reread in full | **yes** |
| 2 | Proposal §4.3 and channel code reread **before** defining denominators | **yes** — granularity taken from §4.3, not the prompt |
| 3 | No orthography reimplemented | **yes** — delegates to `canon`/`decompose`/B3A classifier |
| 4 | Numerator/denominator semantics explicit | **yes** — field names + `UNIT_DENSITY_SEMANTICS` in the artifact |
| 5 | `NA` handling matches authoritative semantics | **yes** — `NA` excluded from the letter denominator, `NONE` included |
| 6 | Example-level tone/letter fields retained | **yes** |
| 7 | UNK counts pathway-unambiguous | **yes** — two named counts + named aggregate |
| 8 | No tokenizer transformation changed | **yes** — pinned by test |
| 9 | Exclusion is the entire group, no relabel | **yes** |
| 10 | Official dataset unchanged | **yes** — derived view only |
| 11 | No downstream score influenced the exclusion | **yes** — none exists |
| 12 | Official validation role unchanged | **yes** — measurement-dev, unfiltered |
| 13 | Official test sealed | **yes** |
| 14 | `max_length` remains 256 | **yes** — feasibility confirmed, selection not reopened |
| 15 | RAW_BASE / no segmentation unchanged | **yes** |
| 16 | No head trainer | **yes** |
| 17 | No real data or model downloaded locally | **yes** |
| 18 | No training or HPO | **yes** |
| 19 | Stage-1 unchanged | **yes** |
| 20 | D-B3B0-002 | **OPEN** |
| 21 | Compiled PDF | **STALE** |
| 22 | Audit says final patched rerun is PENDING | **yes** — §A status 3 |
| 23 | Tests pass | **2134 passed, 56 skipped** |
| 24 | `git diff --check` | **clean** |
| 25 | Everything unstaged | **yes** |
| 26 | No prohibited git operation | **yes** |
| 27 | **Revision 1:** per-example delta p99 (**+14**) distinguished from the marginal p99 shift 55 → 68 (**+13**) | **yes** — §F |
| 28 | **Revision 1:** no real metric altered | **yes** — every supplied number is byte-identical |
| 29 | **Revision 1:** no claim remains that paired annotation noise is *guaranteed* not to cancel | **yes** — §D, D-PREG1-011, `CONFLICTING_GROUP_POLICY` docstring |
| 30 | **Revision 1:** rationale now says cancellation is **not guaranteed** | **yes**, and that exclusion removes the ambiguity **symmetrically** |
| 31 | **Revision 1:** D-PREG1-011 still whole-group exclusion, no relabel | **yes** — unchanged |
| 32 | **Revision 1:** unit-density and UNK repairs unchanged | **yes** — no behavioural code touched |

### Weaknesses I am recording against myself

1. **Statuses 1 and 2 rest on evidence I did not verify.** I did not open the
   Colab artifacts; I cannot. Both PASSes are conditional on the supplied
   numbers being what the run produced. I have marked them as externally
   observed everywhere they appear rather than presenting them as verified.
2. **Gap 1 was precommitted in Audit 021 and shipped unmet.** The Audit-021
   tests exercised the profiler against synthetic fixtures and passed, because
   they tested what the code did rather than what the protocol promised. That is
   the same class of defect as the prose-matching tests found earlier in this
   project: a test that cannot fail for the reason it was written.
3. **The unit densities have never run against real data.** They are tested
   against fixtures with hand-computed expectations. A fixture cannot reveal a
   denominator that is wrong in a way I have not imagined.
4. **The derived-CSV digest is unverifiable here.** I recorded
   `a20c0f77…` as supplied. If the rerun produces a different digest, that is a
   finding, and this file should not be read as having confirmed it.
5. **I conflated two quantiles** (Revision 1). Writing "p99 +13" for what is
   actually a shift between two marginal p99 values, alongside a per-example
   median, read as a single per-example delta summary. The correct per-example
   p99 is **+14**. The table was right throughout; the prose summarising it was
   not, which is the harder kind of error to catch because the numbers verify.
6. **I overstated the paired-design argument** (Revision 1). "It does not cancel
   in `Delta_s`" asserts more than I can support: shared label noise may have
   partially common effects across arms, and pairing simply does not *guarantee*
   cancellation either way. The decision was already correct; the justification
   was stronger than the evidence. Stating "not guaranteed" is both weaker and
   true, and it is still sufficient to justify the exclusion.

---

## L. REQUIRED NEXT ACTION

Rerun `scripts/preg1_dataset_profile.py` on Colab against the real corpus, from
the repaired code, and record the results as the **authoritative** pre-G1
profile. Until then status 3 stands at **PENDING**.

The rerun must supply: tone and letter unit densities for all three splits with
`eligibility_resolved = true`; pathway-separated UNK counts; and a derived-train
digest to check against `a20c0f77…`.

---

```
AUDIT CREATED:
docs/audits/022-uit-vsfc-real-data-profile-integrity-closure.md

VERDICT:
CONDITIONAL PASS / COLAB RERUN REQUIRED

DATA INTEGRITY:
PASS

TOKENIZER/TRUNCATION FEASIBILITY:
PASS

FINAL PROFILE-CONTRACT CLOSURE:
PENDING REAL COLAB RERUN

CONFLICTING GROUP POLICY:
EXCLUDE ENTIRE GROUP / NO RELABEL

OFFICIAL TRAIN ROWS:
11426

DERIVED PRE-G1 TRAIN ROWS:
11424

MAX_LENGTH:
256 FIXED

UNIT-DENSITY PROFILER REPAIR:
IMPLEMENTED / AWAITING REAL RERUN

UNK PATHWAY REPORTING:
UNAMBIGUOUS / AWAITING REAL RERUN

80/20 SPLIT:
NOT MATERIALIZED

HEAD TRAINER:
NOT IMPLEMENTED

REAL DOWNSTREAM SCORE:
NONE

STAGE-1 TRAINING:
NOT RUN

D-B3B0-002:
OPEN

PDF:
STALE

COMMIT CREATED:
NO
```
