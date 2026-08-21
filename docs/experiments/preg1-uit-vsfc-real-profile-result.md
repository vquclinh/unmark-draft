# Pre-G1 UIT-VSFC v1.0 real-data profile

**Status: `PREG1_REAL_PROFILE_CLOSED` — closed on real data by Attempt r2.**

Three real runs, with three different standings. They are kept separate on
purpose: a closure that hides its failed attempts is worth less than one that
shows them.

| Attempt | HEAD | Standing |
|---|---|---|
| **1** | `7654ce1b…` | pre-gap-repair **baseline** — no unit densities, UNK unattributable |
| **2** | `f828ef1e…` | patched but **NON-FINAL** — eligibility unresolved **and** schema inconsistent |
| **r2** | `5e707ecd…` | **AUTHORITATIVE / PASS** |

Jump to the [authoritative result](#attempt-r2--authoritative--pass).

---

## Attempt 1 — pre-repair baseline

**Superseded by Attempt r2.**

This records the **first** real-data profiling pass over the locked pre-G1
dataset. It is not a scientific result: no head was trained, no optimizer
existed, no model weights were loaded, and no downstream score was produced.

**Two profiler-contract gaps were found after this run** (Audit 022, gaps 1 and
2). The code has since been repaired. **The values below were produced by the
pre-repair code** and are recorded as observed history — the numbers the repair
adds (unit-level channel densities, pathway-separated UNK counts) **do not exist
in this pass and cannot be recovered from it.** Attempt r2 supersedes them.

**Raw corpus text appears nowhere in this record.** Digests, ids and counts
only — the same rule the profiler applies to its own artifacts.

| | |
|---|---|
| **Repository HEAD of the run** | `7654ce1bba3eb93d55d7821fbcf10b1fb6741bf9` |
| **Evidence status** | **externally observed on Colab.** The artifacts are not in this repository and were **not** independently opened locally. |
| **Python / transformers** | 3.12.13 / 4.57.6 |
| **torch visible in `.venv-colab`** | **false** |
| **Model weights loaded** | **no** — tokenizer only |
| **Head trained / optimizer / HPO / downstream score** | **none** |
| **Superseded by** | **Attempt r2** (authoritative) |

## Provenance

| | |
|---|---|
| Dataset | **UIT-VSFC v1.0**, sentiment task only |
| Access | **`OFFICIAL_PUBLIC_DISTRIBUTION`** |
| Explicit license | **`NOT_ESTABLISHED`** |
| Tokenizer | `vinai/phobert-base` @ `01daacda68afe13d83023d16ec647239e344a1e6` |

Public availability is **not** a license grant. `OFFICIAL_PUBLIC_DISTRIBUTION`
records how the files were obtained and nothing about redistribution rights;
see [D-PREG1-002b](../spec/decisions.md#d-preg1-002b--access-model-official-public-distribution-is-not-the-same-as-a-license).

### Official file digests — externally observed

SHA-256 of the official download, as reported by the Colab run. **These files
are not present locally and were not hashed here.**

| File | SHA-256 |
|---|---|
| `README.txt` | `6230b0e49414f9ae090439a664dede4aee53a8f12cd57a6c5cf9d6594ee2d99c` |
| `train/sents.txt` | `5481dc1fa51f2fe72f22afd89b8aeb7f8945a126af7e66ac622e2ab0291130cb` |
| `train/sentiments.txt` | `480480c3b9a6bc8bdf53339c721e93f8cc30472ac8f0bd21d4440ce1171aefac` |
| `train/topics.txt` | `09cbb8147f419f225b171cadf82410c442d6c24997ab97f48dfa779f4e68a1b9` |
| `dev/sents.txt` | `fb7c3cc3173e1383edc03779883d91bb4d6110c8dd881612572a256878aa23b4` |
| `dev/sentiments.txt` | `a9584a22c926a54c6042236380c9a65ab8c41467477f7a5d794fb2505c96a9c3` |
| `dev/topics.txt` | `7483805b24f5362aa5a2f708509876e0b69bf35753e90199059c2409caad7d96` |
| `test/sents.txt` | `75100e0559ecc0d7052870e2e7991391bfa7442bbaee91a6e271157c6fea343b` |
| `test/sentiments.txt` | `0f04ebe3ada9655aff0a4c6eec27e5e7b8f552e26d3a6f06a35831e07e102f18` |
| `test/topics.txt` | `ba04f1bd4014be3c1f1940236a8ceae71a455aa0837295bdbe2cdbcc44bd0745` |

`topics.txt` is hashed for completeness of the download identity. The **topic**
annotation is not used: this diagnostic is sentiment-only.

## Official corpus, as received

| Split | Rows | `0` negative | `1` neutral | `2` positive | Empty |
|---|---|---|---|---|---|
| train | 11 426 | 5 325 | 458 | 5 643 | 0 |
| validation | 1 583 | 705 | 73 | 805 | 0 |
| test | 3 166 | 1 409 | 167 | 1 590 | 0 |
| **total** | **16 175** | | | | **0** |

Every count matches the published figures exactly, and each split's classes sum
to its row count. `neutral` is ~4% of train, which is why the reported metric is
macro-F1.

## Integrity finding — one conflicting canonical group

TRAIN contained exactly **one** duplicate group whose members disagree on the
gold label.

| | |
|---|---|
| Digest | `a193a8ff49cc5ab43da189f9126aea19a0a0e9df1e16acc0a710cf7e880d0daa` |
| Members | `train:11293`, `train:11417` |
| Identity | byte-identical, NFC-identical **and** NFD-identical |
| Sentiment | one `0`, one `2` |
| Topic | identical |

The strings agree under **every** normalisation the profiler applies, so this is
an annotation conflict, not an orthographic artifact.

**Resolution:** `EXCLUDE_ENTIRE_CONFLICTING_CANONICAL_GROUP` — both members
dropped, neither relabelled, the official corpus untouched. Decided **before any
downstream score existed**; see
[D-PREG1-011](../spec/decisions.md#d-preg1-011--conflicting-canonical-groups-are-excluded-whole).

### Derived pre-G1 train view

| | Official | Derived |
|---|---|---|
| Rows | 11 426 | **11 424** |
| `0` negative | 5 325 | 5 324 |
| `1` neutral | 458 | **458 — unchanged** |
| `2` positive | 5 643 | 5 642 |

Derived TRAIN csv SHA-256:
`a20c0f7760f32dc48263a79d73ddf5363526c17e9de2afc32d8346b23444d301`
(**not committed** — no corpus file is; the digest is the reproducibility handle).

The **derived view is an analysis artifact.** The official distribution is
unmodified, the official **validation** split keeps its measurement-dev role
unfiltered, and the official **test** split remains SEALED.

Re-profiling the derived view found **0** canonical duplicate groups, **0**
conflicting-label groups and **0** cross-split groups. No leakage.

## Orthographic observables

**`base_equivalent` means "no observed mark under `b(canon(x))`". It is not a
missing-diacritic rate,** and none of the figures below prove the corpus is
correctly diacritized: unmarked Vietnamese is observationally ambiguous (§4.3).

| Split | N | `base_equivalent` | rate | `with_observed_tone` | `with_observed_letter` | `canon_changed` |
|---|---|---|---|---|---|---|
| derived train | 11 424 | 15 | 0.001313 | 11 398 | 11 318 | 217 |
| validation | 1 583 | 4 | 0.002527 | 1 576 | 1 556 | 29 |
| test | 3 166 | 4 | 0.001263 | 3 155 | 3 131 | 41 |

Changed orthographic units per example, derived train:

| min | p25 | p50 | p75 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|
| 0 | 9 | 13 | 21 | 32 | 41 | 63 | 199 |

**These are example-level counters.** The unit-level tone and letter densities
Audit 021 precommitted are **absent from this pass** — that is gap 1. They
cannot be back-derived from the table above, which is why a rerun was required
rather than a recomputation; **Attempt r2 supplied them.**

## Corpus noise — descriptive only

| Split | urls | mentions | hashtags | emoji/symbols | repeated runs | digit-bearing |
|---|---|---|---|---|---|---|
| train (pre-exclusion) | 0 | 0 | 0 | 0 | 3 | 802 |
| validation | 0 | 0 | 0 | 0 | 0 | 73 |
| test | 0 | 0 | 0 | 0 | 1 | 208 |

Zero URLs, mentions, hashtags and emoji across 16 175 examples is consistent
with a curated student-feedback corpus rather than scraped social text. Nothing
is normalised away; these are descriptives.

Train figures are **pre-exclusion**; the patched rerun is authoritative for the
derived view.

## Pinned tokenizer profile — derived train, N = 11 424

Lengths include the special-token convention the evaluator uses.

| | min | p25 | p50 | p75 | p90 | p95 | p99 | max | mean |
|---|---|---|---|---|---|---|---|---|---|
| Vanilla | 4 | 10 | 14 | 20 | 29 | 36 | 55 | 163 | 16.5804 |
| Base-only | 4 | 12 | 16 | 24 | 35 | 45 | 68 | 202 | 20.1119 |
| Base − Vanilla | −1 | 2 | 3 | 5 | 7 | 9 | 14 | 39 | 3.5314 |

Length changed for **10 642 / 11 424 = 0.93155** of examples.

| Budget | Vanilla | Base-only | Joint |
|---|---|---|---|
| 64 | 0.994573 | 0.986432 | 0.986432 |
| 128 | 0.999912 | 0.999650 | 0.999650 |
| **256** | **1.000000** | **1.000000** | **1.000000** |

**Overflow at 256: zero for both pathways.**

`max_length` stays **fixed at 256** by
[D-PREG1-008b](../spec/decisions.md#d-preg1-008b--max_length-fixed-at-256-not-selected-from-data).
This profile **confirms feasibility; it does not reopen the selection.** Had
coverage been chosen from these numbers, 128 would also have qualified — which
is exactly the data-dependent choice the fixed value exists to prevent.

Stripping marks lengthens the sequence. Stated precisely, because two different
statistics are easy to conflate here: the **per-example** Base-minus-Vanilla
delta has median **+3** and **p99 +14** pieces, while the **marginal** p99
sequence length separately shifts from **55** to **68** (**+13**). A quantile of
the per-example differences is not the difference of the two marginal quantiles.

A tokenizer trained on marked Vietnamese fragments the base form into more
pieces. **This is a tokenization observation and carries no downstream
performance claim** — it is a hypothesis about *why* the base-only pathway may
be burdened, not evidence that it is.

### UNK counts — unusable from this pass

This pass reported a single `unk_token_count = 4`. The accumulator ran across
**both** pathways, so the number is a Vanilla + Base-only sum with no
attribution: it is equally consistent with 4/0, 0/4 and every split between.
That is gap 2. **The value is not reinterpreted here**; the repaired code
reports the two pathways separately, and the rerun supplies them.

## What Attempt 1 does not contain

Scoped to the **first** pass above. Attempt 2 partially addresses the first two
items — see that section — but is **NON-FINAL**.

- unit-level tone and letter channel densities (gap 1) — Attempt 2 produced
  **real letter densities**; **resolved tone density is still missing**;
- pathway-attributed UNK counts (gap 2);
- any materialised 80/20 protocol split;
- any head, optimizer, checkpoint, LR selection or downstream score.

**All of this was subsequently delivered by Attempt r2**, which closed the
contract: inventory loaded, `eligibility_resolved = true`, real tone **and**
letter densities in one run, pathway-separated UNK counts, consistent v2 schema,
matching derived-train digest, zero duplicate/conflicting/cross-split groups and
zero overflow at 256.

[D-B3B0-002](../spec/decisions.md#d-b3b0-002--the-first-backbone-checkpoint-is-not-locked) remains **OPEN** — the tokenizer
revision is a probe revision. The compiled proposal PDF is **stale**.

---

## Attempt 2 — patched profiler, NON-FINAL

| | |
|---|---|
| **HEAD** | `f828ef1e892d9777b5a6bf69ca254d94756ca4fb` |
| **Run id** | `uit-vsfc-v1.0-profile-v2-analysis-v1-f828ef1e` |
| **Closure result** | **HOLD — for two independent reasons: B3A inventory absent (eligibility unresolved) *and* schema metadata inconsistent** |
| **Schema bug discovered** | `config.json` **v1** / `provenance.json` **v2** / `summary.json` top level **absent** |
| **Status** | **NON-FINAL. This is not the authoritative profile.** |

**This attempt is not authoritative and nothing in it should be cited as the
pre-G1 profile.** It is recorded because it produced two findings.

### Finding 1 — the fail-visible tone path worked

All three splits reported `eligibility_resolved = false`,
`tone_eligible_syllables = 0`, `tone_observed_syllables = 0` and
`observed_tone_unit_density = null`. **Correct behaviour, deliberately
designed**: the tone denominator needs the B3A syllable inventory, and without
it the profiler reports `null` rather than a tone density of `0.0`. A silent
`0.0` would have looked like a corpus finding while actually reporting an
unloaded resource.

**Letter-unit densities were produced and are defined** — that channel does not
depend on the inventory, which is the intended asymmetry. They are **still not
authoritative**, because the run did not close.

The inventory pin is **unchanged** (`vn-syllables-v1`, revision
`135a4d97…`, SHA-256 `78eeb840…`, 116 290 bytes, 17 974 / 17 954 / 2 489,
`NO_EXPLICIT_LICENSE`). Nothing was downloaded for this record; the committed
manifest was read only. The rerun uses the existing
`scripts/fetch_vietnamese_syllable_inventory.py` and then `--verify-only`.

### Finding 2 — artifact schema disagreed with itself

| Artifact | Declared |
|---|---|
| `config.json` | **`preg1-profile-v1`** (stale hard-coded literal) |
| `provenance.json` | `preg1-profile-v2` |
| `summary.json` (top level) | **absent** |
| `summary.provenance` | `preg1-profile-v2` |

Repaired: the profiler imports the single authoritative
`PROFILE_SCHEMA_VERSION`, and `summary.json` declares it at the top level.
**Artifact metadata only — no metric, denominator, density semantic,
tokenization, duplicate rule, exclusion or `max_length` was affected.**

**The `…-f828ef1e` run directory must not be overwritten.** The subsequent
authoritative attempt used a new run id,
`uit-vsfc-v1.0-profile-v2-analysis-v1-5e707ecd-r2` — the non-overwrite
discipline held.

---

## Attempt r2 — AUTHORITATIVE / PASS

**This is the authoritative pre-G1 dataset and profile result.** It supersedes
Attempts 1 and 2 wherever they overlap.

| | |
|---|---|
| **HEAD** | `5e707ecdabc7df378f535bed80e8dd0adb99861e` |
| **Run id** | `uit-vsfc-v1.0-profile-v2-analysis-v1-5e707ecd-r2` |
| **Profile schema** | `preg1-profile-v2` |
| **Dataset** | UIT-VSFC v1.0, **sentiment only** |
| **Evidence status** | **externally observed on Colab.** Not opened or recomputed locally |
| **Model weights / head / optimizer / training / downstream score** | **none** |

### Corpus identity

| | |
|---|---|
| Official train | 11 426 |
| Derived pre-G1 train | **11 424** |
| Derived train SHA-256 | `a20c0f7760f32dc48263a79d73ddf5363526c17e9de2afc32d8346b23444d301` — **exact match** to the value recorded before this run |
| Exclusion policy | `EXCLUDE_ENTIRE_CONFLICTING_CANONICAL_GROUP`, **no relabel**, official corpus unchanged |
| Tokenizer | `vinai/phobert-base` @ `01daacda68afe13d83023d16ec647239e344a1e6` |
| `max_length` | **256 FIXED** |

### B3A inventory — pinned and verified

| | |
|---|---|
| Schema | `vn-syllables-v1` |
| Source / author | `all-vietnamese-syllables.txt` / hieuthi |
| Revision | `135a4d9716e49a981624474156d6f247b9b46f6a` |
| SHA-256 | `78eeb840d50455b14bd564da5aed7318d96468b8deaad5986b77bf5c538315d2` |
| Size | 116 290 bytes |
| Verified shape | 17 974 raw / 17 954 unique canonical / 2 489 unique stripped |
| License | **`NO_EXPLICIT_LICENSE`** — unchanged, **not** upgraded |

The existing fetcher was used and the pin verified. **The raw inventory was not
copied into persistent evidence and is not committed.**

### Unit-level channel densities — resolved on all three splits

| Split | N | `eligibility_resolved` | tone obs / elig | tone density | letter obs / elig | letter density |
|---|---|---|---|---|---|---|
| derived train | 11 424 | **true** | 104 731 / 142 467 | **0.7351246253518359** | 85 253 / 502 706 | **0.16958818872263312** |
| validation | 1 583 | **true** | 13 826 / 18 794 | **0.7356603171224859** | 11 130 / 66 451 | **0.16749183608975035** |
| test | 3 166 | **true** | 29 100 / 39 197 | **0.742403755389443** | 23 315 / 138 857 | **0.16790655134418864** |

Example-level counters: `with_observed_tone` 11 396 / 1 576 / 3 155;
`with_observed_letter` 11 316 / 1 556 / 3 131.

**Not diacritization rates.** A tone density of 0.735 does **not** mean 73.5% of
sentences are correctly diacritized, and 0.170 is **not** a missing-diacritic
rate. Unmarked Vietnamese is observationally ambiguous — the profiler measures
what is present, never what is absent.

Safe reading: orthographic-channel exposure is measured at the precommitted unit
granularity, and the densities are **similar in scale across all three official
splits**.

### Descriptive fields — also authoritative

| Split | N | `base_equivalent` | `canon_changed` |
|---|---|---|---|
| derived train | 11 424 | **15** | **217** |
| validation | 1 583 | **4** | **29** |
| test | 3 166 | **4** | **41** |

**`base_equivalent` means "no observed mark under `b(canon(x))`" — not a
missing-diacritic rate.**

**Reconciliation with Attempt 1.** These descriptive fields **reproduce
Attempt 1 exactly**. What differs is narrower: train
`with_observed_tone`/`with_observed_letter` are each **2 lower** than Attempt 1
recorded at the same stated N, while validation and test are **identical**.
**Attempt r2 governs those two counters**; the descriptive fields simply agree.

The evidence does not say why. The −2 is arithmetically consistent with the two
excluded members carrying marks while being neither base-equivalent nor
canon-changed — **a reading consistent with the numbers, not an established fact
about those rows**.

### Integrity

`canonical_duplicate_groups` = **0**, `conflicting_label_group_count` = **0**,
`cross_split_group_count` = **0**.

Official **validation** remains measurement-dev. Official **test** remains
**SEALED** for downstream scores and protocol selection; profiling its
descriptive and integrity properties is not a test score and selected nothing.

### Pathway-attributed UNK

| | |
|---|---|
| `vanilla_unk_token_count` | **4** |
| `base_only_unk_token_count` | **0** |
| `total_unk_token_count` | **4** |
| `unk_count_is_per_pathway` | **true** |

The pre-repair total of `4` was **entirely Vanilla**.

**Narrow reading only:** the RAW_BASE token-length burden in this corpus is
**not explained by an increase in Base-only UNK tokens**. It does **not** mean
Base-only carries no tokenizer burden — Base-only sequences are measurably
longer. A tokenization observation, not a downstream result.

### Tokenizer reproduction

| | min | p25 | p50 | p75 | p90 | p95 | p99 | max | mean |
|---|---|---|---|---|---|---|---|---|---|
| Vanilla (n = 11 424) | 4 | 10 | 14 | 20 | 29 | 36 | 55 | 163 | 16.58044467787115 |
| Base-only (n = 11 424) | 4 | 12 | 16 | 24 | 35 | 45 | 68 | 202 | 20.11186974789916 |
| Base − Vanilla, per example | −1 | 2 | 3 | 5 | 7 | 9 | **14** | 39 | 3.5314250700280114 |

`length_changed_count` **10 642**, fraction **0.9315476190476191**. Every value
reproduces the earlier profile exactly.

Per-example delta p99 is **+14**; the marginal p99 length shifts **55 → 68**
(**+13**). Different statistics.

Coverage at 256: Vanilla **1.0**, Base-only **1.0**, joint **1.0**. Overflow:
**0.0** / **0.0**. **`max_length` stays fixed at 256; this does not reopen the
selection.**

### Schema consistency

`config` = `summary` = `summary.provenance` = `provenance` = report heading =
**`preg1-profile-v2`**. The Revision-2 schema defect is confirmed fixed on real
data.

### Artifact digests — externally observed

| Artifact | SHA-256 |
|---|---|
| `config.json` | `66bf8df59439791a77b25cb27619e753cbdc735e3894f15ceffbbcb0d06d9c53` |
| `provenance.json` | `0885c4257f1a8b36ccee085b8c7d8cfc47779fd8b3887d24e09a8985dd1baf8e` |
| `dataset_profile.json` | `123108be2296871ac3086544e1a2def76d76b851897f7b89bd1ffea07e2f9115` |
| `orthography_profile.json` | `95f8e12fb317f125a41c28e2fe2d7bbf69f47d30e8eb6531d91f511b09a26c28` |
| `duplicates.json` | `5bf1a94de7708a52610e3598d1172c458c6717f2b1029d5a41bbf8e5eb6a75be` |
| `token_length_profile.json` | `1ebf77afac2ab3797ff10358f83e27d0c52447c9adf99eeb200abc0cdbb15510` |
| `summary.json` | `301e10e1ef2e4e86302e37b31cdcac3df3cc173a0183bd768ef3b30c508ca36b` |
| `report.md` | `53cf096d44bbd4285b33b114d48c141c922aa4878098b2a447870246e05bd365` |
| `exclusion-manifest.json` | `7e9f6fe8d0a8be747f7890af7096f39d27ab14f3b5c70b31f62aba5048bfa0f1` |
| `inventory-verification.json` | `180737a230391c4f5d024f263f9e750aca12c1ec6b4fc459ed8c7ba43e398bd2` |

Persistent evidence destination:
`/content/drive/MyDrive/UNMARK/preg1-uit-vsfc-data-profile/uit-vsfc-v1.0-profile-v2-analysis-v1-5e707ecd-r2`

**No raw redistribution.** Raw UIT-VSFC was not copied, the derived CSV was not
copied, the raw inventory was not copied. These files are not in this repository
and were not hashed here; no digest was fabricated.

### What this closes, and what it does not

**Closed:** the pre-G1 dataset profile contract — integrity, orthographic
channel exposure at the precommitted granularity, tokenizer geometry, schema.

**Not closed, not even started:** any downstream measurement. **This profile
does not validate UNMARK.** Whether Base-only actually costs task performance,
and whether the adapter recovers it, is the job of the pre-G1
Vanilla-vs-Base-only diagnostic and then G1. **No head, no optimizer, no
training, no HPO, no downstream score.** The 80/20 split is **not materialised**.

[D-B3B0-002](../spec/decisions.md#d-b3b0-002--the-first-backbone-checkpoint-is-not-locked)
remains **OPEN**. The compiled proposal PDF remains **stale**.
