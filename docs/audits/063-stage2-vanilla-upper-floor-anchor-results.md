# Audit 063 — Stage-2 Vanilla UPPER/FLOOR Anchor Results and Comparison Baseline

**Scope:** record the completed Vanilla `UPPER` / `FLOOR` Stage-2 anchor
results and bind them as the comparison baseline for later UNMARK variants.
**Date:** 2026-09-10
**Type:** **documentation-only provenance closeout.** No training, model load,
representation extraction, UIT-VSFC read, official validation read, official
TEST read, RESTORE run, ALIGN run, scientific artifact mutation, historical
audit edit, or provenance rewrite is performed by this audit.

---

## 1. Executive verdict

**VANILLA_UPPER_FLOOR=COMPLETE.**

The Vanilla PhoBERT anchor completed externally at:

```text
VANILLA_UPPER_FLOOR_SCIENTIFIC_EXECUTION_HEAD=a1aa9365ad6de1ebc85186dbff0f5ad3b3840b25
```

This is distinct from the corrected UNMARK scientific execution HEAD:

```text
CORRECTED_UNMARK_EXECUTION_HEAD=cb78114eb0fac3167fbc1f618d7e8267ab24b114
```

Corrected UNMARK-A/B artifacts remain bound to `cb78114...`. Vanilla
`UPPER` / `FLOOR` artifacts are bound to `a1aa9365...`. Audit 063 is written
after both experiments and does not rewrite the provenance of either one.

The completed Vanilla anchor provides:

```text
VANILLA_SCORE_UNITS=30_OF_30
HEADS_UNCHANGED_AFTER_MEASUREMENT=YES
SEPARATE_FLOOR_HEADS=NO
CORRUPTED_LABEL_TRAINING=NO
BEST_SEED_SELECTION=NO
CROSS_SEED_SELECTION=NO
OFFICIAL_VALIDATION_USED_FOR_SELECTION=NO
OFFICIAL_TEST_READ=NO
```

This audit also records the already-available descriptive comparison to the
author-selected corrected UNMARK-A result and the now-computable UNMARK-A GRR.
It does not start Candidate 1, Candidate 2, Candidate 3, RESTORE, or ALIGN.

---

## 2. Starting repository state

Verified before this audit file was written:

```text
branch : main
HEAD   : a1aa9365ad6de1ebc85186dbff0f5ad3b3840b25
status : clean (git status --short --branch produced only ## main...origin/main)
```

No Drive path was opened, no UIT-VSFC split was read, no model was loaded, no
scientific compute was run, and no existing file was modified.

---

## 3. Scientific definition

`UPPER` and `FLOOR` are **not** two independently trained systems. There is
exactly one Vanilla PhoBERT pathway:

```text
SystemPathway.VANILLA
```

Training:

1. use frozen `vinai/phobert-base`;
2. use revision `01daacda68afe13d83023d16ec647239e344a1e6`;
3. extract clean protocol-train representations;
4. extract clean protocol-dev representations;
5. train exactly five independent linear heads on clean protocol-train;
6. select one epoch inside each head on clean protocol-dev only;
7. freeze all five selected heads before official validation is read.

Measurement:

| Condition | Anchor role |
|---|---|
| `FULL` | `UPPER` |
| `P25` | `FLOOR` |
| `P50` | `FLOOR` |
| `P75` | `FLOOR` |
| `P100` | `FLOOR` |
| `STRIP_ALL` | `FLOOR` |

The same five frozen Vanilla heads are used for all six conditions.

Five seeds are scientific replicates:

```text
53148
59945
42941
720
9428
```

Head protocol:

```text
Linear(768, 3, bias=True)
optimizer = AdamW
learning rate = 0.01
batch size = 128
epochs = 30 complete epochs
early stopping = none
epoch selection =
    highest Macro-F1
    -> highest Accuracy
    -> earliest epoch
selection split = clean protocol-dev only
training split = clean protocol-train only
```

Boundary flags:

```text
SEPARATE_FLOOR_HEADS=NO
CORRUPTED_LABEL_TRAINING=NO
BEST_SEED_SELECTION=NO
CROSS_SEED_SELECTION=NO
OFFICIAL_VALIDATION_USED_FOR_SELECTION=NO
OFFICIAL_TEST_READ=NO
```

---

## 4. Clean Vanilla representation provenance

The clean representation evidence artifact was:

```text
/content/drive/MyDrive/UNMARK/UNMARK-BACKUP/stage2-vanilla-anchor/a1aa9365ad6d/audit062-upper-floor-v2/evidence/vanilla-clean-representations-v2.json
```

SHA256:

```text
d8f7731543890ae0b7d0cfa2d0be93ad8a92f233fa2fefefe50719825dd1e00b
```

The repository did **not** open or verify this Drive path during this audit. The
path and hash are recorded from the author's authoritative evidence statement.

### Protocol-train

| Field | Value |
|---|---|
| rows | `9139` |
| shape | `(9139, 768)` |
| dtype | `torch.float32` |
| pathway | `VANILLA` |
| ordered-ID digest | `2cad022dd4aabbc875e388030e453ba2479f1046bea884309398b79f7d878cbd` |
| label digest | `1c1377b2cd8c8016765229079fb469c2ead5c6ec85229d986f4344fa817f6128` |
| representation semantic SHA256 | `1d5ec8948c5e3d6ea4e8e7187c7fc86c94c810bf38f45885d4f250bdfa1f0725` |

### Protocol-dev

| Field | Value |
|---|---|
| rows | `2285` |
| shape | `(2285, 768)` |
| dtype | `torch.float32` |
| pathway | `VANILLA` |
| ordered-ID digest | `63192edf811f8249404597103dc5ba4f48bb7843d242a592d13764013c095860` |
| label digest | `546817741453edcb968c4d5de3530e435c5de9374995c1665260e92aea07ca1c` |
| representation semantic SHA256 | `29ac73af32cd38c0a924f29da8ef35d48a1e60292c2a91eb74b7fba7945953d9` |

Execution detail:

```text
PhoBERT frozen = YES
PhoBERT eval = YES
dtype = FP32
actual Vanilla extraction device = CPU
explicit CUDA move = NO
```

This is a CPU-native Vanilla extraction path. It must not be reinterpreted as a
GPU experiment merely because the Colab runtime contained an NVIDIA L4.

---

## 5. Five selected Vanilla heads

| seed | selected epoch | selected clean-dev Macro-F1 | selected clean-dev Accuracy | selected-state semantic SHA256 |
|---:|---:|---:|---:|---|
| 53148 | 11 | 0.752726 | 0.900656 | `2ebdd867b7a67b338ceb61b1243376c6fbe5d6e96be0efe5dfde5a602a7a5f33` |
| 59945 | 7 | 0.752517 | 0.898906 | `7370f17a783fc6242157433a26d94fdd2b4ac89367d851e2436f8f71e6486f91` |
| 42941 | 21 | 0.743697 | 0.897155 | `f88899ae7b441d00f64d09bb4b7345502a2d21cc4911f92193143bbfcab56d26` |
| 720 | 13 | 0.742956 | 0.904595 | `a7018cb8cf3d3350d4b378d728a63ea9739c57cb2a8456ed3d63f08d6be9f674` |
| 9428 | 8 | 0.751745 | 0.902845 | `6188a1f2566ada81f873a36143b0b266a028d54b2c2bb184ecb895f3bb75df03` |

These selected epochs are within-head checkpoint selections only. They are not
a best-seed selection and the five seeds remain scientific replicates.

Freeze evidence:

```text
/content/drive/MyDrive/UNMARK/UNMARK-BACKUP/stage2-vanilla-anchor/a1aa9365ad6d/audit062-upper-floor-v2/evidence/vanilla-five-head-freeze.json
```

SHA256:

```text
d6bfc4dc73476cc3b6497d409817966d5c07ad077e34df71d962f624c43e8f20
```

---

## 6. Official-validation identity

Official validation was first read only after all five Vanilla heads had been
frozen. This audit did not read official validation.

Exact identity:

| Field | Value |
|---|---|
| raw SHA256 | `9c475c8998871c0c7317ee200b3e7db827128cd2dfec9de5c689aca299acc8d0` |
| rows | `1583` |
| class counts | `{0: 705, 1: 73, 2: 805}` |
| ordered-ID digest | `825714672bcf27543825b85ac72c3fea2875647f55a117c41cc42ecc8e305d1a` |
| label digest | `70d5ed20e31dea2cacddc7cdb737ef3d0d1f980699b7e3c6d6aff0806711852b` |

Measurement corruption seed:

```text
19225
```

Conditions:

```text
FULL
P25
P50
P75
P100
STRIP_ALL
```

Changed-row counts:

| Condition | changed rows |
|---|---:|
| `FULL` | 0 |
| `P25` | 1268 |
| `P50` | 1513 |
| `P75` | 1564 |
| `P100` | 1576 |
| `STRIP_ALL` | 1579 |

These counts reproduce the same corruption realization used by corrected UNMARK
measurement v3.

---

## 7. Six Vanilla measurement representation identities

All measurement representations have:

```text
shape=(1583, 768)
dtype=torch.float32
```

| condition | semantic SHA256 |
|---|---|
| `FULL` | `b72696d581f39a4a4a3163e9b7bb03804f84aca3ac9970914f215da9c648394c` |
| `P25` | `42525c20807603d83bb92a4597f20c9ba4cd3d23df2f661fea5f393b270e4fcb` |
| `P50` | `baad79f7f05fea09eb510fe67534a75d48b4a7affa27aad63bbfe93e9c79a720` |
| `P75` | `11267297ba39cfc1434d2efb96bfa8a6ce8e87c18170e6ca24962a5419bca0b9` |
| `P100` | `b8fac05b0017f6e0694aee94b83c8aa441ef2d9f1a4c33afbb9760a6324efd70` |
| `STRIP_ALL` | `af53c279cb890ced46345ad51d83efb42a9b06981ec4dca93ebb2ab972a579c2` |

Measurement representation evidence:

```text
/content/drive/MyDrive/UNMARK/UNMARK-BACKUP/stage2-vanilla-anchor/a1aa9365ad6d/audit062-upper-floor-v2/evidence/vanilla-measurement-representations-v1.json
```

SHA256:

```text
93771277ee582898efbf9d29e8fe1553c0a429d8dfd359ffe41b2e2b1766c4d8
```

Notebook-only JSON key-normalization repair note:

```text
/content/drive/MyDrive/UNMARK/UNMARK-BACKUP/stage2-vanilla-anchor/a1aa9365ad6d/audit062-upper-floor-v2/evidence/cell4-json-roundtrip-repair-v1.json
```

SHA256:

```text
c184a0b9a4fb7713f6ca7792c4df251c11738b5d9c05f8f1ff2b4cf6b2f349a5
```

The scientific extraction had already completed successfully; all six caches had
passed second-pass cache verification. The failure was only Python-object
equality after JSON integer keys were converted to strings. Primary evidence was
not overwritten, and representations were not re-extracted. This was not a
scientific failure.

---

## 8. Exact per-seed Vanilla official-validation results

| seed | condition | role | Macro-F1 | Accuracy |
|---:|---|---|---:|---:|
| 53148 | FULL | UPPER | 0.753856 | 0.902085 |
| 53148 | P25 | FLOOR | 0.709616 | 0.862287 |
| 53148 | P50 | FLOOR | 0.686064 | 0.822489 |
| 53148 | P75 | FLOOR | 0.625494 | 0.760581 |
| 53148 | P100 | FLOOR | 0.556716 | 0.683512 |
| 53148 | STRIP_ALL | FLOOR | 0.374859 | 0.461150 |
| 59945 | FULL | UPPER | 0.761585 | 0.903348 |
| 59945 | P25 | FLOOR | 0.716105 | 0.857865 |
| 59945 | P50 | FLOOR | 0.675522 | 0.813013 |
| 59945 | P75 | FLOOR | 0.606065 | 0.741630 |
| 59945 | P100 | FLOOR | 0.519617 | 0.668351 |
| 59945 | STRIP_ALL | FLOOR | 0.369543 | 0.493367 |
| 42941 | FULL | UPPER | 0.741928 | 0.898926 |
| 42941 | P25 | FLOOR | 0.692870 | 0.853443 |
| 42941 | P50 | FLOOR | 0.654677 | 0.808591 |
| 42941 | P75 | FLOOR | 0.590983 | 0.737208 |
| 42941 | P100 | FLOOR | 0.523234 | 0.641188 |
| 42941 | STRIP_ALL | FLOOR | 0.332465 | 0.364498 |
| 720 | FULL | UPPER | 0.739316 | 0.905243 |
| 720 | P25 | FLOOR | 0.696932 | 0.866077 |
| 720 | P50 | FLOOR | 0.657817 | 0.824384 |
| 720 | P75 | FLOOR | 0.609634 | 0.771320 |
| 720 | P100 | FLOOR | 0.540660 | 0.690461 |
| 720 | STRIP_ALL | FLOOR | 0.372029 | 0.427669 |
| 9428 | FULL | UPPER | 0.731325 | 0.897031 |
| 9428 | P25 | FLOOR | 0.698890 | 0.861655 |
| 9428 | P50 | FLOOR | 0.645056 | 0.813645 |
| 9428 | P75 | FLOOR | 0.613025 | 0.760581 |
| 9428 | P100 | FLOOR | 0.542098 | 0.682881 |
| 9428 | STRIP_ALL | FLOOR | 0.396035 | 0.471889 |

```text
VANILLA_SCORE_UNITS=30_OF_30
HEADS_UNCHANGED_AFTER_MEASUREMENT=YES
```

---

## 9. Primary Vanilla aggregate table

Mean ± sample standard deviation over the five seeds:

| Condition | Anchor role | Macro-F1 | Accuracy |
|---|---|---:|---:|
| FULL | UPPER | `0.745602 ± 0.012046` | `0.901327 ± 0.003325` |
| P25 | FLOOR | `0.702882 ± 0.009641` | `0.860265 ± 0.004799` |
| P50 | FLOOR | `0.663827 ± 0.016609` | `0.816425 ± 0.006724` |
| P75 | FLOOR | `0.609040 ± 0.012469` | `0.754264 ± 0.014329` |
| P100 | FLOOR | `0.536465 ± 0.015152` | `0.673279 ± 0.019661` |
| STRIP_ALL | FLOOR | `0.368986 ± 0.022964` | `0.443714 ± 0.050239` |

Descriptive equal-weight average over the five degraded conditions:

```text
Vanilla FLOOR degraded equal-weight mean Macro-F1 = 0.576240
Vanilla FLOOR degraded equal-weight mean Accuracy = 0.709589
```

This equal-weight summary is descriptive. The condition-specific results remain
primary.

---

## 10. Final Vanilla evidence

Canonical machine-readable Vanilla `UPPER` / `FLOOR` closeout artifact:

```text
/content/drive/MyDrive/UNMARK/UNMARK-BACKUP/stage2-vanilla-anchor/a1aa9365ad6d/audit062-upper-floor-v2/evidence/vanilla-upper-floor-final-v1.json
```

Raw SHA256:

```text
d4d6914b8cbe440698a125c6e7299f42431beb707e4c7c54e110ee6ccfa8a862
```

---

## 11. Contextual comparison already available: corrected UNMARK-A

This is not a new experiment performed by Audit 063. Corrected UNMARK-A comes
from:

```text
SCIENTIFIC_EXECUTION_HEAD=cb78114eb0fac3167fbc1f618d7e8267ab24b114
```

Evidence:

```text
/content/drive/MyDrive/UNMARK/UNMARK-BACKUP/stage2-measurement/cb78114eb0fa/audit061-8fa991e5/corrected-official-validation-v3/stage2-corrected-measurement-v3-final.json
```

SHA256:

```text
8d9fbd4396f88334b12606e0194123e7009e902f8e7b3e4bd1b94f35b77edff2
```

UNMARK-A aggregate scores:

| Condition | Macro-F1 | Accuracy |
|---|---:|---:|
| FULL | `0.667074 ± 0.017841` | `0.812508 ± 0.005312` |
| P25 | `0.644674 ± 0.011589` | `0.797599 ± 0.001874` |
| P50 | `0.641626 ± 0.007792` | `0.788882 ± 0.009005` |
| P75 | `0.624020 ± 0.006570` | `0.763108 ± 0.014461` |
| P100 | `0.618434 ± 0.013859` | `0.756665 ± 0.015656` |
| STRIP_ALL | `0.610777 ± 0.010004` | `0.751358 ± 0.013330` |

UNMARK-A degraded equal-weight means:

```text
Macro-F1 = 0.627906
Accuracy = 0.771522
```

Compact comparison:

| Condition | Vanilla F1 | UNMARK-A F1 | Δ F1 A−Vanilla | Vanilla Acc | UNMARK-A Acc | Δ Acc A−Vanilla |
|---|---:|---:|---:|---:|---:|---:|
| FULL | 0.745602 | 0.667074 | -0.078528 | 0.901327 | 0.812508 | -0.088819 |
| P25 | 0.702882 | 0.644674 | -0.058209 | 0.860265 | 0.797599 | -0.062666 |
| P50 | 0.663827 | 0.641626 | -0.022202 | 0.816425 | 0.788882 | -0.027543 |
| P75 | 0.609040 | 0.624020 | +0.014980 | 0.754264 | 0.763108 | +0.008844 |
| P100 | 0.536465 | 0.618434 | +0.081969 | 0.673279 | 0.756665 | +0.083386 |
| STRIP_ALL | 0.368986 | 0.610777 | +0.241791 | 0.443714 | 0.751358 | +0.307644 |

Descriptive interpretation only:

* Vanilla is substantially better on `FULL`.
* Vanilla remains better at `P25` and `P50`.
* The crossover occurs between `P50` and `P75` in this measurement grid.
* UNMARK-A is slightly better at `P75`.
* UNMARK-A gains substantially at `P100`.
* UNMARK-A is dramatically more robust at `STRIP_ALL`.
* The current UNMARK-A exhibits a clean/light-corruption cost in exchange for
  strong robustness under severe information loss.

No significance test is performed or claimed.

---

## 12. GRR result already closed

Frozen formula:

```text
GRR = (S_system - S_FLOOR) / (S_UPPER - S_FLOOR)
```

Rules:

```text
zero denominator -> undefined
no epsilon
no clipping
```

Current corrected UNMARK-A GRR:

| Condition | GRR Macro-F1 | GRR Accuracy |
|---|---:|---:|
| P25 | -1.362579 | -1.526154 |
| P50 | -0.271498 | -0.324405 |
| P75 | +0.109692 | +0.060137 |
| P100 | +0.391939 | +0.365651 |
| STRIP_ALL | +0.642010 | +0.672281 |

Equal-weight degraded score headline:

```text
Macro-F1:
UPPER FULL          = 0.745602
FLOOR degraded mean = 0.576240
UNMARK-A mean       = 0.627906
GRR                 = 0.305062

Accuracy:
UPPER FULL          = 0.901327
FLOOR degraded mean = 0.709589
UNMARK-A mean       = 0.771522
GRR                 = 0.323010
```

The headline is computed by averaging degraded scores first and applying GRR
once. It is not the arithmetic mean of the five condition-level GRR ratios.

GRR evidence:

```text
/content/drive/MyDrive/UNMARK/UNMARK-BACKUP/stage2-vanilla-anchor/a1aa9365ad6d/audit062-upper-floor-v2/evidence/vanilla-upper-floor-unmark-a-grr-final-v1.json
```

SHA256:

```text
3088f743260bee92d6aaa5bc6ad79cc327b5a0fef4ccda0e0235de36ecb614a0
```

---

## 13. Metric interpretation for future paper tables

Both downstream metrics must continue to be reported:

```text
Primary downstream metric   = Macro-F1
Secondary downstream metric = Accuracy
```

Official validation is imbalanced:

```text
class 0 = 705
class 1 = 73
class 2 = 805
```

Macro-F1 gives equal weight to each class and is more informative than Accuracy
alone for the primary downstream comparison. Accuracy must not be discarded; it
remains a secondary metric and currently tells the same qualitative robustness
story.

---

## 14. Future experiment comparison registry

Candidate descriptions below are working research directions, not frozen
protocols yet. No candidate acquires a result, protocol, hyperparameter,
execution HEAD, evidence SHA, or scientific claim in this audit.

| System | Status | Main modification | FULL F1 | P25 F1 | P50 F1 | P75 F1 | P100 F1 | STRIP_ALL F1 | Degraded mean F1 | Degraded GRR F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Vanilla | COMPLETE | Frozen PhoBERT baseline | 0.745602 | 0.702882 | 0.663827 | 0.609040 | 0.536465 | 0.368986 | 0.576240 | N/A anchor |
| UNMARK-A | COMPLETE | Current author-selected input adapter | 0.667074 | 0.644674 | 0.641626 | 0.624020 | 0.618434 | 0.610777 | 0.627906 | 0.305062 |
| Candidate 1 | NOT RUN | Clean-preservation / identity regularization | — | — | — | — | — | — | — | — |
| Candidate 2 | NOT RUN | Residual severity-gated UNMARK | — | — | — | — | — | — | — | — |
| Candidate 3 | NOT RUN | Identity preservation + severity gate | — | — | — | — | — | — | — | — |

Analogous Accuracy registry:

| System | Status | FULL Acc | P25 Acc | P50 Acc | P75 Acc | P100 Acc | STRIP_ALL Acc | Degraded mean Acc | Degraded GRR Acc |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Vanilla | COMPLETE | 0.901327 | 0.860265 | 0.816425 | 0.754264 | 0.673279 | 0.443714 | 0.709589 | N/A anchor |
| UNMARK-A | COMPLETE | 0.812508 | 0.797599 | 0.788882 | 0.763108 | 0.756665 | 0.751358 | 0.771522 | 0.323010 |
| Candidate 1 | NOT RUN | — | — | — | — | — | — | — | — |
| Candidate 2 | NOT RUN | — | — | — | — | — | — | — | — |
| Candidate 3 | NOT RUN | — | — | — | — | — | — | — | — |

---

## 15. Future scientific boundary

The observed result motivates the following hypothesis:

> Current UNMARK-A appears to intervene too strongly when the input is clean or
> only lightly corrupted. Future work should investigate whether
> clean-performance preservation can be improved without sacrificing the
> severe-corruption robustness observed at P100 and STRIP_ALL.

This is a hypothesis generated after reviewing official-validation results.
Therefore:

```text
OFFICIAL_VALIDATION_HAS_ALREADY_BEEN_SEEN=YES
```

It must not be used to repeatedly tune and select Candidate 1/2/3 while still
pretending official validation is an untouched confirmatory set. Future
candidate protocols must define their model, training and hyperparameter
selection boundary before claiming confirmatory results.

Audit 063 does not invent that future protocol.

---

## 16. Local verification for this documentation closeout

This task is documentation-only. Local verification must not access Drive, load
Hugging Face, open UIT-VSFC, run a scientific notebook, train, extract
representations, or read official validation or TEST.

Checks:

```text
git diff --check
git diff --no-index --check /dev/null docs/audits/063-stage2-vanilla-upper-floor-anchor-results.md
python -m json.tool docs/spec/stage2-dual-finalist-protocol.json /tmp/stage2-dual-finalist-protocol.json.audit063.formatted
git status --short
```

Results:

| Check | Result |
|---|---|
| `git diff --check` | **PASS** |
| `git diff --no-index --check /dev/null docs/audits/063-stage2-vanilla-upper-floor-anchor-results.md` | **PASS** - no whitespace errors printed. Exit status is nonzero because `/dev/null` differs from the added file. |
| `python -m json.tool docs/spec/stage2-dual-finalist-protocol.json /tmp/stage2-dual-finalist-protocol.json.audit063.formatted` | **PASS** |
| `git status --short` | **PASS** - only `?? docs/audits/063-stage2-vanilla-upper-floor-anchor-results.md` was shown. |

---

## 17. Final state

```text
AUDIT063_VANILLA_RESULT_RECORD=PASS

VANILLA_UPPER_FLOOR=COMPLETE
VANILLA_EXECUTION_HEAD=a1aa9365ad6de1ebc85186dbff0f5ad3b3840b25
VANILLA_SCORE_UNITS=30_OF_30

VANILLA_UPPER_FULL_MACRO_F1=0.745602
VANILLA_UPPER_FULL_ACCURACY=0.901327

VANILLA_FLOOR_DEGRADED_MEAN_MACRO_F1=0.576240
VANILLA_FLOOR_DEGRADED_MEAN_ACCURACY=0.709589

CORRECTED_UNMARK_A_BOUND=YES
CORRECTED_UNMARK_EXECUTION_HEAD=cb78114eb0fac3167fbc1f618d7e8267ab24b114

UNMARK_A_DEGRADED_MEAN_MACRO_F1=0.627906
UNMARK_A_DEGRADED_MEAN_ACCURACY=0.771522

UNMARK_A_HEADLINE_MACRO_F1_GRR=0.305062
UNMARK_A_HEADLINE_ACCURACY_GRR=0.323010

VANILLA_FINAL_EVIDENCE_SHA256=d4d6914b8cbe440698a125c6e7299f42431beb707e4c7c54e110ee6ccfa8a862
GRR_FINAL_EVIDENCE_SHA256=3088f743260bee92d6aaa5bc6ad79cc327b5a0fef4ccda0e0235de36ecb614a0

BEST_SEED_SELECTION=NO
SEPARATE_FLOOR_HEADS=NO
OFFICIAL_VALIDATION_USED_FOR_SELECTION=NO
OFFICIAL_VALIDATION_HAS_ALREADY_BEEN_SEEN=YES
OFFICIAL_TEST_READ=NO

CANDIDATE_1_STATUS=NOT_RUN
CANDIDATE_2_STATUS=NOT_RUN
CANDIDATE_3_STATUS=NOT_RUN

RESTORE_EXECUTED=NO
ALIGN_EXECUTED=NO

SCIENTIFIC_COMPUTE_PERFORMED_BY_THIS_AUDIT=NO
UIT_VSFC_READ_BY_THIS_AUDIT=NO
OFFICIAL_VALIDATION_READ_BY_THIS_AUDIT=NO
MODEL_LOADED_BY_THIS_AUDIT=NO
TRAINING_PERFORMED_BY_THIS_AUDIT=NO
REPRESENTATION_EXTRACTION_PERFORMED_BY_THIS_AUDIT=NO
SCIENTIFIC_ARTIFACT_MODIFIED_BY_THIS_AUDIT=NO
DOCS_ONLY=YES
```

**VANILLA UPPER/FLOOR ANCHOR RECORDED. CORRECTED UNMARK-A GRR IS NOW BOUND FOR
FUTURE PAPER TABLES. OFFICIAL TEST REMAINS SEALED.**
