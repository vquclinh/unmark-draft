# Audit 064: Stage-2 RESTORE Baseline Implementation

## Scope

This audit implements the separate RESTORE baseline executor for the UNMARK
paper evaluation. This Codex implementation task does not run scientific
scoring, does not generate RESTORE results itself, and does not modify
historical Audit 063 evidence. The later real e22 scientific measurement is
recorded below as immutable historical provenance.

RESTORE tests the string-level repair strategy:

```text
source CSV text
  -> unmark.orthography.canon(text) neutral Stage-2 preprocessing
  -> condition corruption if degraded
  -> frozen external diacritic-restoration model
  -> restored text
  -> frozen PhoBERT-base
  -> RESTORE-specific linear classification head
  -> UIT-VSFC sentiment prediction
```

## Implementation Base

```text
RESTORE_IMPLEMENTATION_BASE_HEAD=966cbfa280f7576c5418f850c4e641a1fb8a85d8
RESTORE_CANONICALIZATION_REPAIR_BASE_HEAD=3affe91f6cde5186d6efa4ae8e584767b8792af6
```

The starting repository state was checked with:

```bash
git rev-parse HEAD
git status --short
git log -5 --oneline
```

The working tree was clean before RESTORE implementation began.

## External RESTORE Model

```text
MODEL_ID=nrl-ai/vn-diacritic-vit5-base
REVISION=7ec0710193721ac3321b3bb3741ec92d8b43cad3
LICENSE=apache-2.0
ARCHITECTURE=T5ForConditionalGeneration
PARAMETER_COUNT=225950976
```

The revision was verified through Hugging Face Hub metadata for the exact model
ID and immutable SHA. The implementation fails closed if the revision is `main`,
`latest`, `master`, short, or otherwise non-immutable.

The executor uses only:

```text
AutoTokenizer
AutoModelForSeq2SeqLM
```

RESTORE does not use package pipelines, rule fallbacks, LLM fallbacks, spell
correction, ensembles, ONNX quantization, or alternate checkpoints.

Generation is frozen as:

```text
max_length=256
do_sample=False
num_beams=1
temperature=None
top_k=None
top_p=None
```

Tokenizer input/decode behavior is explicit and recorded in restored-text cache
identity:

```text
AutoTokenizer.from_pretrained(use_fast=True, legacy=True)
tokenizer(..., return_tensors="pt", padding=True, truncation=True, max_length=256,
          add_special_tokens=True, return_attention_mask=True)
batch_decode(..., skip_special_tokens=True, clean_up_tokenization_spaces=False)
transformers==4.57.6
```

The restorer is required to run in eval mode with all model parameters frozen.

## Source-Code Isolation

RESTORE code lives under:

```text
unmark/baselines/restore/
scripts/baselines/restore_stage2.py
scripts/baselines/restore_e22_postprocess.py
tests/baselines/restore/
```

RESTORE does not import or execute:

```text
unmark/modeling/adapter.py
unmark/stage1/
```

It reuses only neutral project infrastructure where appropriate:

```text
PREG1 split identities
PhoBERT representation extraction semantics
head training and metrics helpers
corruption implementation
hash and provenance helpers
```

## Artifact Isolation

All runtime artifacts are placed under the RESTORE-only namespace:

```text
/content/drive/MyDrive/UNMARK/UNMARK-BACKUP/
  stage2-baselines/
    restore/
      <execution-head-prefix>/
        audit064-restore-v1/
```

The executor never writes to historical namespaces:

```text
stage2-measurement/
stage2-training/
stage2-vanilla-anchor/
```

Previously closed Vanilla and corrected UNMARK evidence are authenticated
read-only for GRR and optional contextual comparison.

## Stage-2 Fairness Protocol

RESTORE uses the frozen PREG1 TRAIN source and protocol-train/dev membership.
The clean head-training pathway is:

```text
clean protocol-train source text
  -> unmark.orthography.canon(text)
  -> RESTORE
  -> PhoBERT
  -> RESTORE clean train representations
  -> RESTORE head
```

The clean protocol-dev pathway is analogous and controls only within-head epoch
selection. Five independent RESTORE heads are trained with seeds:

```text
53148
59945
42941
720
9428
```

Each head must run exactly 30 complete epochs with:

```text
Linear(768, 3, bias=True)
AdamW
LR=0.01
batch=128
early_stopping=NO
```

## Validation Boundary

The official validation set is opened only after all five RESTORE heads have
durable closeouts: each expected seed directory must contain a valid head
artifact and selected-state file whose SHA-256 matches the artifact. The
implementation records truthfully:

```text
OFFICIAL_VALIDATION_PREVIOUSLY_SEEN_BY_AUTHORS=YES
```

RESTORE validation scores cannot select a model, restorer revision, decoding
setting, preprocessing rule, head hyperparameter, or seed.

There is no official TEST path, role, CLI argument, hidden branch, or production
mock closeout path.

## Corruption and RESTORE Pathway

RESTORE Stage-2 shares the same neutral canonical-clean input semantics as the
Vanilla and corrected UNMARK Stage-2 paths:

```text
source CSV text -> unmark.orthography.canon(text)
```

This preprocessing is not RESTORE itself; it is the common Stage-2 spelling
anchor before any intervention. FULL observes canonical clean text. Degraded
conditions apply the authoritative `unmark.corruption.corrupt` implementation
to that canonical clean text. Changed-row counts compare observed pre-RESTORE
text to canonical clean text, not to raw CSV spelling.

The degraded corruption seed is frozen:

```text
corruption_seed=19225
```

FULL is the clean condition and carries no corruption seed in scientific/cache
identity:

```text
FULL corruption_seed=None
```

The six validation conditions are:

```text
FULL
P25
P50
P75
P100
STRIP_ALL
```

Expected changed-row counts are:

```text
FULL=0
P25=1268
P50=1513
P75=1564
P100=1576
STRIP_ALL=1579
```

Every condition, including FULL, passes through the same RESTORE pathway:

```text
CONDITION_AWARE_ROUTING=NO
FULL_BYPASS=NO
ONE_RESTORE_PATHWAY=YES
```

Special-casing FULL at condition-stream construction only means "no corruption";
it does not bypass the frozen RESTORE model.

## Failed 3affe91 Execution

The real execution at repository head
`3affe91f6cde5186d6efa4ae8e584767b8792af6` under
`stage2-baselines/restore/3affe91f6cde/audit064-restore-v1/` exposed a
condition-stream canonicalization bug before validation-restored caches were
written. No scientific RESTORE result exists from that failed execution.

The failed run counted raw-vs-canonical spelling differences as corruption
damage and observed:

```text
FULL=29
P25=1272
P50=1513
P75=1564
P100=1576
STRIP_ALL=1579
```

The frozen expected changed-row counts were not altered:

```text
FULL=0
P25=1268
P50=1513
P75=1564
P100=1576
STRIP_ALL=1579
```

That namespace is historical failed provenance only and must not be reused by a
patched execution.

## Scoring

Five frozen heads are evaluated across six conditions:

```text
5 seeds x 6 conditions = 30 score units
```

Each score unit records Macro-F1, Accuracy, and per-class F1. Aggregates use
mean and sample standard deviation across the five heads. Macro-F1 is the
primary downstream metric.

## GRR

Previously closed Vanilla anchor evidence is authenticated at:

```text
/content/drive/MyDrive/UNMARK/UNMARK-BACKUP/stage2-vanilla-anchor/a1aa9365ad6d/audit062-upper-floor-v2/evidence/vanilla-upper-floor-final-v1.json
SHA256=d4d6914b8cbe440698a125c6e7299f42431beb707e4c7c54e110ee6ccfa8a862
```

GRR is:

```text
GRR = (S_RESTORE - S_FLOOR) / (S_UPPER - S_FLOOR)
```

The denominator is not adjusted:

```text
zero denominator -> UNDEFINED
no epsilon
no clipping
```

Headline degraded GRR averages the five degraded RESTORE scores first, averages
the five degraded Vanilla FLOOR scores first, then applies the ratio once.
Condition GRRs are not averaged into the headline.

## E22 Post-Measurement Integration Repair

The real execution at repository head
`e22c5ea8dbcbea4060193dca3f40fc97694fbb4a` successfully completed the
expensive/scientific phases through 30-unit RESTORE measurement and diagnostics.
The durable measurement artifact is:

```text
measurement/restore-measurement.json
SHA256=c175d492834d5bb32e9423fd8f14fbd65874f1c22906f014dbd6727a0ba7b38f
score_units=30
```

`RESTORE_GRR` then exposed a post-measurement integration bug: the reader did
not support the already-authenticated closed Vanilla evidence schema, where
condition means live at:

```text
aggregate[CONDITION].macro_f1.mean
aggregate[CONDITION].accuracy.mean
```

This is not a RESTORE scientific measurement failure. The measurement,
diagnostics, heads, representations, restored text caches, corruption
realisation, tokenizer/generation settings, and canonicalization remain
unchanged. The repair adapts the read-only evidence parser to the historical
schema; it does not rewrite or normalize the Vanilla JSON.

The provenance-preserving recovery is an explicit post-processing closeout that
authenticates and binds:

```text
source_execution_head=e22c5ea8dbcbea4060193dca3f40fc97694fbb4a
restore_measurement_sha256=c175d492834d5bb32e9423fd8f14fbd65874f1c22906f014dbd6727a0ba7b38f
vanilla_evidence_sha256=d4d6914b8cbe440698a125c6e7299f42431beb707e4c7c54e110ee6ccfa8a862
unmark_a_evidence_sha256=8d9fbd4396f88334b12606e0194123e7009e902f8e7b3e4bd1b94f35b77edff2
```

The repair-head closeout must not claim to have generated the e22 model,
representation, head, text-cache, measurement, or diagnostic artifacts, and it
must not silently reuse artifacts across heads.

The dedicated post-processing executor writes only derived JSON under:

```text
stage2-baselines/restore-postprocess/<repair-head-prefix>/audit064-e22-closeout-v1/
```

It must not write under:

```text
stage2-baselines/restore/e22c5ea8dbcb/audit064-restore-v1/
```

Allowed post-processing outputs are GRR JSON, contextual comparison JSON, and a
post-processing closeout evidence JSON. No model, head, representation,
restored-text, or measurement directories are produced by this recovery path.

## Diagnostics

Restoration diagnostics are descriptive only and cannot influence any model,
decoding, seed, or epoch decision. They report, per condition:

```text
sentence exact-match rate to clean gold after NFC normalization
character error rate to clean gold after NFC normalization
word error rate to clean gold after NFC normalization
fraction of outputs differing from observed input
fraction of outputs whose stripped/base form differs from observed input
```

For FULL they also report:

```text
fraction_restore_leaves_clean_sentence_unchanged
fraction_restore_changes_clean_sentence
```

## TEST Sealing

RESTORE has no TEST enum role, no TEST path, no `--test-*` CLI argument, and no
method that can open official TEST data.

The final runtime must end with:

```text
OFFICIAL_TEST_READ=NO
HARD_STOP=YES
```

## Local Verification

This audit adds static and synthetic tests only. No real RESTORE model weights,
PhoBERT weights, Drive data, UIT-VSFC data, official validation rows, or
official TEST rows were loaded during this Codex implementation/post-processing
repair task. This local verification statement is scoped to Codex work in the
repository and does not negate the durable e22 scientific measurement recorded
above.

The required local verification commands are recorded in the final Codex
response for this task.

## Closeout Flags

```text
AUDIT064_RESTORE_IMPLEMENTATION=PASS

RESTORE_EXECUTOR_IMPLEMENTED=YES
RESTORE_POSTPROCESS_EXECUTOR_IMPLEMENTED=YES
RESTORE_REAL_RESULTS_GENERATED_BY_CODEX_IMPLEMENTATION_TASK=NO
RESTORE_E22_REAL_SCIENTIFIC_MEASUREMENT_EXISTS=YES
RESTORE_E22_MEASUREMENT_SHA256=c175d492834d5bb32e9423fd8f14fbd65874f1c22906f014dbd6727a0ba7b38f

RESTORE_MODEL_DOWNLOADED_BY_THIS_AUDIT=NO
PHOBERT_LOADED_BY_THIS_AUDIT=NO
UIT_VSFC_READ_BY_THIS_AUDIT=NO
OFFICIAL_VALIDATION_READ_BY_THIS_AUDIT=NO
OFFICIAL_TEST_READ=NO

COLAB_RUNNER_READY=YES
```
