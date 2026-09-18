# Audit 076 - PhoNER Cross-Task Transfer Probe

Date: 2026-09-18

Task: implement the ViUnMark Cross-Task NER Transfer Probe for official
PhoNER_COVID19 word-level data, comparing native PhoBERT, ViUnMark-Gate and
ViUnMark-Scale with frozen encoders/pathways and one shared token-level NER
probe protocol.

## Starting Provenance

Recorded before implementation:

```text
git rev-parse HEAD
871ed6a11fdbeea20de6ca3f254c0343f5ee1827

git status --short
<clean>

git diff --stat
<empty>
```

The starting tree was clean. No commit or push was made.

## Dataset Contract

Implemented a PhoNER_COVID19 word-level contract in
`unmark/cross_task/phoner_transfer.py`.

The supported external files are:

- `train_word.conll` or `train_word.json`
- `dev_word.conll` or `dev_word.json`
- `test_word.conll` or `test_word.json`

The repository does not download, commit, or redistribute PhoNER data. The user
must provide `DATA_ROOT`.

Dataset identity records:

- `dataset_id = PhoNER_COVID19`
- `representation = word`
- split file identities
- source provenance
- expected schema for CoNLL and JSON
- TRAIN-only label inventory
- stable sample IDs from dataset id, representation, split, deterministic
  source ordinal and word-sequence hash
- SHA-256 values for TRAIN and DEV when local raw files are supplied
- TEST hash/count as `SEALED_UNREAD` before test stages

Stable sample IDs deliberately exclude labels so `test-predict` text-only IDs
match `test-score` gold IDs, but include the deterministic source ordinal so
duplicate sentences cannot collide.

Dataset terms are recorded in the identity layer: official PhoNER_COVID19 terms
permit research/educational use and prohibit redistribution of original or
modified dataset content. Version-controlled/public artifacts must not contain
raw or corrupted PhoNER text.

## Corruption Contract

The implementation reuses `unmark.corruption.corrupt` and the existing
`CorruptionProtocol` with the frozen scientific seed:

```text
PHONER_CORRUPTION_SEED = 19225
conditions = FULL, P25, P50, P75, P100, STRIP_ALL
```

Correction: the authoritative downstream ViUnMark path applies corruption once
to the whole sample:

```text
corrupt(text, condition, seed, sample_id)
```

PhoNER now follows that exact sample-level semantics. The word sequence is
joined with single spaces as a reversible word-level sentence representation,
corrupted once, then split back to the original word slots. Per-word independent
corruption is not used. Guards assert that corruption does not change word
count, whitespace slot identity, underscores, BIO labels, sample ID or split
membership. The same corrupted word sequence is shared by native PhoBERT,
ViUnMark-Gate and ViUnMark-Scale for a given
dataset/split/sample/condition/seed.

Focused parity test:

```text
test_corruption_parity_is_sample_level_not_per_word
```

## Token-Label Alignment

The pinned PhoBERT identity is imported from the existing Stage-I protocol:

```text
encoder_checkpoint = vinai/phobert-base
encoder_revision = 01daacda68afe13d83023d16ec647239e344a1e6
max_length = 256
```

The word-level PhoNER labels are the annotation authority. Alignment policy:

- first subtoken of a word receives the word BIO label
- remaining subtokens receive `ignore_index = -100`
- tokenizer special tokens receive `ignore_index = -100`
- padding receives `ignore_index = -100`
- condition-invariant word chunks are derived before training/evaluation
- each chunk boundary is shared across FULL/P25/P50/P75/P100/STRIP_ALL
- boundaries never split BIO entity continuations
- every chunk must fit the pinned max length under all six conditions

No existing authoritative PhoBERT NER implementation was found in the
repository. Existing downstream code is sentiment/readout-specific, and the
generic Stage-2 extraction module explicitly leaves downstream head training
open. Therefore this is a new cross-task probe protocol, not a historical
ViUnMark protocol.

The dataset-audit stage now reports, for TRAIN and DEV only:

- max subword length by condition
- overlength sample count by condition
- gold entities affected by naive condition-specific truncation
- final chunking/coverage policy
- proof that evaluated word/entity coverage is identical across conditions

TEST is not included in this audit until it is legitimately opened.

## Probe Architecture

All three pathways use the same task-specific probe:

```text
FrozenTokenProbe = Linear(hidden_dim, num_ner_labels)
hidden_dim = 768
hidden layers = 0
dropout = 0.0
loss = unweighted cross entropy with ignore_index=-100
```

Only the token classifier is trainable. The PhoBERT backbone and any Gate/Scale
Stage-I adapters are frozen and checked with `require_frozen_module` /
pathway-specific frozen loaders.

Exact frozen assets are hard-required before training:

```text
PhoBERT = vinai/phobert-base
PhoBERT revision = 01daacda68afe13d83023d16ec647239e344a1e6
ViUnMark-Gate sha256 = 6773fbb59c7381ba8ddaa944302124a124f5b8a5cb0a5dbb1a5063f3db4a2a91
ViUnMark-Scale sha256 = a32c0167817d457d5067c2a351f2d1b73b26229033f03727f43e2d79f59ef685
```

Mismatch hard-fails before training. Verified identities are written into probe
checkpoint manifests.

Negative controls recorded in config:

```text
sentiment_mlp_used = False
mm_cat_pooling_used = False
uit_vsfc_class_weights_used = False
sentiment_calibration_used = False
final_sentiment_fusion_heads = 0
```

## Training Policy

No generic frozen token-probe recipe existed. A new explicit cross-task protocol
was added and labeled as prospective:

```text
optimizer = AdamW
learning_rate = 1e-3
betas = (0.9, 0.999)
eps = 1e-8
weight_decay = 0.0
batch_size = 16
gradient_accumulation_steps = 1
max_optimizer_updates = 1000
eval_every_updates = 100
scheduler = none
warmup_updates = 0
gradient_clipping = none
AMP = disabled
dataloader shuffle = deterministic per-update reshuffle
shuffle seed = run_seed * 100000 + optimizer_update
epoch rollover = cycle until max_optimizer_updates
checkpoint tie break = DEV/FULL entity F1, lower validation loss if recorded, earlier update
selection = PhoNER DEV / FULL exact entity-level micro F1
final seeds = 53148, 59945, 42941, 720, 9428
smoke seed = 53148
```

Training data for the first experiment is restricted to PhoNER TRAIN/FULL.
Corrupted PhoNER examples are not used for training.

Before protocol freeze, only TRAIN/FULL and DEV/FULL may be used for smoke
validation, checkpoint selection and implementation debugging. Corrupted DEV
metrics may be generated only after the `freeze-protocol` artifact exists, and
no protocol, hyperparameter, model or selection change may follow from them.

## Metrics

Implemented `EntityF1` over exact entity span and entity type. Primary reported
values:

- entity-level micro F1
- entity precision
- entity recall

Condition summaries derive:

- FULL/P25/P50/P75/P100/STRIP_ALL F1
- Corrupt Avg F1
- All-6 F1
- FULL-to-STRIP absolute drop
- robustness retention per condition

Token accuracy is not used as the primary metric.

BIO/IOB2 compatibility policy: invalid predicted transitions are repaired as
common CoNLL/seqeval-compatible chunk extraction does. Sentence-initial `I-X`,
`O -> I-X`, and type-changing `B-Y/I-Y -> I-X` begin a new `X` entity. Exact
entity span and entity type are required.

Five-seed reporting is now frozen:

- primary evidence is mean entity-level micro F1 across the five independent
  runs, plus sample standard deviation
- per-seed precision/recall/F1 are retained
- no five-head logit ensembling is the primary result
- no best-seed selection
- Corrupt Avg F1 is the arithmetic mean of P25/P50/P75/P100/STRIP_ALL F1,
  computed per seed first, then summarized across seeds
- All-6 F1 is the arithmetic mean of all six condition F1 values, computed per
  seed first, then summarized across seeds

## TEST Guard

Implemented stage guards:

- `dataset-audit`
- `smoke-train`
- `train-dev`
- `dev-evaluate`
- `freeze-protocol`
- `test-predict`
- `test-score`

Script aliases required by the Colab interface are also supported:

- `--stage audit`
- `--stage smoke`
- `--stage train-dev`
- `--stage freeze`
- `--stage test-predict`
- `--stage test-score`

`test-predict` reads TEST words only and writes a sealed prediction artifact with
`contains_gold_labels = False` and `contains_original_or_corrupted_text = False`.
Sealed prediction artifacts contain IDs, word counts, predictions and metadata,
not original or corrupted PhoNER text. `test-score` refuses to run without such
an artifact and is the only stage allowed to read TEST gold labels.

Neither `test-predict` nor `test-score` was executed in this task.

## Colab Entry Point

Added:

```text
scripts/cross_task/run_phoner_transfer.py
```

Required external roots:

```text
--data-root
--asset-root
--output-root
```

Optional checkpoint overrides:

```text
--gate-checkpoint
--scale-checkpoint
```

No personal Drive path is embedded.

## Files Added

```text
unmark/cross_task/__init__.py
unmark/cross_task/phoner_transfer.py
scripts/cross_task/run_phoner_transfer.py
tests/test_phoner_cross_task_transfer.py
docs/audits/076-phoner-cross-task-transfer-probe.md
```

No UIT-VSFC result or protocol file was modified. No Stage-I checkpoint was
modified or retrained.

## Tests

Focused tests:

```text
pytest -q tests/test_phoner_cross_task_transfer.py
24 passed, 1 skipped in 0.89s
```

Repository cross-entropy guard after repair:

```text
pytest -q tests/test_viunmark_training.py::test_the_repository_has_no_other_cross_entropy_call_site
1 passed in 0.48s
```

Syntax check:

```text
python -m py_compile unmark/cross_task/phoner_transfer.py scripts/cross_task/run_phoner_transfer.py
passed
```

Full regression suite after repair:

```text
pytest -q
7 failed, 5080 passed, 274 skipped in 147.43s
```

The seven failures are all in `tests/test_stage1_parallel.py` and share the same
environment error:

```text
PermissionError: [Errno 1] Operation not permitted
```

The failure occurs when Python's multiprocessing forkserver attempts to bind a
Unix socket inside the sandbox. No PhoNER cross-task test failed in the full run.

## Unresolved Issues

- The real PhoNER dataset was not provided, so no dataset audit artifact with
  real SHA-256 values was generated.
- No smoke training run was executed; model/tokenizer/checkpoint loading is
  intended for Colab or another ML environment with external `DATA_ROOT` and
  `ASSET_ROOT`.
- Full regression completion is blocked in this sandbox by the Stage-1 parallel
  forkserver permission issue described above.

## Final Seal Status

```text
PHONER_TEST_READ=NO
PHONER_TEST_SCORING=NO
PHONER_TEST_TUNING=NO
STAGE1_RETRAINING=NO
UIT_VSFC_RETUNING=NO
```

## Final Git Status

At audit write time:

```text
?? scripts/cross_task/
?? tests/test_phoner_cross_task_transfer.py
?? unmark/cross_task/
?? docs/audits/076-phoner-cross-task-transfer-probe.md
```

`git diff --stat` reports no tracked-file diff because all implementation files
for this task are newly added and untracked. This is intentional; no commit was
created.
