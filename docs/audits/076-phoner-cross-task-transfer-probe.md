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

## Frozen-Head Producer/Execution Provenance Repair Addendum

This addendum records a provenance-gate repair at execution HEAD
`c843d2874066bacebc275286e79927d12c6f6232`.

Observed failure:

- `dev-evaluate` refused an existing immutable `frozen_probe_heads.json` because
  the artifact's `repository_head` differed from the current checkout.
- That field is now interpreted as the producer HEAD of the frozen-head
  manifest, not a requirement that every later execution-only evaluator run at
  the same HEAD.

Scientific status:

- `frozen_probe_heads.json` was produced under an earlier repaired
  implementation HEAD.
- Fan-out `dev-evaluate` runs under a later execution-only HEAD.
- The 15 selected probe checkpoint SHA-256 identities are unchanged.
- No frozen head manifest was rewritten, regenerated, deleted or replaced.
- No `frozen_protocol.json` rewrite occurred.
- No re-freeze occurred after partial corrupted DEV metric exposure.
- No scientific model, protocol, hyperparameter, seed, corruption semantics,
  condition order, label handling, metric definition, checkpoint-selection rule
  or Stage-I checkpoint changed.

Repair:

- `frozen_probe_heads.json.repository_head` is preserved as
  `producer_repository_head` in `dev_evaluate_results.json`.
- The current clean checkout is recorded separately as
  `execution_repository_head`.
- The evaluator still fails closed on all scientific identities: the exact 15
  pathway/seed entries, selected-head checkpoint SHA-256 values, frozen protocol
  digest/static coherence, Stage-I Gate/Scale checkpoint SHA-256 values,
  PhoBERT checkpoint/revision, current scientific config constants and sealed
  TEST state.
- A dirty execution tree fails closed before expensive work.

Performance/preflight repair:

- Cheap gates now run before tokenizer loading and TRAIN/DEV parsing: result
  overwrite check, clean Git HEAD, frozen protocol schema/static digest, Stage-I
  checkpoint byte SHA-256, frozen-head manifest schema and selected-head SHA-256.
- `dev-evaluate` no longer recomputes TRAIN condition-invariant coverage. TRAIN
  is used only to verify local file SHA/row count and TRAIN-derived label
  inventory against the frozen protocol artifact.
- DEV file SHA/row count are still verified before DEV measurement, and DEV
  condition-invariant chunks are materialized with the frozen tokenizer and
  corruption semantics.

Actual real corrupted DEV evaluation status in this repair task:

- `NOT RUN`.

Post-repair local verification:

```text
pytest -q tests/test_phoner_cross_task_transfer.py
41 passed, 3 skipped in 0.94s

pytest -q tests/test_viunmark_training.py::test_the_repository_has_no_other_cross_entropy_call_site
1 passed in 0.44s

python -m py_compile unmark/cross_task/phoner_transfer.py scripts/cross_task/run_phoner_transfer.py
passed

pytest -q
7 failed, 5097 passed, 276 skipped in 140.29s
```

The seven full-suite failures are the known Stage-1 multiprocessing forkserver
sandbox failures in `tests/test_stage1_parallel.py`
(`PermissionError: [Errno 1] Operation not permitted`). No PhoNER cross-task
test failed.

```text
PHONER_TEST_READ=NO
PHONER_TEST_SCORING=NO
PHONER_TEST_TUNING=NO
STAGE1_RETRAINING=NO
UIT_VSFC_RETUNING=NO
```

## Dev-Evaluate Fan-Out Optimization Addendum

This addendum records an execution-only optimization after the repaired
`dev-evaluate` path was started and interrupted for performance.

Runtime observation:

- The first repaired `dev-evaluate` implementation was load-only, but it still
  ran the full encoder separately for every `(pathway, seed, condition)`.
- That created 90 full DEV encoder passes.
- The run was interrupted because execution was prohibitively slow.
- Partial corrupted DEV metrics were exposed during runtime inspection: `YES`.
- No scientific model, protocol, hyperparameter, seed, corruption semantics,
  condition order, label handling, metric definition, checkpoint-selection rule
  or checkpoint state was changed because of those partial metrics.

Optimization:

- For a fixed pathway, condition and DEV batch, encoder/adaptor hidden states are
  identical across the five frozen seed heads.
- `dev-evaluate` now streams each DEV batch through the encoder once for each
  `(pathway, condition)` and fans the hidden tensor out to all five frozen token
  probes.
- Expensive encoder work is reduced from 90 logical full DEV passes to 18
  pathway-condition passes.
- The output still contains exactly 90 logical metric records in the same
  pathway/seed/condition order, with the same EntityF1 implementation and the
  same summary calculations.
- No hidden-state cache is persisted; the optimization is streaming only.

Equivalence evidence:

- Focused tests compare the old reference seed loop against the fan-out path on
  deterministic synthetic fixtures.
- They assert exact equality of entity TP/FP/FN, precision, recall, micro-F1,
  all 90 record identities/order, Corrupt Avg, All-6, FULL-to-STRIP drop and
  retention summaries.
- The same fixture asserts hidden-state calls drop from 90 to 18.
- Side-effect tests confirm selected head checkpoints, `train_dev_results.json`
  and `frozen_protocol.json` remain untouched, TEST is not read, and optimizer,
  backward and checkpoint-save paths are not reachable from `dev-evaluate`.

Actual real corrupted DEV evaluation status in this task:

- `NOT RUN`.

Post-optimization local verification:

```text
pytest -q tests/test_phoner_cross_task_transfer.py
37 passed, 3 skipped in 0.93s

pytest -q tests/test_viunmark_training.py::test_the_repository_has_no_other_cross_entropy_call_site
1 passed in 0.46s

python -m py_compile unmark/cross_task/phoner_transfer.py scripts/cross_task/run_phoner_transfer.py
passed

pytest -q
7 failed, 5093 passed, 276 skipped in 140.95s
```

The seven full-suite failures are the known Stage-1 multiprocessing forkserver
sandbox failures in `tests/test_stage1_parallel.py`
(`PermissionError: [Errno 1] Operation not permitted`). No PhoNER cross-task
test failed.

```text
PHONER_TEST_READ=NO
PHONER_TEST_SCORING=NO
PHONER_TEST_TUNING=NO
STAGE1_RETRAINING=NO
UIT_VSFC_RETUNING=NO
```

## Dev-Evaluate Dispatch Repair Addendum

This addendum records an implementation/provenance repair at HEAD
`326e874171464970293bd08929debffc6c6f5ee1`.

Root cause:

- `scripts/cross_task/run_phoner_transfer.py` dispatched both `train-dev` and
  `dev-evaluate` to `stage_train(args, config, smoke=False)`.
- Therefore `--stage dev-evaluate` constructed `AdamW`, called `backward()` and
  `optimizer.step()`, evaluated only DEV/FULL for checkpoint selection, and
  could overwrite the selected `.pt` probe checkpoints.

Observed post-freeze incident:

- An accidental post-freeze `dev-evaluate` execution occurred through the faulty
  dispatch.
- Corrupted DEV metrics exposed: `NO`.
- PhoNER TEST read/scoring/tuning: `NO`.
- Stage-I retraining: `NO`.
- UIT-VSFC retuning: `NO`.
- A read-only pre-run inspection had captured all 15 selected probe checkpoint
  SHA-256 values. Every current checkpoint remains byte-identical to that
  inspection, so the scientific selected-checkpoint state is recovered exactly.

Recovered selected probe head identities:

```text
PHOBERT_NATIVE_seed-42941_best.pt  bbdb765a885afcb0855dcaa25d4444b91ab3fbf1ee8e14127bce4c696e4208fd
PHOBERT_NATIVE_seed-53148_best.pt  971606d6f0f82e04c2a9892a1bb1320577739223bec303edfad44f178730a228
PHOBERT_NATIVE_seed-59945_best.pt  ca485ad85c66058923cf1ffaaed9af676dc09d13cdd4c79cce97d165e8c3c30b
PHOBERT_NATIVE_seed-720_best.pt    6ad51575fd04ec31e1bdd291d2d363be23748bc20cca91a3d2575f91c825508b
PHOBERT_NATIVE_seed-9428_best.pt   bd7c688bccca42f618ef9b9955b35b03496f82af13dc92547920ea6ac8f0d640
VIUNMARK_GATE_seed-42941_best.pt   a8e8fcffa5b2be02dc794296df31aedaf2c3bbd9a6f709edf1ef89d0063f330d
VIUNMARK_GATE_seed-53148_best.pt   273c5a16e970084ba10056f648135ec36966ebeed1bc62e464a6e6835c3e7efd
VIUNMARK_GATE_seed-59945_best.pt   395ce4514e741c57f20b21d9e37f1e4500eb0d238b46152bb67668d44fd8d978
VIUNMARK_GATE_seed-720_best.pt     ddaf6ed4df283434255227024060b9be61407eb93dc594f0b68d7ec41424085b
VIUNMARK_GATE_seed-9428_best.pt    48c2fd27101434671dc960925287dc29b89fd38459fa3b8c0f2a099d2d2af982
VIUNMARK_SCALE_seed-42941_best.pt  78370a41cbbe014482e69efecd236a0bb2174732581c442b8427a0ed67fe5377
VIUNMARK_SCALE_seed-53148_best.pt  982ae3d41d8d96849cce33ec782b78f196f8fb26e497bd9f8f58c26941a967d2
VIUNMARK_SCALE_seed-59945_best.pt  97e5872c6e1b7760665f4705074e8c32cedf73fdcdffbe60981acc354d4dbd03
VIUNMARK_SCALE_seed-720_best.pt    caa3c9575fc45feb914effb6af478d3e2c2cc691cf919b50da7fb7d5c40bd65d
VIUNMARK_SCALE_seed-9428_best.pt   a44e45d1e328200421b3ac79f34481bb29326a7939cfc7997d4c2a8752b826c0
```

Repair:

- `train-dev` and `dev-evaluate` now dispatch separately.
- `dev-evaluate` is load-only: it requires `frozen_protocol.json`, requires a
  write-once `frozen_probe_heads.json`, verifies all 15 checkpoint SHA-256
  identities before evaluation, reads TRAIN/DEV only, and writes the separate
  write-once `dev_evaluate_results.json`.
- The selected-head freeze artifact is an addendum artifact; the already
  existing `frozen_protocol.json` is not overwritten.
- The repair does not alter any scientific protocol constant, hyperparameter,
  model architecture, seed, corruption semantics, checkpoint-selection rule,
  metric definition, or Stage-I checkpoint.

Actual corrupted DEV evaluation status:

- `NOT RUN` in this repair task.

Post-repair local verification:

```text
pytest -q tests/test_phoner_cross_task_transfer.py
37 passed, 2 skipped in 0.94s

pytest -q tests/test_viunmark_training.py::test_the_repository_has_no_other_cross_entropy_call_site
1 passed in 0.51s

python -m py_compile unmark/cross_task/phoner_transfer.py scripts/cross_task/run_phoner_transfer.py
passed

pytest -q
7 failed, 5093 passed, 275 skipped in 147.03s
```

The seven full-suite failures are the known Stage-1 multiprocessing forkserver
sandbox failures in `tests/test_stage1_parallel.py`
(`PermissionError: [Errno 1] Operation not permitted`). No PhoNER cross-task
test failed.

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

## Real-Data Smoke Hardening Addendum

This addendum updates Audit 076 in place after the real-data dataset audit and
3-pathway 1-seed smoke run at HEAD
`b7961b1cd3c1621989adbd8109464a41933cd0de`.
It supersedes the earlier local-only unresolved note that no real-data audit or
smoke run had been executed.

Real-data audit facts:

- Official TRAIN rows: `5027`.
- Official DEV rows: `2000`.
- TEST remained `SEALED_UNREAD`.
- Coverage audit passed for TRAIN/DEV only:
  - zero overlength samples in all six conditions;
  - zero gold entities affected by naive truncation;
  - identical evaluated word/entity coverage across conditions.

Smoke status:

- The smoke run used exactly 2 optimizer updates and is implementation
  validation only, not scientific evidence.
- Native PhoBERT, ViUnMark-Gate and ViUnMark-Scale each produced a best
  checkpoint.
- No PhoNER TEST prediction, scoring or tuning was run.

Runtime integration bugs found and repaired:

- The runner now creates the output root's `checkpoints/` directory with
  `parents=True, exist_ok=True` before checkpoint writes, so a fresh output root
  is sufficient.
- Gate and Scale pathway loads now keep the runtime syllable inventory separate
  from the Stage-I provenance identity. The runtime inventory is used for
  linguistic eligibility/classification; the Stage-I checkpoint verifiers
  receive `verify_scientific_inputs().inventory`, whose pinned identity remains:
  revision `135a4d9716e49a981624474156d6f247b9b46f6a`, sha256
  `78eeb840d50455b14bd564da5aed7318d96468b8deaad5986b77bf5c538315d2`.

No protocol, model-selection rule or scientific/training hyperparameter changed.
No raw or corrupted PhoNER text was added to repository artifacts.

Post-hardening verification:

```text
pytest -q tests/test_phoner_cross_task_transfer.py
29 passed, 1 skipped in 0.93s

pytest -q tests/test_viunmark_training.py::test_the_repository_has_no_other_cross_entropy_call_site
1 passed in 0.49s

python -m py_compile unmark/cross_task/phoner_transfer.py scripts/cross_task/run_phoner_transfer.py
passed

pytest -q
7 failed, 5085 passed, 274 skipped in 145.56s
```

The seven full-suite failures are the known Stage-1 multiprocessing forkserver
sandbox `PermissionError: [Errno 1] Operation not permitted` failures in
`tests/test_stage1_parallel.py`. No PhoNER cross-task test failed.

```text
PHONER_TEST_READ=NO
PHONER_TEST_SCORING=NO
PHONER_TEST_TUNING=NO
STAGE1_RETRAINING=NO
UIT_VSFC_RETUNING=NO
```
