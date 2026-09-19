# Audit 077: PhoNER Stage-II Post-Diagnostic Development Funnel

## Scope

This audit supersedes the first unrun Stage-II optimization scaffold and defines
the intended post-diagnostic development funnel:

```text
Audit 076 linear diagnostic
-> D1-NER
-> D2-NER
-> D3-NER
-> D4-NER
-> SYS1-NER
-> SYS2-1-NER
-> SYS2-2-NER
-> COMP-D1-NER / COMP-D2-NER
-> freeze-final
-> TEST later
```

No real training or DEV evaluation from this new optimized campaign has been
run. Existing Audit 076 diagnostic artifacts remain immutable historical inputs
and are not overwritten or retargeted.

All new outputs are namespaced under:

```text
phoner_stage2_optimized/
```

## Protocol Amendment

The original `umbrella_post_diagnostic_protocol.json` is preserved byte-for-byte
as the v2 parent artifact. Before any representation-bank build, a write-once
v3 amendment must be created:

```text
phoner_stage2_optimized/umbrella_post_diagnostic_protocol_v3_amendment.json
```

The v3 amendment supersedes v2 without mutating it. `protocol-amend` requires an
explicit external Audit-076 parent via:

```text
--audit076-frozen-protocol PATH
```

The parent is not inferred from the Audit-077 output root and there is no
fallback to `OUTPUT_ROOT/frozen_protocol.json`. The verified real parent is:

```text
/content/drive/MyDrive/UNMARK/UNMARK-BACKUP/phoner-transfer/326e87417146-official-v1/frozen_protocol.json
```

Expected parent identity:

```text
schema_version = phoner-cross-task-frozen-protocol-v1
sha256 = c64dd5dc5bd162582491b91b4c015dfe29279c76985edeebf778014e8bddee59
protocol_digest = ba0d07ba891aa01a89c4f80d26b6efc5e889e894e9e4a16c328a63e9f5ad3433
protocol_frozen = true
```

Audit-077 initially interpreted the Audit-076 TEST seal too narrowly by requiring
the parent `dataset_identity.split_files.test.path` field itself to be the
`SEALED_UNREAD` sentinel. That was incorrect. Audit-076 sealed TEST content,
hash, and count while retaining a historical external/local pathname as metadata.
The corrected v3 validator treats that pathname as inert parent metadata only:
it does not open, stat, resolve, hash, or otherwise dereference the TEST path.
The fail-closed seal checks are:

- `dataset_identity.split_files.test` exists
- `split = test`
- `gold_labels_read = false`
- `row_count = SEALED_UNREAD`
- `sha256 = SEALED_UNREAD`
- `test_read_before_freeze = false`
- `test_scored_before_freeze = false`
- parent schema, digest, and `protocol_frozen = true` remain exact

The v3 amendment records:

- v2 path, schema, and SHA-256 parent identity
- amendment reason
- `no_new_campaign_scientific_results_observed_between_v2_and_v3 = true`
- execution repository HEAD resolved from Git
- `clean_execution_tree = true`
- TRAIN and DEV local file SHA-256 values
- TRAIN and DEV row counts
- dataset source/revision provenance if available
- supplied external Audit-076 frozen protocol path, SHA-256, schema,
  `protocol_digest`, config digest, and producer/frozen identity where present
- safe Audit-076 TEST seal summary:
  `test_gold_labels_read`, `test_row_count`, `test_sha256`,
  `test_path_recorded_in_parent`, and
  `test_path_dereferenced_by_audit077 = false`
- all contracts below in machine-readable form

Future `build-bank` requires this v3 amended/final protocol, not v2. In this
contract audit, `build-bank` remains fail-closed until a later real bank
materializer writes actual bank files, SHA-256 values, stream digests, and
training schedule closure. No placeholder bank is allowed to unlock training.

## Post-Diagnostic Status

This is explicitly post-diagnostic development. Corrupted PhoNER DEV metrics
were already observed during Audit 076, so this campaign must not claim that its
final configuration was frozen before corrupted DEV evidence existed.

The diagnostic motivation is the clean-vs-robustness tradeoff observed in the
linear probe: Stage-I Gate/Scale provided corruption robustness evidence, while
the next development question is whether stronger Stage-II readouts and staged
fusion can recover Native clean NER strength without discarding robustness.

## Frozen Inputs

- PhoBERT: `vinai/phobert-base`
- PhoBERT revision: `01daacda68afe13d83023d16ec647239e344a1e6`
- ViUnMark-Gate Stage-I SHA-256:
  `6773fbb59c7381ba8ddaa944302124a124f5b8a5cb0a5dbb1a5063f3db4a2a91`
- ViUnMark-Scale Stage-I SHA-256:
  `a32c0167817d457d5067c2a351f2d1b73b26229033f03727f43e2d79f59ef685`

No Stage-I checkpoint is modified or retrained.

## Shared NER Mechanics

- Max length: `256`
- First-subtoken BIO supervision
- Continuation/special/pad labels: `-100`
- Metric: exact entity span + entity type micro-F1
- Existing IOB2 repair semantics retained
- Seeds: `53148`, `59945`, `42941`, `720`, `9428`
- AMP: off
- TF32: off
- Precision: float32
- No CRF
- No sentiment class bias or calibration
- No best-seed selection

Stage-II MLP:

```text
LayerNorm(768)
-> Linear(768, 256)
-> GELU
-> Dropout(0.1)
-> Linear(256, num_labels)
```

## D1-NER

D1 is a robust-readout preflight on `VIUNMARK_SCALE` only.

Predeclared candidates:

- `D1-A-NER`: MLP + FULL + unweighted CE
- `D1-B-NER`: MLP + AUG6 + unweighted CE
- `D1-C-NER`: MLP + AUG6 + sqrt-inverse-frequency weighted CE

Weighted CE formula:

```text
count_l = TRAIN first-subtoken label count for label l
raw_l = 1 / sqrt(count_l)
weight_l = raw_l / mean(raw over labels)
```

Every TRAIN-derived label must have positive count. The exponent is fixed at
`0.5` and is not tuned.

The historical Audit 076 linear probe is a reference only and is not retrained.

D1 selects the readout recipe with the recipe-selection rule below. If `D1-C`
does not survive that closed rule, later Native weighted candidates are not
admitted.

## D2-NER

D2 compares decoding policies on the selected D1 emission stream:

- token argmax plus current IOB2 repair
- deterministic hard BIO-constrained Viterbi decoding

The hard decoder has no learned transition parameters, creates no checkpoint,
uses the exact same token logits, and only forbids illegal BIO transitions such
as sentence-initial `I-X`, `O -> I-X`, and `B/I-Y -> I-X` where `X != Y`.

Pinned transition semantics:

- `I-X` cannot start a sequence.
- `O -> I-X` is forbidden.
- `B-X/I-X -> I-X` is allowed.
- `B-Y/I-Y -> I-X` is forbidden when `X != Y`.
- Transitions into `O` or any `B-X` are allowed.
- Path score is the sum of raw emission logits only.
- There are no learned or fixed transition bonus scores.
- No checkpoint or parameter is created.
- Any post-repair after constrained decoding must be a verified no-op because
  the constrained path is already valid BIO.

These semantics are materialized in the v3 protocol artifact as
`d2_hard_bio_contract`.

## D3-NER

D3 applies the D1-selected readout recipe and D2-selected decoding policy,
unchanged, to:

- `PHOBERT_NATIVE`
- `VIUNMARK_GATE`
- `VIUNMARK_SCALE`

This is the matched-head controlled representation comparison. Each pathway has
five heads, one per fixed seed. No best seed is selected.

## D4-NER

D4 performs no training. It uses D3 DEV outputs to compute:

- exact-entity correct sets
- Native-only, Gate-only, and Scale-only correct entities
- pairwise correctness overlap and disagreement
- boundary-error counts
- entity-type-error counts
- pathway logit/prediction disagreement where well-defined

Exact entity identity for D4 overlap/correctness analysis is:

```text
(sample_id, word_start, word_end, entity_type)
```

Including `sample_id` prevents cross-sentence entity collisions.

This key is materialized in the v3 protocol artifact as `d4_entity_contract`.

## SYS1-NER

SYS1 is adapted-only fusion:

```text
Gate_branch_raw  = arithmetic mean of exactly five Gate head logits
Scale_branch_raw = arithmetic mean of exactly five Scale head logits
```

Gate/Scale beta candidates:

- `0.75/0.25`
- `0.50/0.50`
- `0.25/0.75`

Raw emissions are fused before exactly one final selected decoder.

SYS1 beta weights are selected on the actual deployed adapted ensemble output,
not on five independent per-seed F1 values:

```text
Gate_branch_raw  = mean(raw logits from the five closed Gate heads)
Scale_branch_raw = mean(raw logits from the five closed Scale heads)
candidate_raw    = beta_gate * Gate_branch_raw + beta_scale * Scale_branch_raw
metrics          = one selected decoder(candidate_raw)
```

This is materialized separately from recipe selection in the v3 protocol artifact
as `fusion_selection_contract`.

## SYS2-1-NER

SYS2-1 develops the Native branch separately.

Minimum Native candidates:

- Native MLP FULL unweighted
- Native MLP AUG6 unweighted

A Native weighted candidate is admitted only if the closed D1 selection artifact
keeps the weighted readout alive under the five-seed aggregate rule. The final
Native branch is an arithmetic mean of exactly five selected Native head logits.

This admission policy is materialized in the v3 protocol artifact as
`sys2_1_admission_contract`.

## SYS2-2-NER

SYS2-2 fuses:

```text
adapted_raw = SYS1 raw emissions
native_raw  = five-head Native mean
```

Native/Adapted gamma candidates:

- `0.75/0.25`
- `0.50/0.50`
- `0.25/0.75`

Raw token emissions are fused first and exactly one selected structured decoder
is applied at the end. There is no parent decoder stacking.

SYS2-2 gamma weights are selected on the actual deployed Native/Adapted ensemble
output:

```text
adapted_raw   = closed SYS1 raw ensemble
native_raw    = mean(raw logits from the five closed Native heads)
candidate_raw = gamma_native * native_raw + gamma_adapted * adapted_raw
metrics       = one selected decoder(candidate_raw)
```

## COMP-D1 / COMP-D2

- `COMP-D1-NER`: matched-recipe complementarity analysis
- `COMP-D2-NER`: factorized gains from linear diagnostic through MLP, AUG6,
  structured decoding, SYS1 adapted fusion, Native addition, and SYS2-2

## Training Budget

The sentiment constant `72 updates/boundary x 30 = 2160` is not reused.

Every training candidate is trained for exactly five complete passes over its
own frozen training distribution.

With the audited official TRAIN count of `5027` and zero overlength chunking
from Audit 076, FULL-only candidates use:

```text
examples per pass = 5027
updates per complete pass = ceil(5027 / 16) = 315
max optimizer updates = 315 x 5 = 1575
evaluation boundaries = 30 unique quantile boundaries through update 1575
```

AUG6 candidates use:

```text
examples per pass = 5027 x 6 = 30162
updates per complete pass = ceil(30162 / 16) = 1886
max optimizer updates = 1886 x 5 = 9430
evaluation boundaries = 30 quantile boundaries through update 9430
```

For every candidate budget, boundaries are derived independently, are strictly
monotonic/unique, and the final boundary equals that candidate's max update.
If future legitimate TRAIN chunking changes the TRAIN chunk count, the same
rule derives both FULL and AUG6 budgets from the frozen chunk count before
training.

The emitted budget schema field is:

```text
updates_per_complete_distribution_pass
```

The misleading old field name `updates_per_complete_aug6_pass` is not emitted.

D1 training remains locked until build-bank produces an immutable
`training_schedule_closure.json` derived from the actual frozen TRAIN chunk
count. For the current authoritative 5027-chunk corpus, closure must verify the
expected `1575` and `9430` max updates and fail closed on drift.

## Representation Bank

Frozen encoders/pathways are banked before head training:

- splits: TRAIN and DEV only
- grid: pathway x condition
- pathways: Native, Gate, Scale
- conditions: `FULL`, `P25`, `P50`, `P75`, `P100`, `STRIP_ALL`
- dtype: float32
- no TEST

Every bank is bound to dataset SHA, sample IDs, pathway, Stage-I checkpoint SHA
when applicable, PhoBERT checkpoint/revision, tokenizer max length, corruption
condition/seed, and representation-bank schema. Any mismatch fails closed.

Head training consumes frozen representation banks; seed-specific shuffles and
dropout remain head-level randomness.

The v3 protocol artifact materializes `representation_bank_contract` with:

- splits: TRAIN and DEV only
- TEST excluded
- pathways: Native, Gate, Scale
- conditions: all six frozen conditions
- dtype: float32
- PhoBERT checkpoint/revision
- tokenizer max length
- corruption seed
- Gate/Scale Stage-I SHA-256 identities
- sample-ID/chunk provenance requirement
- dataset SHA requirement
- representation schema/version
- fail-closed mismatch policy

The eventual bank manifest must contain actual verified train/dev chunk counts,
per-pathway/condition representation identities, sample/chunk stream digests,
bank file SHA-256 values, and all provenance bindings above. TEST is
structurally unreachable.

## Mandatory Exact Head Reuse

Every Stage-II head has a scientific identity binding at least:

- pathway
- Stage-I/PhoBERT representation identity
- representation-bank identity
- head architecture
- training distribution
- loss definition and weights
- optimizer hyperparameters
- training budget and boundaries
- seed
- checkpoint-selection semantics

If a later stage requests the exact same identity as an already closed earlier
stage head, it must reuse that checkpoint byte-for-byte. Retraining an identical
scientific head is prohibited and fails closed.

Required reuse cases:

- D1-selected Scale heads are reused as D3 Scale heads when the recipe is
  identical.
- D3 Native heads are reused by SYS2-1 when the Native candidate is identical.
- SYS1 uses closed D3 Gate/Scale branch heads; no retraining.
- SYS2-2 uses closed SYS1 and SYS2-1 artifacts; no retraining.

This is materialized in the v3 protocol artifact as
`scientific_head_reuse_contract`, including the scientific head identity schema.

## Selection

Within one head/seed:

```text
All6 entity micro-F1
-> worst-condition F1
-> FULL F1
-> earlier boundary
```

Across recipes/systems:

```text
five-seed mean All6
-> five-seed mean worst-condition
-> five-seed mean FULL
-> simpler/fewer-branch system
```

Different seeds may not choose different model families and then be reported as
one fixed system.

Fusion-weight selection is separate from recipe selection. SYS1 beta and SYS2-2
gamma are selected by evaluating the actual deployed five-head ensemble outputs:

```text
ensemble All6 F1
-> ensemble worst-condition F1
-> ensemble FULL F1
-> simpler/default fusion
```

Five independent per-seed F1 averages are not used to select fusion weights.

## Compute Plan

Representation bank extraction:

```text
2 splits x 3 pathways x 6 conditions = 36 frozen encoder bank jobs
```

Maximum genuinely new head trainings after mandatory exact artifact reuse:

```text
D1 Scale preflight:
  3 recipes x 5 seeds = 15 new heads

D3 matched comparison:
  Native selected recipe x 5 seeds = 5 new heads
  Gate selected recipe x 5 seeds = 5 new heads
  Scale selected recipe x 5 seeds = 0 new heads if identical to the D1-selected Scale heads

SYS2-1 Native development:
  Native FULL unweighted x 5 seeds = 5 new heads if not identical to D3 Native
  Native AUG6 unweighted x 5 seeds = 5 new heads if not identical to D3 Native
  Native AUG6 weighted x 5 seeds = 5 new heads only if D1 weighted survives and not identical to D3 Native
```

Therefore the maximum new head trainings are:

```text
15 + 10 + 10 = 35
```

Exact mandatory reuse:

```text
D3 Scale:
  5 D1-selected Scale heads reused

SYS2-1 Native:
  5 D3 Native heads reused for the Native candidate whose recipe is identical
  to the D1-selected recipe

SYS1:
  5 D3 Gate heads reused
  5 D3 Scale heads reused

SYS2-2:
  closed SYS1 adapted branch reused
  closed SYS2-1 Native branch reused
```

Reused heads are byte-identical checkpoint reuse, not retraining.

Evaluation-only stages:

```text
D2, D4, SYS1, SYS2-2, COMP-D1, COMP-D2, freeze-final
```

## Implementation

Revised:

```text
unmark/cross_task/phoner_stage2_optimization.py
scripts/cross_task/run_phoner_stage2_optimization.py
tests/test_phoner_stage2_optimization.py
```

The runner exposes the staged namespace and write-once artifact chain. TEST
stages remain disabled.

## Verification

Focused tests added for:

- D1/D2/D3/D4 stage boundaries
- exact D1 candidate family
- deterministic weighted-CE formula
- BIO-constrained decoder
- no learned D2 parameters
- matched D3 recipes
- D4 no-training property
- exactly-five-head branch ensemble
- SYS1 beta grid
- SYS2-1 Native weighted-candidate rule
- SYS2-2 gamma grid
- fuse-before-single-decode semantics
- no sentiment bias/calibration
- no best-seed selection
- five-seed aggregate recipe/system selection
- deployed-ensemble fusion selection distinct from per-seed recipe selection
- representation-bank provenance and TEST exclusion
- derived training-budget math
- exact scientific head identity and mandatory reuse
- D4 entity key
- v2 immutable plus v3 amendment/supersession
- execution repository HEAD and clean-tree binding
- TRAIN/DEV SHA and row-count binding
- build-bank requiring v3 and failing closed without actual bank materialization
- schedule closure required before training
- budget field rename to `updates_per_complete_distribution_pass`
- old diagnostic artifacts never targeted
- TEST unreachable

Real training, real DEV evaluation, and TEST access were not run in this
implementation task.

Commands run:

```text
pytest -q tests/test_phoner_stage2_optimization.py tests/test_phoner_cross_task_transfer.py
```

Result:

```text
68 passed, 5 skipped
```

```text
pytest -q tests/test_viunmark_training.py::test_the_repository_has_no_other_cross_entropy_call_site
```

Result:

```text
1 passed
```

```text
python -m py_compile unmark/cross_task/phoner_stage2_optimization.py scripts/cross_task/run_phoner_stage2_optimization.py
```

Result:

```text
passed
```

```text
pytest -q
```

Result:

```text
7 failed, 5124 passed, 278 skipped
```

The 7 failures are the known Stage-I multiprocessing forkserver sandbox
`PermissionError: [Errno 1] Operation not permitted` cluster in
`tests/test_stage1_parallel.py`. No PhoNER Stage-II funnel failures were
observed.

## Seal

```text
PHONER_TEST_READ=NO
PHONER_TEST_SCORING=NO
PHONER_TEST_TUNING=NO
STAGE1_RETRAINING=NO
UIT_VSFC_RETUNING=NO
```
