# Audit 073 - Final ViUnMark Public-Facing System Recovery

**Scope:** Phase P0. Bring the final post-hoc ViUnMark system back into the
repository, where it had never been defined. Implement it as a clean, additive,
public-facing layer under paper naming, and record everything in
machine-readable specs.

**Amended (evidence amendment, same uncommitted audit):** frozen Drive evidence
inspected by the author closes almost every Stage-II retraining field this audit
had first left unresolved. The one remaining training field, cross-entropy
reduction, was traced through the repository (§10.4) and remains unresolved.

**Final review correction (same uncommitted audit):** the first amendment
under-claimed two parts of the frozen evidence. The native PhoBERT protocol
confirms much more of the PhoBERT Readout training policy (§10.5), and the
semantics of the historical OPT1 and OPT2 stages are now recovered (§5, §13.2,
§13.3). Status wording now separates method/protocol recovery from bit-exact
historical execution.

**Date:** 2026-09-17
**Type:** recovery, formalisation and additive implementation. **Nothing was
selected, trained, tuned, scored or recalibrated.** No historical module, spec,
audit, checkpoint, head or evidence artifact was modified.

---

## 1. Executive verdict

**FINAL_VIUNMARK_SYSTEM=RECOVERED_AND_FORMALISED. PUBLIC_LAYER=IMPLEMENTED.**

```text
P0                               = RESOLVED
FINAL_INFERENCE_DEFINITION       = RECOVERED
ADAPTED_READOUT_TRAINING_POLICY  = RECOVERED_AT_PROTOCOL_LEVEL   (Gate and Scale readout heads, §10.1)
PHOBERT_READOUT_TRAINING_POLICY  = SUBSTANTIALLY_RECOVERED        (native protocol, §10.5)
HISTORICAL_BIT_EXACT_RETRAINING  = NOT_FULLY_RECOVERED            (§10.3, §16.2)
PHOBERT_BIT_EXACT_RETRAINING     = NOT_RECOVERED
CROSS_ENTROPY_REDUCTION          = UNRESOLVED                     (robust-MLP runner not in the repository, §10.4)
STAGE2_DEVELOPMENT_STAGE_SEMANTICS = RECOVERED                    (readout/augmentation screen, class-balance stage, §13.3)
SCIENTIFIC_CORRUPTION_SEED       = 19225                          (UIT-VSFC reproduction value, not a method default)
CROSS_PATHWAY_INPUT_IDENTITY     = REQUIRED
WEIGHTED_CE_RULE                 = RECOVERED

Adapted readout policy components (protocol level):
MLP_INITIALIZATION           = RECOVERED
OPTIMIZER_POLICY             = RECOVERED
AUGMENTED_TRAIN_CONSTRUCTION = RECOVERED
DETERMINISTIC_BATCH_STREAM   = RECOVERED
DROPOUT_SEED_POLICY          = RECOVERED
CHECKPOINT_SELECTION         = RECOVERED

Method/protocol recovery is not bit-exact historical execution recovery. P0 is
resolved because its purpose is to make the final ViUnMark method and its
provenance explicit in the repository.

FIVE_HEADS_PER_BRANCH=YES
WITHIN_BRANCH_LOGIT_MEAN=YES
BEST_SEED_SELECTION=NO
FINAL_HEAD_COUNT=20
FINAL_SYSTEM_IS_20_HEAD_ENSEMBLE=YES
PARENT_BIASES_STACKED=NO

NEW_MODEL_SELECTION=NO
NEW_HYPERPARAMETER_SEARCH=NO
NEW_TRAINING=NO
VALIDATION_READ=NO
TEST_READ=NO
TEST_TUNING=NO
TEST_RECALIBRATION=NO
POST_TEST_RETUNING=NO
```

The system was **recovered, not newly selected.** Every structural fact, weight,
training value and calibration value was fixed in frozen evidence before this
work. None was chosen here.

| | |
|---|---|
| Starting branch / HEAD | `main` / `93227e134b31b850763d9c1dd33ccc9abf3c9ae7` (unchanged) |
| Historical tracked files modified | **0** (`git diff --stat` empty) |
| Files in the P0 set | **27**: 16 package modules, 2 specs, 8 test files, 1 audit |
| Audit 072 | preserved byte-for-byte (`6f0266d7…c7281`), still untracked |
| ViUnMark tests | **218 passed, 14 skipped** (torch-gated), 0 failed |
| Historical regression subset | **1060 passed, 69 skipped**, 0 failed |
| Full suite | **5063 passed, 273 skipped, 0 failed** |

---

## 2. Repository state

At the start of P0:

```text
branch              : main
HEAD                : 93227e134b31b850763d9c1dd33ccc9abf3c9ae7
git status --short  : ?? docs/audits/072-sa-vlsp2016-external-evaluation-repository-reconnaissance.md
git diff --stat     : (empty)
Audit 072 sha256    : 6f0266d7bcd74602ba81d11292be416e3dcd165b701658d264070a5ce8fc7281 (58 466 bytes)
```

At the start of the evidence amendment, the HEAD was the same and no tracked
diff existed. The untracked files were Audit 072 plus the P0 set (this audit, the
two specs, seven test files, `unmark/viunmark/`). This audit's own digest at that
point was `b915508a1e855cee0dad91d41a76029909699f64ccf0c0357f706ecda8f809a3`. It
was re-checked before the rewrite and was unchanged.

No commit, push, pull, fetch, merge, rebase, branch operation, reset, checkout,
restore or clean was run at any point. Audit 072 was never opened for writing.

---

## 3. Motivation: Audit 072

Audit 072 found that none of the following appeared in any tracked file,
reachable commit or untracked file:

```text
SYS1  SYS2-2  R4-W  R6-U  R6-W  ViUnMark  Scale-U  Scale-W  logit fusion
```

It also found that the committed dual-finalist Stage-2 protocol forbids
ensembles. Blocker **P0** was that the method had to exist in the repository
before an SA-VLSP2016 campaign could retrain it. This audit resolves P0.

---

## 4. Evidence and where each fact comes from

This work did **not** open any Drive artifact. The facts fall into three classes.

### 4.1 Corroborated by the repository itself

| Fact | Repository evidence |
|---|---|
| ViUnMark-Gate checkpoint `6773fbb5…2a91`, selected update **3500**, source HEAD `7773c77b…` | `unmark/stage1/finalists.py:244` (`FINALIST_A`); cross-checked by test |
| ViUnMark-Scale checkpoint `a32c0167…f685`, selected update **8000**, source HEAD `8de83f0d…` | `unmark/evaluation/stage2_scf_pathway.py` (`V2_SCF_CHECKPOINT`); cross-checked by test |
| Fusion equations and ids | `unmark/modeling/adapter.py:96,136,308`; `modeling/contracts.py:256,267,288` |
| Adapter shape, gate init, 3 551 232 parameters | `modeling/contracts.py:64,193,301-304`; `stage1/protocol.py:37,39` |
| Stage-I objective, pooling, UVW-2026 corpus | `stage1/protocol.py:45-46,110-138`; `stage1/objective.py` |
| Six corruption conditions | `unmark/corruption/conditions.py` |
| Scientific corruption seed **19225** | `unmark/evaluation/stage2_head_campaign.py:176` (`STAGE2_MEASUREMENT_CORRUPTION_SEED`) holds the same value; cross-checked by test |
| Inventory `78eeb840…315d2` @ `135a4d97…` | `unmark/stage1/finalists.py` (`INVENTORY_SHA256`, `INVENTORY_SOURCE_REVISION`); cross-checked by test |
| Representation sources | `stage2_dual_finalist.stage2_first_token_representation` (`:952`), `collate_stage2_unmark_batch` (`:628`), `modeling/pooling.masked_mean_non_special` (`:41`) |
| Protocol-train counts `4259 / 366 / 4514` (9 139 rows) | Audits 023, 052; `tests/test_preg1_split.py:261` |
| Weight vector `[0.5573523044586182, 1.9012669324874878, 0.5413808226585388]` | equals the float32 evaluation of the rule on those counts **exactly** (test) |
| Head parameters 199 171 / 397 315; total 6 955 580 | derived from the architecture (test) |
| Decision-geometry quantity names | `stage1/protocol.py:193-196`; Audit 068 §1 |

### 4.2 Supplied by the author from frozen authoritative artifacts

These facts come from the author's inspection of the frozen artifacts, not from
the repository:

* 20 head digests and selected boundaries;
* artifact lineage and protocol lineage digests;
* fusion weights and calibration values;
* the within-branch head mean;
* the MLP architecture and the Stage-II training recipe (§10);
* the corruption identity, including `base_grid_invariance` and
  `cross_arm_input_identity`;
* the representation-bank facts;
* system-selection facts.

The specs record them verbatim, and tests pin each one to a literal.

### 4.3 Searched for across every commit and not found

`git grep` over `git rev-list --all` finds **0 hits** in any commit for each of:

* `GELU`, `nn.GELU`, `AUGMENTED`, `6COND`;
* `SQRT_INV_FREQ`, `s2-opt`, `representation_bank`;
* `cycle_seed`, `cycle_index`, `dropout_seed`, `updates_per_boundary`,
  `max_optimizer_updates`;
* `199171`, `397315`, `6955580`, `CAT-U`, `R4-W`, `OPT3`, `SYS1`.

The runner that trained the robust MLP heads is **not in the repository.**

### 4.4 Corrections applied during P0

1. **Five heads per branch, averaged.** The author corrected an earlier
   description of the seeds as independent system replicates. When the
   correction arrived, only the two specs existed. They were regenerated, and
   all code and tests implement within-branch head-logit means only.
2. **Training facts scoped.** The first version of the reproduction record
   listed batch 128, LR 0.01, 2 160 updates, 30 boundaries and float32 in an
   unscoped block. That implied they also applied to the PhoBERT Readout heads.
   The evidence scopes them to the Gate/Scale readout optimisation, so the
   amendment moves them into `method.robust_mlp_training_recipe`, restricted to
   the three adapted-pathway branches (§10.5).
3. **Under-claims corrected (final review).**
   * The first amendment recorded the PhoBERT Readout heads as "partially
     specified", with initialisation, optimizer, budget and checkpoint selection
     unconfirmed. The frozen native protocol `s2-sys2-1-protocol-v1.json`
     (`e2b6dcff…8cab`) directly confirms those; §10.5 now records exactly what it
     confirms, and exactly what it does not establish.
   * OPT1 and OPT2 were recorded as "semantics not recovered". Their frozen
     protocols define them; they are now recorded as the Stage-II Readout and
     Augmentation Screen and the Stage-II Class-Balance Loss Optimization (§13.3).

---

## 5. Naming: historical label → public name

The authoritative mapping is `docs/spec/viunmark-historical-aliases-v1.json`
(40 entries, `public_api: false`).

| Historical | Public / descriptive | Python |
|---|---|---|
| UNMARK-A | ViUnMark-Gate | `ViUnMarkGate`, `ViUnMarkGateConfig` |
| V2-SCF | ViUnMark-Scale | `ViUnMarkScale`, `ViUnMarkScaleConfig` |
| R4-W-MLP | Gate Robust Readout | `GateReadoutConfig` |
| R6-U-MLP / V2_U | Scale Unweighted Readout (branch ensemble) | `ScaleUnweightedReadoutConfig` |
| R6-W-MLP / V2_W | Scale Weighted Readout (branch ensemble) | `ScaleWeightedReadoutConfig` |
| VANILLA CAT-U-MLP | PhoBERT Readout | `PhoBERTReadoutConfig` |
| seed_ensemble | within-branch head-logit mean | `ensemble_branch` |
| SYS1 | Adapted-Only Fusion (diagnostic / ablation) | `AdaptedOnlyFusionConfig` |
| SYS2-1 | PhoBERT Robust Readout Selection (development stage) | **none, not public** |
| SYS2-2 | ViUnMark | `ViUnMarkConfig`, `ViUnMark` |
| D1 / D2 / D3 / D4 | Scale Pathway Preflight / Checkpoint-Free Pooling Bridge / Matched-Head Pooling Comparison / Native-Adapted Decision Geometry | none / `diagnostics.pooling_bridge` / `.pooling_comparison` / `.decision_geometry` |
| COMP-D1 / COMP-D2 | Matched-Recipe Complementarity Analysis / Robustness Gain Factorization | `diagnostics.complementarity` / `.factorization` |
| OPT parent freeze | Stage-II optimization parent freeze | lineage `stage2_optimization_parent_freeze` |
| OPT0 | Stage-II representation bank | lineage `stage2_representation_bank_final` |
| OPT1 | Stage-II Readout and Augmentation Screen | lineage `stage2_readout_and_augmentation_screen_protocol`; `development_stages` |
| OPT2 | Stage-II Class-Balance Loss Optimization | lineage `stage2_class_balance_loss_optimization_protocol`; `development_stages` |
| R1 … R6 | screen candidates 1–6 (readout × training distribution, §13.3) | none, not public |
| F (FOCAL_GAMMA_2) | focal loss, gamma 2.0, screened in the class-balance stage | none, not public |
| OPT3 | adapted-pathway robust readout optimization | lineage `adapted_robust_readout_optimization_protocol` / `_final` |
| SQRT_INV_FREQ_WEIGHTED_CE | SQRT_INVERSE_FREQUENCY_CROSS_ENTROPY | `LossKind.*` |
| cross_arm_input_identity | cross-pathway input identity | `require_cross_pathway_input_identity` |
| full_condition_api_seed | FULL-condition API placeholder | `FULL_CONDITION_API_SEED` |
| FT / MM / CAT; U / W; AUGMENTED_6COND | FIRST_TOKEN / MASKED_MEAN / CONCAT; UNWEIGHTED / SQRT_INVERSE_FREQUENCY CE; AUGMENTED_SIX_CONDITIONS | enums |

The historical research modules that already use the old names were not
modified.

---

## 6. Stage-I identities

**Method:**

* encoder `vinai/phobert-base` @ `01daacda68afe13d83023d16ec647239e344a1e6`,
  frozen, FP32;
* hidden 768, tone rows 7, letter rows 5;
* `q = concat[e, tone, letter]`, `f = LayerNorm(Linear(3d, d)(q))`,
  `g = σ(Linear(3d, d)(q))`, `W_gate = 0`, `b_gate = logit(0.01)`;
* objective `L = L_align + L_clean` (cosine distance, masked mean over
  non-special tokens);
* unlabeled `undertheseanlp/UVW-2026`.

| | ViUnMark-Gate | ViUnMark-Scale |
|---|---|---|
| Fusion | `z = g * f + (1 - g) * e` | `scale = ||e||_2 / max(||f||_2, 1e-8)`; `z = g * (scale * f) + (1 - g) * e` |
| Fusion id | `historical-fusion-v1` | `scale-calibrated-fusion-v1` |
| **UIT-VSFC reproduction:** checkpoint sha256 | `6773fbb59c7381ba8ddaa944302124a124f5b8a5cb0a5dbb1a5063f3db4a2a91` | `a32c0167817d457d5067c2a351f2d1b73b26229033f03727f43e2d79f59ef685` |
| **UIT-VSFC reproduction:** selected update | 3500 | 8000 |
| **UIT-VSFC reproduction:** source repository HEAD | `7773c77b1df92a6e685dac13c49765ce974f84d8` | `8de83f0da8d2b38f312e2ff16cb78ebcf9e8e526` |

Selected updates and source HEADs are provenance only. Retraining on another
dataset does not require them, and a test forbids `3500` and `8000` in the
package. `ViUnMarkGate` and `ViUnMarkScale` build through
`stage1.initialisation.fresh_adapter` and load through
`stage1.reconstruct.require_loadable_as` then `reconstruct_adapter` (strict).
Each refuses the other pathway's recorded fusion before touching a tensor.

---

## 7. Inputs: corruption identity and cross-pathway identity

### 7.1 Method (`unmark/viunmark/inputs.py`)

* **Conditions**, exactly: `FULL, P25, P50, P75, P100, STRIP_ALL`. They are
  taken from `unmark.corruption` and checked at import. The corruption
  implementation is unchanged.
* **`CorruptionProtocol(scientific_corruption_seed)`** requires the seed and has
  no default. A test asserts `CorruptionProtocol()` raises, and that `19225`
  appears nowhere in the package.
* **FULL is the no-corruption condition.** `FULL_CONDITION_API_SEED = 0` is an
  **API placeholder, not a scientific seed**
  (`FULL_CONDITION_API_SEED_IS_SCIENTIFIC = False`). A test runs the real
  `corrupt()` on FULL with seeds 0, 19225 and 7, and gets the canonical clean
  text every time.
* **Cross-pathway input identity (REQUIRED).** ViUnMark-Gate and ViUnMark-Scale
  receive the **same** corrupted realization for the same dataset, role,
  sample_id, condition and scientific seed. Corruption is drawn once and
  shared. `require_cross_pathway_input_identity` refuses:
  * a missing Gate or Scale pathway;
  * mismatched key sets;
  * any key with different texts (an independently redrawn corruption is
    refused by test).

  Base-grid invariance is likewise required.

### 7.2 UIT-VSFC reproduction (`uit_vsfc_reproduction.corruption`)

```text
conditions                 FULL, P25, P50, P75, P100, STRIP_ALL
scientific_corruption_seed 19225
full_condition_api_seed    0   (scientific: false — API placeholder for FULL)
eligibility_policy         VIETNAMESE_SYLLABLE_INVENTORY
inventory_sha256           78eeb840d50455b14bd564da5aed7318d96468b8deaad5986b77bf5c538315d2
inventory_source_revision  135a4d9716e49a981624474156d6f247b9b46f6a
max_length                 256
evidence                   stage2_optimization_parent_freeze   f07fa83b38063b81a237eed68f59d2ddda45cd46ff283b7398587af486a8c900
                           stage2_representation_bank_final    0044ba7bda8caa2aabbfbcb2a30fe3bff224d988c5806417c76822715917da50
```

`uit_vsfc_reproduction.corruption_protocol()` is the only place 19225 becomes a
`CorruptionProtocol`. An external protocol must freeze its own seed, or
explicitly freeze reuse of this one.

---

## 8. Readouts and the representation bank

### 8.1 Method (`readout.py`)

```text
FIRST_TOKEN  hidden[:, 0, :]
MASKED_MEAN  mean over attention_mask=1 AND special_tokens_mask=0 (tokenizer-authoritative)
CONCAT       torch.cat([FIRST_TOKEN, MASKED_MEAN], dim=-1)    # exactly [FT ; MM], one hidden tensor
```

PhoBERT-base widths are FT 768, MM 768 and CAT 1 536.
`concat_first_token_masked_mean` reads both halves from one `hidden`. This is
enforced by an AST test; a torch test checks that `[MM ; FT]` differs.

### 8.2 UIT-VSFC reproduction (`uit_vsfc_reproduction.representation_bank`)

```text
dtype torch.float32 · hidden_size 768 · max_length 256
FIRST_TOKEN source          stage2_dual_finalist.stage2_first_token_representation (repository Stage-2 extractor)
MASKED_MEAN source          modeling.pooling.masked_mean_non_special (Stage-1 pooling)
special_tokens_mask source  stage2_dual_finalist.collate_stage2_unmark_batch (authoritative Stage-2 collator)
same_encoder_forward true · base_grid_invariance true · cross_pathway_input_identity true
```

The public `first_token` / `masked_mean` are tested (torch-gated) to equal
`stage2_first_token_representation` and `masked_mean_non_special` on the same
tensors.

---

## 9. Head architecture and losses

### 9.1 Robust MLP head (`heads.py`)

```text
LayerNorm(d) -> Linear(d, 256) -> GELU -> Dropout(p=0.1) -> Linear(256, n)
parameters = 2d + (256d + 256) + (256n + n);  d=768,n=3 -> 199 171;  d=1536,n=3 -> 397 315
```

### 9.2 Losses (`losses.py`)

```text
UNWEIGHTED_CROSS_ENTROPY               class weights: none
SQRT_INVERSE_FREQUENCY_CROSS_ENTROPY   raw[c] = 1 / sqrt(train_count[c]);  weight[c] = raw[c] / mean(raw)
```

**WEIGHTED_CE_RULE = RECOVERED.** The historical loss is named
`SQRT_INV_FREQ_WEIGHTED_CE`, and the native PhoBERT robust-readout protocol
freezes "sqrt-inverse class frequency normalised to mean 1". The **rule** is the
reusable method: the weighted loss refuses to run without the dataset's own
counts, and a test shows new counts give new weights. The **vector** is
UIT-VSFC reproduction data only:

```text
protocol-train counts  negative 4259 · neutral 366 · positive 4514  (9 139 rows)
weight vector          [0.5573523044586182, 1.9012669324874878, 0.5413808226585388]
                       = the rule in float32 arithmetic, exactly (the loader checks equality)
```

---

## 10. Stage-II training recipe

### 10.1 Adapted readout recipe (RECOVERED_AT_PROTOCOL_LEVEL), scoped to the Gate and Scale heads

Machine-readable in `method.robust_mlp_training_recipe`. It applies to
`gate_robust_readout`, `scale_unweighted_readout` and `scale_weighted_readout`,
and is implemented torch-free in `unmark/viunmark/training.py` (torch builders in
`heads.py`).

| Field | Recovered value |
|---|---|
| **Initialisation** | LayerNorm weight 1, bias 0; Linear Xavier-uniform weight, zero bias (`initialize_robust_mlp_head`) |
| **Optimizer** | AdamW, lr 0.01, betas (0.9, 0.999), eps 1e-8; weight decay **0.01 on matrix weights**, **0 on biases, vectors and LayerNorm** (`RobustMLPOptimizerPolicy`, `build_robust_mlp_optimizer`) |
| **Budget** | batch 128; **2160 optimizer updates**; **30 selection boundaries**; **72 updates per boundary** (`2160 = 30 × 72`, `HeadTrainingBudget`) |
| **Augmented training set** | condition-major concatenation in the exact order FULL, P25, P50, P75, P100, STRIP_ALL; each condition preserves the training split's row order; labels repeated once per condition (`augmented_row_order`, `augmented_labels`) |
| **Batch stream** | `cycle_index` starts at 1; `cycle_seed = seed * 1000 + cycle_index`; cycles continue until the update budget is exhausted (`cycle_seed`) |
| **Dropout RNG** | `dropout_seed = seed * 100000 + optimizer_update` (`dropout_seed`) |
| **Checkpoint selection (per head / per seed)** | evaluated every 72 updates on protocol-dev; primary six-condition Macro-F1 mean → worst-condition Macro-F1 → FULL Macro-F1 → earlier boundary (`select_checkpoint_boundary`) |
| **Precision** | float32, no AMP, no TF32 |
| **Best-seed selection** | none |

The rows are **UIT-VSFC reproduction data only**: protocol-train 9 139 and
augmented 54 834 = 9 139 × 6. Neither is a method default, and a test forbids
both in the package.

`select_checkpoint_boundary` is the only function in the package whose name
contains `select`. A structural test confirms it ranks `BoundaryScore` values
inside one head and has no seed or head field. It cannot select a best seed.

### 10.2 Historical selected boundaries

The recorded boundaries are 1-based over 30. For the adapted-pathway heads,
boundary `b` is evaluated at optimizer update `72 · b`.

### 10.3 Implementation details the recovered evidence does not state

The policies are recovered. These lower-level details, needed for bit-exact
retraining, are not stated in the evidence available here. They are recorded in
`robust_mlp_training_recipe.implementation_details_not_stated` rather than
guessed:

* the permutation algorithm inside a cycle, and the handling of a partial final
  batch;
* how the head-initialisation random stream is seeded;
* how `dropout_seed` is applied, and whether `optimizer_update` counts from 0 or 1;
* the state-dict key layout of the historical head files.

### 10.4 Cross-entropy reduction — traced through the repository, UNRESOLVED

**Call path traced.** The repository contains exactly four Stage-II head
training loops that construct and invoke cross entropy. No other AST-level
cross-entropy call exists in `unmark/`, `scripts/` or `docs/colab/`, which is
enforced by test.

| Trainer (file · symbol) | Construction | Invocation | Head built | Class weights |
|---|---|---|---|---|
| `unmark/evaluation/preg1_head.py` · `train_head` | line 925 `loss_fn = nn.CrossEntropyLoss()  # unweighted, no label smoothing` | line 940 `loss_fn(logits, train_y[index])` | `build_head` → `nn.Linear` (line 698) | none |
| `unmark/evaluation/stage2_head_campaign.py` · `train_stage2_head` | line 916 `loss_fn = nn.CrossEntropyLoss()` | line 927 | `build_head` → `nn.Linear` | none |
| `unmark/evaluation/stage2_scf_campaign.py` · `train_scf_head` | line 753 `loss_fn = nn.CrossEntropyLoss()` | line 764 | `build_head` → `nn.Linear` | none |
| `unmark/baselines/restore/stage2.py` · `train_restore_head` | line 775 `loss_fn = nn.CrossEntropyLoss()` | line 786 | `build_head` → `nn.Linear` | none |

Related documentation: `unmark/evaluation/preg1_protocol.py` lines 257–263
declare `LOSS_REDUCTION = "mean"` (line 260) and
`LOSS_SPEC = 'CrossEntropyLoss(weight=None, label_smoothing=0.0, reduction="mean")'`
(lines 261–263) for the **linear** pre-G1 protocol. `LOSS_SPEC` is a string
constant, not a call.

**What this establishes:**

```text
at all four repository call sites:
    explicit repository reduction argument = ABSENT
    effective library default              = "mean"
    head                                   = single nn.Linear
    class weights                          = none
```

**Why it does not resolve the robust MLP heads.** None of these trainers builds
the robust MLP head or uses a weighted loss. The runner that trained the robust
MLP U/W heads appears in **no commit** (§4.3): no `nn.GELU`, `dropout_seed`,
`cycle_seed`, `max_optimizer_updates` or `SQRT_INV_FREQ` anywhere in history.
Carrying the linear trainers' reduction over to those heads would be an
inference from a different code path.

```text
CROSS_ENTROPY_REDUCTION (robust MLP U/W heads) = UNRESOLVED
```

The claim is enforced, not only stated:

* `training.CROSS_ENTROPY_REDUCTION_RECOVERED = False`;
* the spec status is `UNRESOLVED`;
* the provenance loader refuses a spec that claims a reduction;
* `test_viunmark_training.py` verifies by AST that each of the four cited call
  sites is a bare `nn.CrossEntropyLoss()` inside a `build_head` (linear)
  trainer, that no other call site exists, and that no GELU/dropout-seed runner
  exists outside the public layer.

### 10.5 Native PhoBERT Readout training — SUBSTANTIALLY RECOVERED

**Evidence:** the frozen native PhoBERT robust-readout protocol
`s2-sys2-1-protocol-v1.json`, sha256
`e2b6dcff0cb131d5bd10b51eded6c4fe35141bdc04328503018c662d07738cab` (lineage key
`phobert_robust_readout_protocol`). It is recorded in
`method.phobert_readout_training`, whose `confirmed` keys must equal
`training.PHOBERT_READOUT_POLICY_CONFIRMED`.

```text
PHOBERT_READOUT_TRAINING_POLICY = SUBSTANTIALLY_RECOVERED
PHOBERT_BIT_EXACT_RETRAINING    = NOT_RECOVERED
```

**Confirmed by the native protocol:**

| Field | Confirmed value |
|---|---|
| Head / MLP | `LayerNorm(d) -> Linear(d, 256) -> GELU -> Dropout(0.1) -> Linear(256, 3)` |
| Initialisation | LayerNorm weight 1, bias 0; Linear Xavier-uniform weights, zero biases |
| Optimizer (confirmed portion only) | AdamW; lr 0.01; matrix weights weight decay 0.01; bias/vector parameters weight decay 0 |
| Budget | batch 128; 2160 optimizer updates; 30 selection boundaries; 72 updates per boundary |
| Training mode | six-condition augmented; condition order FULL, P25, P50, P75, P100, STRIP_ALL |
| Seeds | 53148, 59945, 42941, 720, 9428 (in `uit_vsfc_reproduction.stage2_head_training.phobert_readout_heads`) |
| Best-seed selection | none |
| Precision | float32; AMP false; TF32 false |
| Per-seed checkpoint selection | six-condition protocol-dev Macro-F1 mean → worst-condition Macro-F1 → FULL Macro-F1 → earlier selection boundary |
| Selected native candidate | CONCAT [FIRST_TOKEN ; MASKED_MEAN], input_dim 1536, unweighted cross entropy, robust MLP |
| Deployment | uniform mean logits over all five selected heads |

**Not established by the native protocol** (`not_established_by_protocol`,
equal to `training.PHOBERT_READOUT_NOT_ESTABLISHED`):

| Field | Why it stays open |
|---|---|
| AdamW betas, AdamW eps | the native protocol does not state them; the Gate/Scale values are not promoted |
| augmented-concatenation implementation (condition-major layout, row preservation, repeated-label implementation) | only the mode and condition order are stated |
| deterministic batch `cycle_seed` | stated only for the Gate/Scale heads |
| `dropout_seed` | stated only for the Gate/Scale heads |
| cross-entropy reduction | not stated; unresolved repository-wide (§10.4) |
| head-initialisation RNG stream mechanics | not stated |
| partial-final-batch mechanics | not stated |
| historical state-dict layout | not stated |

**How the boundary is enforced.** `provenance.require_phobert_record_matches_code`
refuses a record that does any of the following:

* confirms any key outside `PHOBERT_READOUT_POLICY_CONFIRMED`;
* lists other not-established fields;
* claims betas, eps, a cycle seed, a dropout seed, a reduction, an
  augmented-set implementation or a row-order rule anywhere inside the confirmed
  fields;
* misstates a confirmed value against the code's policy constants;
* upgrades `bit_exact_retraining`.

Each refusal is tested. The full Gate/Scale recipe keeps
`evidence_scope = gate_robust_readout, scale_unweighted_readout,
scale_weighted_readout`.

---

## 11. Fusion, systems and calibration

### 11.1 Within-branch head ensemble and hierarchical raw fusion (`fusion.py`)

```text
gate_raw     = mean(5 Gate Robust Readout heads)        # head seeds 53148, 59945, 42941, 720, 9428
scale_u_raw  = mean(5 Scale Unweighted Readout heads)
scale_w_raw  = mean(5 Scale Weighted Readout heads)
phobert_raw  = mean(5 PhoBERT Readout heads)

scale_raw    = 0.5 * scale_u_raw + 0.5 * scale_w_raw
adapted_raw  = 0.5 * gate_raw    + 0.5 * scale_raw
viunmark_raw = 0.5 * adapted_raw + 0.5 * phobert_raw
```

| Branch | ViUnMark weight | per head | Adapted-Only weight | per head |
|---|---|---|---|---|
| PhoBERT Readout | **0.500** | 0.100 | — | — |
| Gate Robust Readout | **0.250** | 0.050 | 0.50 | 0.10 |
| Scale Unweighted Readout | **0.125** | 0.025 | 0.25 | 0.05 |
| Scale Weighted Readout | **0.125** | 0.025 | 0.25 | 0.05 |

`ensemble_branch` takes an equal-weight mean over **exactly** the expected head
set. A missing, extra, duplicated, foreign-branch or misaligned head is refused,
so no seed is ever selected. Weights are declared once and used by both the
fusion and the expansions. Tests check exact equality on synthetic logits
against the nested equations, the branch expansion and a weighted sum over all
20 heads.

### 11.2 Calibration is dataset-specific

`CalibrationConfig(class_index | None, additive_logit_bias=0.0, label_name=None)`.
The generic `ViUnMarkConfig` / `AdaptedOnlyFusionConfig` default to the
identity. A test forbids `1.25` and `0.75` in the package and in the spec's
method section.

```text
Adapted-Only Fusion (diagnostic)  adapted_final  = adapted_raw  + [0, +0.75, 0]   # UIT-VSFC, class 1 "neutral"
ViUnMark (final)                   viunmark_final = viunmark_raw + [0, +1.25, 0]   # UIT-VSFC, class 1 "neutral"
```

### 11.3 Parent biases are not stacked

```text
PARENT_BIASES_STACKED=NO
```

This is enforced by type:

* fusion accepts only RAW `BranchLogits`;
* `CalibratedLogits` is terminal, and passing it in any of the four fusion
  positions is refused;
* `apply_system_calibration` accepts only an uncalibrated system output, so no
  public path produces a calibrated single PhoBERT branch;
* `fuse_viunmark_raw` recomputes the adapted fusion from RAW branches;
* containers cannot be laundered into raw logits.

The historical standalone PhoBERT +0.75 is recorded with
`inherited_by_viunmark: false`, and the loader refuses `true`. A non-vacuity test
shows that stacking would shift class-1 logits by exactly 0.75.

---

## 12. UIT-VSFC reproduction record

### 12.1 The 20 selected heads (seed → boundary, sha256)

| Branch | Heads |
|---|---|
| Gate Robust Readout | 53148→13 `8d068ec69f4c67b9117197ef52141c97038844ca2a8db65066cbb107b3ccfca3` · 59945→14 `cdf52d615d7bc5f0bcf64af9ba51a98b9313753657abee1aff5e7b16706435db` · 42941→14 `00e7a00c293c565ee8ecff0f8d862126733df76729e261a4430a9a584c8d89f8` · 720→12 `0e8ce86edad10199a1e6b670b792d3442a2528d5a371388eefca9612157ccdc4` · 9428→20 `c6409f9d74c38619e956dd5698e6c968c7eef139261d5473ffb7bf9319725cc9` |
| Scale Unweighted Readout | 53148→22 `be5448e72285cf8563fb9235f570c281f149d52290d3974e3c82ee23fdc8bb7b` · 59945→24 `9eace3bb3b22568154eda12aef02b2727bc2f5c33fb8f71d1f7154c546e5d291` · 42941→14 `877defe241573f728e3a659a03ff5e6d39e04bc9de2a9876f85f81b32e07edae` · 720→27 `484818e703a9084ba8594addec629a8c0ba7d5588dce897ee75eec3c4d4ae291` · 9428→26 `c1db40e71518a044c81b9f556b166b8fdba8a05040e05c2189ba23724ba55c7f` |
| Scale Weighted Readout | 53148→19 `481471e02a92c99b7a78c714ed80986c35ad9d7d74d97f81170ebc8a9b021971` · 59945→15 `ba75892a9e03e81fd096ba168e5b0ebf563f77916d5a514d8e70e4b9c3a7901b` · 42941→25 `dc04eeb9972b97b3ec8f849d320fd38ec5821d5c36179890a7e051849aaaace7` · 720→30 `942a7380aa9f86a7df10ca210a5787aa0d6cc5c3862534fecd78997858644189` · 9428→22 `c9066d857f3a0ff3ca7bd9a6b20dfac5c179e79db147ffa3ac31ca7e90e165d2` |
| PhoBERT Readout | 53148→17 `94f8238f391f498bab69449ed0f623281315068d4575a481bb6951c0cdbd2431` · 59945→24 `e2b7315ec6ac623e253f355126a9ba1a7908f5d47f437f7ed1736b8ac04c0ab8` · 42941→12 `5f4503709e2574ea5af6f400a23d7506462d930d22db8bc4990c37830d76d868` · 720→26 `c13f5905c78d8620f959690bc990d347acb57881693c6167107d3abfc25b9bc3` · 9428→27 `84451b7a34128a17dad9f16eb3e7d06beec16cea113b1cdfac3d9d9e703d115b` |

In total that is 20 heads and 6 955 580 Stage-II parameters.

### 12.2 System selection (historical, not rewritten)

* status: `posthoc-exploratory`;
* data: **protocol-dev only**;
* primary criterion: six-condition Macro-F1 mean;
* tie-breaks: worst-condition Macro-F1 → FULL Macro-F1 → smaller absolute
  class-1 bias → fewer deployed/selected heads;
* official validation: no selection, tuning or recalibration;
* official TEST: no training, selection, tuning, recalibration or post-test
  retuning.

Hardware is recorded as not part of the method.

### 12.3 The loader fails closed on drift

`load_uit_vsfc_reproduction()` refuses each of the following, and each refusal is
tested:

* other than 20 heads, or a branch with other seeds;
* an out-of-range boundary or a duplicated digest;
* a weight vector that is not the float32 rule;
* augmented rows ≠ 6 × training rows, or training rows ≠ the sum of counts;
* a parameter total that disagrees with the architecture;
* a calibration label mismatch, or an inherited standalone calibration;
* a FULL placeholder marked scientific;
* a protocol digest reused as an artifact digest;
* a recipe that disagrees with the code's policy constants;
* a claimed cross-entropy reduction;
* the recipe promoted to the PhoBERT heads.

---

## 13. Durable lineage (identities only; the files live outside Git)

### 13.1 Artifact lineage

| Key | Historical alias | sha256 |
|---|---|---|
| `stage1_gate_checkpoint` | UNMARK-A Stage-I | `6773fbb59c7381ba8ddaa944302124a124f5b8a5cb0a5dbb1a5063f3db4a2a91` |
| `stage1_scale_checkpoint` | V2-SCF Stage-I | `a32c0167817d457d5067c2a351f2d1b73b26229033f03727f43e2d79f59ef685` |
| `adapted_robust_readout_optimization_final` | OPT3 final | `8f78de272bd0f5046cbb76053148c74febc6af280dd632bae9a8cd2aa9e2835c` |
| `adapted_only_fusion_final` | SYS1 final | `d520e792df6df1284411fb397a9b15bcd035406868b8b94e1884b54f53fbd5f4` |
| `phobert_robust_readout_selection_final` | SYS2-1 | `d6eeecbf9f8c13cb0cdc89d47c2a1d122d4c2b9b6e9887563dad6f4aa9a0530e` |
| `viunmark_protocol` | SYS2-2 protocol | `f1d3887307ae5ad36abdb7ea307d89af2488933d6e8b1fe25dd01b388089e121` |
| `viunmark_final` | SYS2-2 final | `0f0801b19945ba60537865a32663eed3468f0b4c7968511ddd7d4608a164911a` |
| `viunmark_official_validation_evidence` | — | `a2d9db0e964b48a0084ac2db4173cc585489c5ca04d166516a0cc7f109015a34` |
| `viunmark_official_test_evidence` | — | `7f996fdaadbacd075d407dc4aeea529c824f3df9d476f6f928bad16bfe1269d2` |

The earlier lineage key for OPT3 final, `pre_fusion_optimization_record`, was
renamed. Its semantics are now recovered, and the digest is unchanged.

### 13.2 Protocol lineage (kept distinct from artifact lineage)

| Key | Historical alias | sha256 |
|---|---|---|
| `stage2_optimization_parent_freeze` | OPT parent freeze (`s2-opt-protocol-freeze-v1.json`) | `f07fa83b38063b81a237eed68f59d2ddda45cd46ff283b7398587af486a8c900` |
| `stage2_representation_bank_final` | OPT0 bank final (`s2-opt0-representation-bank-final-v1.json`) | `0044ba7bda8caa2aabbfbcb2a30fe3bff224d988c5806417c76822715917da50` |
| `stage2_readout_and_augmentation_screen_protocol` | OPT1 protocol | `3fbb1b3dc581df123ca3f0f40bd293f3964aba17995cd5b2385e0eae923169e8` |
| `stage2_class_balance_loss_optimization_protocol` | OPT2 protocol | `c802a1b277c6fd48b7d76588737a7abf9e6058c3421d9f0e94adea2c5ea7a4b2` |
| `adapted_robust_readout_optimization_protocol` | OPT3 protocol | `b39bb40df5495479e3ba55ddb25ca45e19990deb8be6230b8f0b65b45eb91689` |
| `phobert_robust_readout_protocol` | native PhoBERT robust-readout protocol (`s2-sys2-1-protocol-v1.json`) | `e2b6dcff0cb131d5bd10b51eded6c4fe35141bdc04328503018c662d07738cab` |

No digest appears in both tables. The loader enforces this, and a test covers it.
The two protocol keys formerly named `stage2_optimization_stage_1_protocol` and
`stage2_optimization_stage_2_protocol` were renamed once their semantics were
recovered. Their digests are unchanged.

### 13.3 Stage-II development stages (recovered semantics, historical only)

Machine-readable in `uit_vsfc_reproduction.development_stages` under descriptive
names. The historical labels appear only in the alias spec. Neither stage has a
public API, and no screened readout (L2_MASKED_MEAN) or loss (focal) was added to
the package; a test enforces this.

**Stage-II Readout and Augmentation Screen** (historical alias OPT1; protocol
`3fbb1b3d…69e8`):

| Candidate (historical alias) | Readout | Training distribution |
|---|---|---|
| 1 (R1) | MASKED_MEAN | CLEAN_ONLY |
| 2 (R2) | L2_MASKED_MEAN | CLEAN_ONLY |
| 3 (R3) | [FIRST_TOKEN ; MASKED_MEAN] | CLEAN_ONLY |
| 4 (R4) | MASKED_MEAN | AUGMENTED_6COND |
| 5 (R5) | L2_MASKED_MEAN | AUGMENTED_6COND |
| 6 (R6) | [FIRST_TOKEN ; MASKED_MEAN] | AUGMENTED_6COND |

All six candidates share a common head `Linear(input_dim, 3, bias=True)` and
unweighted cross entropy. Candidates are ranked across all five seeds, with no
best-seed selection.

**Stage-II Class-Balance Loss Optimization** (historical alias OPT2; protocol
`c802a1b2…a4b2`):

* promoted bases: screen candidates 4 (R4) and 6 (R6);
* loss variants: U = UNWEIGHTED_CE; W = SQRT_INV_FREQ_WEIGHTED_CE; F =
  FOCAL_GAMMA_2 (focal gamma 2.0);
* calibration: the existing shared class-1 calibration protocol.

The loader refuses a screen that is not the full readout × distribution grid, a
screen with best-seed selection, promoted bases other than 4 and 6, a focal gamma
other than 2.0, or a stage whose protocol has no lineage digest.

---

## 14. Analyses under descriptive names (`unmark/viunmark/diagnostics/`)

| Analysis | Implemented | Not implemented, and why |
|---|---|---|
| Scale Pathway Preflight | no module; a structural preflight with no scientific result, covered by config validation and tests | — |
| Checkpoint-Free Pooling Bridge | `same_forward_pooling_pair` (FT and MM from one hidden tensor; no head) | centroid and 1-NN decoders: metric, normalisation and tie-breaks unrecorded |
| Matched-Head Pooling Comparison | `MatchedPoolingComparisonPlan` contract: same pathway, FT vs MM, identical ordered seeds, identical schedule, identical initial-state digests, no best seed, selection within each head | trainer (out of scope) |
| Native-Adapted Decision Geometry | per-example `cosine_distances`; `prediction_agreement` | centered-logit cosine: centring axis and aggregation unrecorded |
| Matched-Recipe Complementarity Analysis | matched recipe; `matched_recipe_streams()` derived from configs; equal-weight `matched_recipe_average` over RAW branch ensembles | which branches the historical analysis averaged is left to the caller |
| Robustness Gain Factorization | seven-step order; steps 4–7 from the final graph with no bias carried forward | steps 1–3: substitutions unrecorded |

---

## 15. Files

### 15.1 Created by P0 (all untracked)

| File | Lines |
|---|---|
| `unmark/viunmark/__init__.py` | 171 |
| `unmark/viunmark/config.py` | 561 |
| `unmark/viunmark/readout.py` | 85 |
| `unmark/viunmark/heads.py` | 132 |
| `unmark/viunmark/losses.py` | 114 |
| `unmark/viunmark/training.py` | 294 |
| `unmark/viunmark/inputs.py` | 133 |
| `unmark/viunmark/fusion.py` | 521 |
| `unmark/viunmark/system.py` | 174 |
| `unmark/viunmark/provenance.py` | 492 |
| `unmark/viunmark/diagnostics/__init__.py` | 42 |
| `unmark/viunmark/diagnostics/pooling_bridge.py` | 32 |
| `unmark/viunmark/diagnostics/pooling_comparison.py` | 90 |
| `unmark/viunmark/diagnostics/decision_geometry.py` | 84 |
| `unmark/viunmark/diagnostics/complementarity.py` | 112 |
| `unmark/viunmark/diagnostics/factorization.py` | 92 |
| `docs/spec/viunmark-final-system-v1.json` | 780 |
| `docs/spec/viunmark-historical-aliases-v1.json` | 362 |
| `tests/test_viunmark_naming.py` | 156 |
| `tests/test_viunmark_historical_integrity.py` | 84 |
| `tests/test_viunmark_method.py` | 335 |
| `tests/test_viunmark_fusion.py` | 400 |
| `tests/test_viunmark_provenance.py` | 740 |
| `tests/test_viunmark_training.py` | 354 |
| `tests/test_viunmark_diagnostics.py` | 220 |
| `tests/test_viunmark_torch.py` | 196 |
| `docs/audits/073-final-viunmark-public-facing-system-recovery.md` | this file |

### 15.2 Changed by the evidence amendment

| File | Change |
|---|---|
| `unmark/viunmark/training.py` | **new**: recovered recipe policies; `CROSS_ENTROPY_REDUCTION_RECOVERED = False` |
| `unmark/viunmark/inputs.py` | **new**: `CorruptionProtocol` (seed required), FULL placeholder, cross-pathway identity guard |
| `tests/test_viunmark_training.py` | **new**: 32 tests |
| `unmark/viunmark/heads.py` | `initialize_robust_mlp_head`, `robust_mlp_parameter_groups`, `build_robust_mlp_optimizer` |
| `unmark/viunmark/provenance.py` | loads and cross-checks corruption, bank, Stage-I provenance, rows, protocol lineage, recipe scope, and recipe-versus-code agreement |
| `unmark/viunmark/__init__.py` | exports |
| `docs/spec/viunmark-final-system-v1.json` | recovery status; method input identity; `robust_mlp_training_recipe`; `phobert_readout_training`; reproduction corruption, representation bank, Stage-I provenance, augmented rows, scoped training record, full-precision weights, protocol lineage; lineage key rename |
| `docs/spec/viunmark-historical-aliases-v1.json` | OPT family, evidence filenames, historical loss and identity field names; 26 → 33 entries |
| `tests/test_viunmark_provenance.py` | 3 existing tests re-pointed to the scoped / renamed / full-precision values, with every assertion kept and one strengthened (exact vector); **+22 tests** |
| `tests/test_viunmark_fusion.py` | the "no best/select function" guard now permits exactly the within-head `select_checkpoint_boundary`; **+1** structural test proving it ranks boundaries, not seeds or heads |
| `tests/test_viunmark_torch.py` | **+3** torch-gated tests (initialisation, optimizer groups, readouts equal the historical bank sources) |
| this audit | amended in place |

### 15.3 Changed by the final review correction

| File | Change |
|---|---|
| `unmark/viunmark/training.py` | `PHOBERT_READOUT_POLICY_CONFIRMED`, `PHOBERT_READOUT_NOT_ESTABLISHED`; docstring states protocol-level recovery |
| `unmark/viunmark/provenance.py` | `require_phobert_record_matches_code` (closed confirmed set, deep rejection of unestablished claims, value agreement with the code); `require_development_stages`; `development_stages` field; PhoBERT head seeds checked |
| `docs/spec/viunmark-final-system-v1.json` | new recovery statuses; `recovery_level: PROTOCOL_LEVEL` on the adapted recipe; structured `phobert_readout_training` (confirmed / not established); PhoBERT head seeds and deployment; `development_stages`; protocol-lineage key renames |
| `docs/spec/viunmark-historical-aliases-v1.json` | OPT1 and OPT2 semantics; R1–R6 and F entries; native protocol filename; 33 → 40 entries |
| `tests/test_viunmark_provenance.py` | 3 tests corrected (lineage keys; status block; the PhoBERT scope test, which had encoded the under-claim and now checks the true boundary); **+20 test functions** (27 cases) |
| `tests/test_viunmark_naming.py` | alias expectations extended (OPT1, OPT2, OPT3, R4, R6, F) |
| this audit | amended in place |

### 15.4 Modified historical files

**None.** `git diff --stat` is empty. None of the new modules imports
`preg1_protocol`, `wandb` or `RunProvenance(**…)`.

---

## 16. Closeout: resolved and unresolved

### 16.1 Resolved by the evidence amendment

The following were previously unresolved and are now closed:

* MLP initialisation;
* optimizer policy;
* training budget;
* boundary spacing (72 updates);
* six-condition augmented construction;
* deterministic batch stream (cycle-seed rule);
* dropout seed rule;
* within-head checkpoint selection;
* weighted-CE rule and its exact UIT-VSFC vector;
* scientific corruption seed and inventory identity;
* cross-pathway input identity;
* representation-bank sources;
* Stage-I selected updates;
* OPT3 semantics;
* protocol lineage.

Closed by the final review correction:

* PhoBERT Readout training policy (head, initialisation, the stated optimizer
  portion, budget, six-condition mode and order, seeds, precision, checkpoint
  selection, selected candidate, deployment);
* OPT1 semantics (Stage-II Readout and Augmentation Screen);
* OPT2 semantics (Stage-II Class-Balance Loss Optimization).

### 16.2 Genuinely unresolved

1. **Cross-entropy reduction of the robust MLP U/W heads.** The runner is absent
   from the repository; the traced evidence is in §10.4.
2. **PhoBERT Readout heads: only the fields the native protocol does not
   establish** (§10.5): AdamW betas and eps; the augmented-concatenation
   implementation; the deterministic batch `cycle_seed`; `dropout_seed`;
   cross-entropy reduction; head-initialisation RNG stream mechanics;
   partial-final-batch mechanics; historical state-dict layout. The Gate/Scale
   values for these are not promoted.
3. **Bit-exact implementation details** (§10.3): the within-cycle permutation
   algorithm and partial-batch handling, the seeding of the initialisation
   stream, the `dropout_seed` application and update indexing base, and the
   head-file state-dict layout.
4. **Analysis gaps** (§14): pooling-bridge decoders, centered-logit cosine, and
   factorization steps 1–3.
5. **Artifact verification.** The 20 head digests and all lineage digests are
   recorded, not re-verified against the real bytes.
6. **Torch-gated tests.** 14 tests did not run in the ML-free `.venv`; no runtime
   was installed.
7. **Carried over from Audit 072, out of scope:** historical Stage-2 cache keys
   do not validate dataset identity at construction.

---

## 17. Tests run and results

| Run | Result |
|---|---|
| Focused: `pytest tests/test_viunmark_*.py` (`.venv`, ML-free) | **218 passed, 14 skipped** (the `requires_torch` tests), 0 failed. After the first amendment: 196 passed, 14 skipped. Before it: 143 passed, 11 skipped. |
| Historical regression subset: `test_stage2_dual_finalist_infra`, `test_stage2_head_campaign`, `test_stage2_campaign`, `test_stage2_scf_pathway`, `test_stage2_scf_campaign`, `test_stage2_tone_channel_regression`, `test_preg1_import_contract`, `test_preg1_head`, `test_stage1_provenance_contract`, `test_corruption`, `baselines/restore/test_restore_contracts` | **1060 passed, 69 skipped**, 0 failed |
| Full suite: `pytest -q -p no:cacheprovider` | **5063 passed, 273 skipped, 0 failed** (181.15 s). After the first amendment: 5041 passed, 273 skipped; the +22 passed are exactly the correction's new focused tests. Before the first amendment: 4988 passed, 270 skipped. |
| Baseline before P0 (Audit 072 §12) | 4845 passed, 259 skipped |

Required amendment tests:

| Requirement | Test |
|---|---|
| historical scientific corruption seed == 19225 | `test_historical_scientific_corruption_seed`; `test_historical_seed_agrees_with_the_research_measurement_seed` |
| FULL API placeholder == 0 and non-scientific | `test_full_placeholder_is_zero_and_labelled_non_scientific`; `test_full_condition_api_seed_is_a_non_scientific_placeholder`; `test_the_full_placeholder_cannot_change_the_text`; `test_loader_refuses_a_scientific_full_placeholder` |
| cross-pathway corruption identity required | `test_cross_pathway_identity_is_required`; `test_independently_redrawn_corruption_is_refused`; `test_missing_pathway_or_mismatched_keys_are_refused` |
| historical max_length == 256 | `test_historical_corruption_identity`; `test_historical_representation_bank` |
| Gate selected update 3500 / Scale 8000 | `test_stage1_selected_updates_and_source_heads`; `test_stage1_provenance_agrees_with_the_historical_research_modules` |
| protocol SHA values | `test_protocol_lineage`; `test_protocol_and_artifact_digests_are_kept_distinct` |
| 2160 == 30 × 72 | `test_budget_is_thirty_boundaries_of_seventy_two_updates`; `test_loader_refuses_a_recipe_that_disagrees_with_the_code` |
| exact condition-major order | `test_augmented_order_is_condition_major_in_the_exact_order`; `test_augmented_labels_repeat_once_per_condition_in_order` |
| weighted CE uses new dataset counts | `test_weighted_ce_uses_a_new_datasets_own_counts`; `test_no_weight_vector_is_a_default` |
| no implicit UIT seed in a generic config | `test_corruption_protocol_has_no_default_seed`; `test_no_row_count_or_seed_of_a_dataset_is_baked_into_the_package`; `test_seed_and_inventory_values_live_only_in_the_reproduction_section` |
| CE reduction claim matches the repository | `test_reduction_is_declared_unresolved`; `test_every_repository_cross_entropy_call_is_a_bare_call_in_a_linear_head_trainer`; `test_the_repository_has_no_other_cross_entropy_call_site`; `test_the_robust_mlp_runner_is_absent_from_the_repository`; `test_the_documented_linear_protocol_loss_is_mean_reduction`; `test_spec_records_the_reduction_as_unresolved`; `test_loader_refuses_a_claimed_cross_entropy_reduction` |
| PhoBERT recipe not promoted | `test_recipe_is_scoped_to_the_adapted_pathway_heads_only`; `test_adapted_recipe_is_not_promoted_to_the_phobert_readout`; `test_loader_refuses_promoting_the_recipe_to_the_phobert_readout` |
| native record confirms initialisation, AdamW, lr, weight decays, budget, order, seeds, no best seed, precision, selection, candidate, deployment | `test_native_protocol_record_is_the_phobert_readout_protocol_digest`; `test_native_protocol_record_confirms_head_and_initialisation`; `test_native_protocol_record_confirms_only_the_stated_optimizer_portion`; `test_native_protocol_record_confirms_training_budget_order_and_precision`; `test_native_protocol_record_confirms_selection_candidate_and_deployment`; `test_native_record_constants_match_the_code` |
| native record rejects betas, eps, cycle_seed, dropout_seed, CE reduction, augmented-concatenation implementation | `test_native_protocol_record_does_not_establish_the_unsupported_fields`; `test_loader_rejects_native_claims_the_protocol_does_not_establish` (8 cases); `test_loader_rejects_a_misstated_native_optimizer`; `test_loader_rejects_an_overclaimed_native_status` |
| OPT1 / OPT2 semantics pinned | `test_readout_and_augmentation_screen_semantics`; `test_class_balance_loss_optimization_semantics`; `test_historical_aliases_pin_the_development_stage_semantics`; `test_no_public_api_exposes_the_screened_readouts_or_focal_loss`; `test_loader_rejects_a_misstated_screen`; naming `EXPECTED_ALIASES` |

All P0 tests from before the amendment still pass. No assertion was removed.

---

## 18. Final git status and diff stat

```text
branch : main
HEAD   : 93227e134b31b850763d9c1dd33ccc9abf3c9ae7   (unchanged)

$ git status --short
?? docs/audits/072-sa-vlsp2016-external-evaluation-repository-reconnaissance.md
?? docs/audits/073-final-viunmark-public-facing-system-recovery.md
?? docs/spec/viunmark-final-system-v1.json
?? docs/spec/viunmark-historical-aliases-v1.json
?? tests/test_viunmark_diagnostics.py
?? tests/test_viunmark_fusion.py
?? tests/test_viunmark_historical_integrity.py
?? tests/test_viunmark_method.py
?? tests/test_viunmark_naming.py
?? tests/test_viunmark_provenance.py
?? tests/test_viunmark_torch.py
?? tests/test_viunmark_training.py
?? unmark/viunmark/

$ git diff --stat
(empty -- no tracked file modified)

Audit 072 sha256 at end: 6f0266d7bcd74602ba81d11292be416e3dcd165b701658d264070a5ce8fc7281 (unchanged)
```

---

## 19. Final verdict

**P0 RESOLVED.** P0 existed to make the final ViUnMark method and its provenance
explicit in the repository. That is done.

```text
FINAL_INFERENCE_DEFINITION       = RECOVERED
ADAPTED_READOUT_TRAINING_POLICY  = RECOVERED_AT_PROTOCOL_LEVEL
PHOBERT_READOUT_TRAINING_POLICY  = SUBSTANTIALLY_RECOVERED
HISTORICAL_BIT_EXACT_RETRAINING  = NOT_FULLY_RECOVERED
CROSS_ENTROPY_REDUCTION          = UNRESOLVED
P0                               = RESOLVED
```

* **Inference.** The final ViUnMark inference definition is fully recovered: one
  20-head ensemble, four within-branch head means, fixed 0.500 / 0.250 / 0.125 /
  0.125 raw-logit fusion, and a single dataset-specific calibration with no
  stacked parent bias.
* **Adapted readout training.** The Gate/Scale readout training policy is
  recovered at protocol level and exposed as reusable policy.
* **PhoBERT Readout training.** The native protocol confirms the head,
  initialisation, the stated optimizer portion, budget, six-condition order,
  seeds, precision, selection, candidate and deployment. It does not establish
  betas, eps, cycle and dropout seeds, reduction, the concatenation
  implementation, or the low-level mechanics, and none is borrowed from the
  Gate/Scale recipe.
* **Development stages.** The Stage-II development stages that preceded the
  final readouts are recovered under descriptive names.
* **What remains.** Method/protocol recovery is not bit-exact historical
  execution recovery. The robust-MLP cross-entropy reduction and the low-level
  execution mechanics remain unresolved.

Nothing was selected, searched, trained, tuned, recalibrated, retuned, or read
from validation or TEST. SA-VLSP2016 implementation has not begun.

No commit and no push was performed.
