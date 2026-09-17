# Audit 072 - SA-VLSP2016 External Evaluation Repository Reconnaissance

**Scope:** Phase E0. Audit the repository to find the smallest additive
engineering path for a later SA-VLSP2016 ViUnMark campaign, while leaving the
whole UIT-VSFC scientific lineage untouched.
**Date:** 2026-09-17
**Type:** **reconnaissance and design only.** No production code, test,
protocol constant, spec, historical audit, checkpoint, head or evidence artifact
was created or modified. This audit is the only file this task writes.

> **Nothing here is implemented, and nothing here is authorised for
> implementation.** Every design in §7–§10 is a proposal for the author to review.
> Every number about SA-VLSP2016 in this document is either (a) quoted from an
> earlier *unaudited* repository record and labelled as such, or (b) absent.

---

## 1. Executive verdict

**E0_RECONNAISSANCE=COMPLETE. E1+ IMPLEMENTATION=BLOCKED.**

There are three findings. The first one changes the plan.

### Finding 1 — the final ViUnMark system is not in this repository

The final system described in the task brief cannot be rebuilt from the
repository, because the repository does not contain it. At starting HEAD
`93227e1`, none of these strings appear in any tracked file:

```text
SYS1  SYS2  SYS2-2  R4-W  R6-U  R6-W  ViUnMark  Scale-U  Scale-W
heterogeneous  "logit fusion"  "bias calibration"  "fusion weight"
```

The search covered more than HEAD. It found nothing in **any commit reachable
from any ref** (`git grep` over `git rev-list --all`), in the stash (empty), or
in untracked or git-ignored files on disk. The four `1.25` hits in the repository
are all Gap-Recovery-Rate examples (`gap_recovery_rate(90, 60, 84) == 1.25`,
`tests/test_evaluation_harness.py:122`). None of them is a class-1 bias.

The frozen Stage-2 protocol goes further and **forbids** the thing the final
system is built from. `docs/spec/stage2-dual-finalist-protocol.json:30` pins
`arms.ensemble = false`, and
`tests/test_stage2_dual_finalist_infra.py:184` refuses `ENSEMBLE` and `A+B`. A
heterogeneous logit fusion is therefore outside the committed Stage-2 protocol by
construction. It must have been defined under a later protocol that was never
committed here.

**Consequence.** Part 1-C of the brief (exact branch weights, ordering of
calibration and fusion, parent-bias stacking) and most of Part 1-B (the Gate
readout recipe, Scale-U, Scale-W, native PhoBERT *as a fusion branch*, and the
FT / masked-mean / CAT readouts) **cannot be cited from the repository**. §4.3
records the values the author supplied, labelled **UNVERIFIED**. An external
campaign that promises to "retrain the ViUnMark method" cannot freeze that method
until its definition is committed. This is prerequisite **P0** in §10.

### Finding 2 — Stage-I never consumed UIT-VSFC

Stage-I trains on the **unlabeled `undertheseanlp/UVW-2026` Vietnamese Wikipedia
corpus** (`configs/data/uvw_2026.json`; `unmark/stage1/protocol.py:45-46`). It
consumes no downstream labels. It selects checkpoints on a held-out **unlabeled**
cosine-distance signal. Its only link to UIT-VSFC is a contamination screen, and
that screen accepts **exactly two** reference sources:

```text
CONTAMINATION_SCREEN_INPUTS = ("uitvsfc_derived_train", "uitvsfc_official_validation")
CONTAMINATION_METHOD        = "exact_canonical_duplicate"        # protocol.py:492-500
```

`screen_contamination` rejects any other key (`unmark/stage1/corpus.py:436`).
So the frozen Stage-I adapters are **dataset-independent**, and they have **never
been screened against SA-VLSP2016**. Phases E3/E4 in the brief ("Stage-I Gate /
Scale training") are therefore a real scientific fork, not a formality. §8
blocker B-11 sets out the choice; this audit does not make it.

### Finding 3 — the existing cache keys do not enforce dataset identity

A check run during this audit (§12, check C4) shows that both the historical
`Stage2RepresentationKey` and the Audit-071 `ScfRepresentationKey` **accept**
`dataset="SA-VLSP2016"` under a UIT-VSFC arm or pathway and a UIT-VSFC protocol
version. Both keys *record* `dataset`. Neither *validates* it. Reuse across
datasets is still refused when a cache is loaded, because `require_compatible`
matches every field exactly. The gap is at construction: an SA-VLSP2016 tensor
could be keyed as, for example, `UNMARK-A / stage2-dual-finalist-protocol-v1`
and pass validation. External evaluation must **not** reuse these key types
(§6, class 5).

### Summary table

| | |
|---|---|
| Starting HEAD | `93227e134b31b850763d9c1dd33ccc9abf3c9ae7` |
| Final ViUnMark system definition in repo | **ABSENT** |
| Stage-I consumes UIT-VSFC labels or text for training | **NO** — unlabeled UVW-2026 |
| Stage-I contamination screen covers SA-VLSP2016 | **NO** |
| Cache keys validate dataset identity at construction | **NO** (verified by execution) |
| Corruption / orthography / linguistics dataset coupling | **NONE** (verified by `git grep`) |
| SA-VLSP2016 facts audited by the repository | **NONE** |
| Production code changed | **0** |
| Files created | **1** (this audit) |

---

## 2. Starting repository state

Recorded before any other action:

```text
branch              : main
HEAD                : 93227e134b31b850763d9c1dd33ccc9abf3c9ae7
git status --short  : (empty)
tree clean before   : YES
refs                : refs/heads/main, refs/remotes/origin/HEAD, refs/remotes/origin/main
                      -- all at 93227e1
stash               : (empty)
latest audit        : 071-stage2-v2-scf-posthoc-pathway-implementation.md
this audit number   : 072 (path verified free before writing)
```

Recent history, for context: `93227e1 Add post-hoc V2-SCF Stage-2 pathway`,
`8de83f0 preflight 5a provenance text repair`, `9b73595 repair c1, c2, c3`,
`4b7ff94 implement c1, c2, c3`.

No commit, push, pull, rebase, merge, branch switch, reset, checkout, restore or
clean was run.

---

## 3. Files, audits and specs inspected

### 3.1 Production modules (read in full or at the cited symbols)

| Area | Files |
|---|---|
| Pre-G1 / UIT-VSFC protocol | `unmark/evaluation/preg1_protocol.py`, `preg1_split.py`, `preg1_head.py`, `profiling.py`, `contracts.py`, `pathways.py`, `metrics.py`, `__init__.py` |
| Stage-2 | `unmark/evaluation/stage2_dual_finalist.py`, `stage2_head_campaign.py`, `stage2_campaign.py`, `stage2_scf_pathway.py`, `stage2_scf_campaign.py`, `stage2_scf_measurement.py` |
| Stage-1 | `unmark/stage1/protocol.py`, `candidates.py`, `contracts.py`, `reconstruct.py`, `initialisation.py`, `trainer.py`, `finalists.py`, `corpus.py`, `objective.py`, `selection.py`, `execute.py`, `checkpoint.py` |
| Adapter | `unmark/modeling/adapter.py`, `config.py`, `contracts.py`, `pooling.py` |
| Corruption | `unmark/corruption/conditions.py`, `corrupt.py`, `deterministic.py`, `eligibility.py`, `__init__.py` |
| Orthography / linguistics | `unmark/orthography/canonical.py`; `unmark/linguistics/classify.py`, `inventory.py` |
| RESTORE baseline | `unmark/baselines/restore/config.py`, `stage2.py`; `scripts/baselines/restore_stage2.py` |
| Scripts | `scripts/preg1_dataset_profile.py`, `scripts/stage1_runner.py` |
| Config | `configs/data/uvw_2026.json` |

### 3.2 Specs

`docs/spec/decisions.md` (D-PREG1-001, -001b, -002, -002b, -003, -004, -004b;
D-G1-005), `stage2-dual-finalist-protocol.json`,
`stage2-v2-scf-posthoc-protocol.json`, `stage1-adapter-finalists.json`,
`restore-baseline-protocol.json`; `unmark-proposal.md` (searched).

### 3.3 Audits

021 (SA-VLSP2016 supersession), 049–051 (dual-finalist protocol and runners),
061 (corrected clean campaign), 062 (author arm selection and v3 measurement),
063 (Vanilla UPPER/FLOOR), 067 (V2-SCF Stage-1), 071 (V2-SCF Stage-2 pathway).

### 3.4 Tests that lock the contracts discussed here

`tests/test_evaluation_harness.py`, `test_preg1_import_contract.py`,
`test_preg1_profiling.py`, `test_stage1_provenance_contract.py`,
`test_stage1_v2_scf.py`, `test_stage1_v2_grd.py`,
`test_linguistics_eligibility.py`, `test_stage2_dual_finalist_infra.py`,
`test_stage2_tone_channel_regression.py`, `test_stage2_scf_pathway.py`,
`test_stage2_scf_campaign.py`.

---

## 4. Reconstructed ViUnMark scientific lineage

This section separates **what the repository records** (§4.1, §4.2) from **what
it does not** (§4.3). Every claim in §4.1–§4.2 carries a source location.

### 4.1 Stage-I — representation adaptation (recorded)

#### A1. The adapter and its two fusion rules

There is **one** adapter class, `OrthographyInputAdapter`
(`unmark/modeling/adapter.py:147`), whose fusion rule is selected by
`AdapterConfig.fusion_id` (`unmark/modeling/config.py:105`). The closed set is:

| Fusion id | Equation | Source |
|---|---|---|
| `historical-fusion-v1` | `z = g·f + (1−g)·e` | `convex_combination`, `adapter.py:136` |
| `scale-calibrated-fusion-v1` | `f_cal = f·‖e‖ / clamp(‖f‖, 1e-8)`, then `z = g·f_cal + (1−g)·e` | `scale_calibrated_fusion`, `adapter.py:96`; applied at `adapter.py:308` |

Here `q = [e; t; l]` (concatenation, `FUSION_INPUT_MULTIPLIER = 3`,
`config.py:29`), `f = LayerNorm(W_f q + b_f)`, and `g = σ(W_g q + c_g)`.

**On the names "Gate" and "Scale".** The repository never uses "ViUnMark-Gate"
or "ViUnMark-Scale". What it does record:

* `UNMARK-A` is the **author-selected continuation arm** (Audit 062 §1). It is
  finalist A (`unmark/stage1/finalists.py:244`): `final_main`, `run_seed=36930`,
  `update=3500`, sha256 `6773fbb5…2a91`, source HEAD `7773c77b…`. It was trained
  under `historical-fusion-v1`.
* `V2-SCF` / C1 is the scale-calibrated candidate. It is bound in
  `stage2_scf_pathway.V2_SCF_CHECKPOINT`: `v2_scf`, `run_seed=36930`,
  `update=8000`, sha256 `a32c0167…f685`, source HEAD `8de83f0d…`.

Mapping **Gate → UNMARK-A** and **Scale → V2-SCF** is consistent with those
records, but the mapping itself is **author-supplied, not repo-recorded**.
Blocker B-18 asks for it to be committed.

#### A2. Objective (shared)

Both trained under `align-clean-pooled-v1` (`HISTORICAL_OBJECTIVE_ID`,
`protocol.py:138`; register `candidates.py:256-288`):

```text
L_align = D(h'(x_p), h(x))      L_clean = D(h'(x), h(x))      D = cosine distance
L       = λ_align·L_align + λ_clean·L_clean                  (objective.py:1-19)
pooling = attention-masked mean over non-special tokens      (STAGE1_POOLING, protocol.py:115)
λ-scale : λ_align + λ_clean = 2 ; r = λ_clean/λ_align = 1.0 → both 1.0   (lambdas_for_r, protocol.py:126)
```

#### A3. What differs between Gate (UNMARK-A) and Scale (V2-SCF)

| | UNMARK-A | V2-SCF |
|---|---|---|
| Fusion id | `historical-fusion-v1` | `scale-calibrated-fusion-v1` |
| Objective | `align-clean-pooled-v1` | **same** |
| run_seed / init_seed | 36930 / 51800 | **same** |
| LR / r | 1e-4 / 1.0 | **same** |
| Screening budget | `PRECOMMITTED_CONTINUATION` (20k → one continuation to 40k) | `FIRST_SCREEN_HARD_CAP` (20k, no continuation) — `candidates.py:172-196` |
| Stage | `final_main` | `v2_scf` |
| Selected update | 3500 | 8000 |
| Source HEAD | `7773c77b…` | `8de83f0d…` |

The parameter set is identical, and so is the initial state for a given seed
(`initialisation.py:30-62`). The fusion equation is the only architectural
difference.

#### A4. Data consumed

* **Corpus:** `undertheseanlp/UVW-2026` at revision `a0a79294…`. Three parquet
  shards are pinned by name, byte size and sha256 (`configs/data/uvw_2026.json`).
  `shard_labels_are_a_split = false`.
* **Labels:** none. `NO_DOWNSTREAM_SELECTION` (`protocol.py:502`).
* **Held-out:** exactly 5 000 documents (`DEV_DOCUMENTS`, `protocol.py:68`),
  partitioned at document level **before** chunking (`corpus.py:1-25`).
* **Contamination screen:** a UVW document is excluded **iff** its whole
  `canon()` digest equals a reference text's (`screen_contamination`,
  `corpus.py:402`). References are limited to the two UIT-VSFC sources.

#### A5. Corruption during Stage-I

Rate `p ~ U(0,1)` per example and per visit. Scope is `TONE_AND_LETTER` with
`PI_STRIP = 0.25`, else `TONE` (`protocol.py:94-108`).

#### A6. Checkpoint selection

`SELECTION_SCORE` = the maximum over `VALIDATION_CONDITIONS = (FULL, P50, P100,
STRIP_ALL)` of mean cosine distance to `h(x)`. Ties go to lower `d_clean`, then
the earliest update (`protocol.py:330-340`). Evaluation runs every 500 updates
(`protocol.py:322`).

#### A7. Frozen encoder

`vinai/phobert-base` at revision `01daacda68afe13d83023d16ec647239e344a1e6`
(`preg1_protocol.py:276-277`). The encoder is frozen and in eval mode. Precision
is FP32 with no AMP (`protocol.py:316`).

#### A8. Dimensions and optimisation needed to retrain the method

| Item | Value | Source |
|---|---|---|
| hidden size `d` | 768 | `stage1/protocol.py:37` |
| tone table / letter table rows | 7 / 5 | `modeling/contracts.py:64,193` |
| fusion / gate | `Linear(3d, d)` each; gate init `W_g=0`, `c_g=logit(0.01)` | `config.py:29`; `contracts.py:301-304` |
| scale epsilon | 1e-8 | `contracts.py:288` |
| trainable parameters | 3 551 232 | `stage1/protocol.py:39` |
| max length | 256; overflow **FAIL** (pre-chunked) | `protocol.py:75-79` |
| optimizer | AdamW, betas (0.9, 0.999), eps 1e-8, no amsgrad | `protocol.py:302-305` |
| weight decay | 0.01 on weights; 0.0 on biases, LayerNorm and **both embedding tables** | `protocol.py:310-311` |
| schedule | constant LR, no warmup, no clipping, no accumulation | `protocol.py:306-309` |
| batch | 128 | `protocol.py:321` |
| base policy | `RAW_BASE`, `b(canon(x))`, no word segmentation | `protocol.py:85` |

### 4.2 Stage-II (recorded)

#### B1. Representation extraction API

`extract_stage2_unmark_representations(pathway, batch)`
(`stage2_dual_finalist.py:966`) runs `base_word_embeddings` → adapter → encoder
with authoritative position ids → `stage2_first_token_representation`
(`stage2_dual_finalist.py:952`). It returns detached FP32 `[B, 768]`. Inputs come
from `prepare_stage2_unmark_input` (`:450`), which carries the Audit-059
tone-channel guard `require_resolved_tone_channel` (`:411`), and from
`collate_stage2_unmark_batch` (`:628`). The Vanilla pathway uses
`pathways.pathway_text` (`pathways.py:64`): `canon(x)` for `VANILLA`,
`decompose(canon(x)).base_text` for `BASE_ONLY`.

#### B2. FT / masked-mean / CAT

| Readout | Status in repo |
|---|---|
| **FT** (`<s>` first token) | **The only Stage-2 readout implemented.** `Preg1Pooling` has one member, `FIRST_TOKEN` (`preg1_protocol.py:190-197`). Both Stage-2 specs pin `pooling.strategy = FIRST_TOKEN`. |
| **masked-mean** | Exists only as the **Stage-1** pooling utility `masked_mean_non_special` (`modeling/pooling.py:41`) and as `TEST_ONLY_masked_mean_pool` (`pathways.py:288`). D-G1-005 (`decisions.md:2504`) states that Stage-1's rule does **not** transfer to Stage-2. |
| **CAT** | **No presence.** The case-sensitive `CONCAT` has zero hits. `concatenat…` hits refer to the adapter's `[e;t;l]`, not to a readout. |

#### B3. Recipes the repository records

| Pathway | Readout | Head / optimiser | Seeds | Training data | Selection | Evidence |
|---|---|---|---|---|---|---|
| UNMARK-A, UNMARK-B | FT | `Linear(768,3)`, Xavier / zero; AdamW LR 0.01, wd 0.01 / 0.0; batch 128; 30 epochs, no early stop | `53148, 59945, 42941, 720, 9428` | **clean** `FULL` protocol-train | clean protocol-dev; macro-F1 → acc → earliest epoch | Audits 061/062; `stage2_head_campaign.py:852` |
| VANILLA (native PhoBERT) | FT | same | same | clean protocol-train | same | Audit 063 §3 |
| V2-SCF | FT | same | same | clean protocol-train | same | **infrastructure only** (Audit 071); `stage2-v2-scf-posthoc-protocol.json` state `scf_stage2_training_started=false`. No V2-SCF Stage-2 **result** is recorded. |

Details shared by every recorded recipe:

* **Class weighting:** none. `LOSS_CLASS_WEIGHTS = None`
  (`preg1_protocol.py:258`). Both Stage-2 specs pin `loss_class_weights = null`.
  **No weighted loss exists anywhere in the repository.**
* **Corrupted samples in Stage-II training:** none. `require_clean_condition`
  (`stage2_head_campaign.py:215`). Audit 063 records
  `CORRUPTED_LABEL_TRAINING=NO`.
* **Roles:** `PROTOCOL_TRAIN` (training), `PROTOCOL_DEV` (checkpoint selection)
  and `OFFICIAL_VALIDATION` (measurement-dev, reporting). `Preg1Role`
  (`preg1_head.py:144`) has **no TEST member**.
* **Measurement corruption seed:** 19225, D-S2-002
  (`stage2_head_campaign.py:176`).

**Selection history that matters for transfer.** UNMARK-A was chosen over B by
the **author, after reviewing official-validation results** (Audit 062 §1). That
choice was post-measurement, on UIT-VSFC data. An external campaign that carries
UNMARK-A forward is carrying a component chosen on UIT-VSFC. §8 B-12 records the
consequence.

### 4.3 Final ViUnMark composition (NOT recorded — author-supplied, UNVERIFIED)

The brief states the following. **None of it can be cited from the repository.**
It is copied here so the gap is visible, **not** so it can be treated as
authority:

```text
SYS2-2 raw fusion (UIT-VSFC), per the task brief:
    native PhoBERT 0.50 · Gate 0.25 · Scale-U 0.125 · Scale-W 0.125
final calibration: a class-1 bias, described as +1.25
lineage names: UNMARK-A, V2-SCF, R4-W, R6-U, R6-W, SYS1, SYS2-2
```

The questions the brief asks about this layer are all **UNRESOLVABLE FROM THE
REPOSITORY**:

| Question | Repository answer |
|---|---|
| Exact Gate readout recipe in the final system | not recorded |
| Scale-U recipe / Scale-W recipe; what "U" and "W" denote | not recorded; no class-weighted loss exists, so if "W" involves weighting, it has no implementation here |
| Native PhoBERT as a fusion branch (clean or corrupted input at inference) | only the stand-alone Vanilla anchor (Audit 063) is recorded |
| R4-W, R6-U, R6-W definitions | not recorded |
| Fusion space (logits, probabilities, log-probabilities) and branch weights | not recorded |
| Order of calibration and fusion; whether parent biases stack | not recorded |
| What SYS1 is, and how SYS2-2 differs from it | not recorded |
| Class index the "+1.25" applies to | not recorded. Under UIT-VSFC's `LABEL_MAPPING`, index 1 is `neutral` (`preg1_protocol.py:41`), but a class-1 bias is not documented anywhere. |
| Which pieces are method and which are UIT-VSFC calibration | cannot be separated without the definitions |

A dataset-independent reading can be offered only for the recorded parts. The
adapter architecture, the Stage-I objective and selection rule, the corruption
semantics, the FT head protocol and the macro-F1 / accuracy metrics are all
dataset-agnostic in content. The **clearly dataset-specific** recorded artifacts
are the UIT-VSFC split, its seeds and hashes, the conflicting-group exclusions,
the author's A-over-B choice on UIT-VSFC official validation, and the
contamination-screen inputs. Whether the unrecorded fusion weights and the
calibration count as "method" or as "UIT-VSFC calibration" is **the author's call**
(B-13, B-14).

---

## 5. Exact reusable APIs and contracts

Legend: **Frozen** = scientific behaviour locked by a decision or spec.
**UIT-VSFC?** = whether the symbol embeds a UIT-VSFC value.
**Reuse** = *verbatim* (call as-is), *wrapper* (call from a new module with
external constants), or *do not reuse*.

### 5.1 Orthography and corruption — verbatim

| Symbol | Source | Frozen | UIT-VSFC? | Reuse |
|---|---|---|---|---|
| `canon(text, placement=DEFAULT)` | `orthography/canonical.py:89` | yes | no | verbatim |
| `CorruptionCondition`, `CONDITIONS` (`FULL, P25, P50, P75, P100, STRIP_ALL`), `get_condition` | `corruption/conditions.py` | yes | no | verbatim |
| `corrupt(text, condition, seed, sample_id, *, purpose=SCIENTIFIC, source_is_clean=True, schema_version="b2-v1", eligibility_policy=None)` | `corruption/corrupt.py:48` | yes | no | verbatim |
| `corrupt_batch(samples, condition, seed, **kw)` — order-independent | `corrupt.py:278` | yes | no | verbatim |
| Key derivation: `BLAKE2b(schema\|seed\|sample_id\|sha256(canon text)\|unit_index)` | `corruption/deterministic.py:1-50` | yes | no | verbatim — **the sample-ID definition fixes the realisation** |
| `CorruptionPurpose`, `EligibilityPolicy`, `require_resolved_eligibility` | `corruption/eligibility.py:59-158` | yes | no | verbatim; scientific runs need the pinned inventory |
| `make_classifier`, `load_inventory` | `linguistics/classify.py:69`; `inventory.py:245` | yes | no | verbatim; **required** by the tone-channel guard |

`git grep` for `vsfc`, `uit-` and `PRIMARY_DATASET` over `unmark/corruption`,
`unmark/orthography` and `unmark/linguistics` returns **nothing** (§12, C3).

### 5.2 Stage-2 input construction and forward — verbatim

| Symbol | Source | Frozen | UIT-VSFC? | Reuse |
|---|---|---|---|---|
| `apply_stage2_corruption(text, condition, *, seed, sample_id, …)` | `stage2_dual_finalist.py:340` | yes | no | verbatim |
| `prepare_stage2_unmark_input(*, text, sample_id, tokenizer, condition, corruption_seed, classifier, …)` | `:450` | yes | `MAX_LENGTH` default only | verbatim |
| `require_resolved_tone_channel` | `:411` | yes | no | verbatim — **keep** |
| `collate_stage2_unmark_batch` | `:628` | yes | no | verbatim |
| `extract_stage2_unmark_representations(pathway, batch)` — duck-typed over `.encoder`, `.adapter`, `.require_frozen()` | `:966` | yes | no | verbatim, as Audit 071 did |
| `stage2_first_token_representation` | `:952` | yes | no | verbatim |
| `load_stage2_phobert_components`, `require_stage2_encoder_identity` | `:701`, `:811` | yes | encoder pin only | verbatim |
| `pathway_text(text, pathway)` | `pathways.py:64` | yes | no | verbatim, for the Vanilla branch |

### 5.3 Stage-1 construction and verification — verbatim or wrapper

| Symbol | Source | Frozen | UIT-VSFC? | Reuse |
|---|---|---|---|---|
| `reconstruct_adapter(payload, hidden)`, `require_loadable_as`, `recorded_identity` | `stage1/reconstruct.py:117,138,69` | yes | no | verbatim |
| `fresh_adapter(hidden, init_seed, fusion_id)` | `stage1/initialisation.py:30` | yes | no | verbatim |
| `candidate_for_stage`, `candidate_for_identity` | `stage1/candidates.py:323,366` | yes | no | verbatim |
| `RunProvenance`, `verify_checkpoint` | `stage1/trainer.py:78,697` | yes | no (fields are plan inputs) | verbatim; `RunProvenance(**mapping)` is **banned** (`test_stage1_provenance_contract.py:137`) |
| `verify_finalist_checkpoint`, `expected_run_provenance` | `stage1/finalists.py:674,652` | yes | binds the Audit-048 corpus and inventory pins | wrapper **only if** B-11 keeps the frozen checkpoints |
| `verify_scf_checkpoint`, `build_scf_adapter` | `stage2_scf_pathway.py` | yes | protocol-bound | verbatim for checkpoint and adapter identity only — **not** for its cache or head types |
| `screen_contamination` | `stage1/corpus.py:402` | yes | **yes** — accepts only UIT-VSFC keys (`:436`) | **do not modify.** A new screen requires B-11. |
| `sha256_file`, `atomic_write_bytes` | `stage1/checkpoint.py:137,159` | yes | no | verbatim |

### 5.4 Head protocol primitives — verbatim, but read the defaults

| Symbol | Source | Frozen | UIT-VSFC? | Reuse |
|---|---|---|---|---|
| `build_head(hidden_size, seed, num_labels=PRIMARY_NUM_LABELS)` | `preg1_head.py:675` | yes | **default** `num_labels=3` | verbatim; **always pass `num_labels` explicitly** |
| `build_optimizer(head, learning_rate)` | `:705` | yes | no | verbatim |
| `deterministic_batches(count, seed, batch_size=128)` | `:731` | yes | no | verbatim |
| `EpochScore(epoch, macro_f1, accuracy)` — epochs must be in `1..30` | `:752` | yes | bound to `EPOCHS=30` | verbatim |
| `select_checkpoint(scores)` | `:771` | yes | no | verbatim |
| `require_full_schedule(scores, epochs=30)` | `:790` | yes | no | verbatim |
| `score_predictions(predictions, labels)` | `:806` | yes | **hard-codes `PRIMARY_NUM_LABELS`** | wrapper: call `macro_f1` / `accuracy` with an explicit external `num_labels` |
| `require_protocol_settings()` | `:113` | yes | no | verbatim |
| `require_head_only_optimizer`, `require_clean_condition`, `label_digest`, `require_batch_provenance` | `stage2_head_campaign.py:777,215,226,652` | yes | no (duck-typed on `.condition`, `.corruption_seed`) | verbatim |
| `ordered_id_digest(sample_ids)` | `preg1_head.py:443` | yes | no | verbatim |

### 5.5 Metrics and dataset machinery — verbatim

| Symbol | Source | Frozen | UIT-VSFC? | Reuse |
|---|---|---|---|---|
| `macro_f1(p, y, num_labels=None)`, `accuracy`, `per_class_scores(p, y, num_labels=None)` | `evaluation/metrics.py:98,35,67` | yes | no | verbatim; pass `num_labels` |
| `DatasetAccess` (incl. `OFFICIAL_AGREEMENT_AUTHORISED`), `DatasetProvenance`, `FileProvenance`, `file_sha256`, `text_digest` | `evaluation/profiling.py:43,104,86,172,181` | yes | no | verbatim — **built as the generic mechanism** (D-PREG1-002b) |
| `profile_split(name, records, classifier)`, `analyse_duplicates(indexes)` | `profiling.py:479,577` | yes | no | verbatim |
| `stratified_group_split(records, fractions, seed)` | `profiling.py:801` | yes | no | verbatim, but **fractions and seed are blocked decisions** (B-9) |
| `derive_seeds(tag, count)` | `profiling.py:866` | yes | no | verbatim, with a **new, distinct tag** |

---

## 6. UIT-VSFC hard-codings discovered

Classes: **1** dataset-agnostic and reusable · **2** generic API instantiated with
UIT-VSFC values · **3** UIT-VSFC-specific, must stay untouched · **4** candidate
for a new external wrapper · **5** unsafe to reuse for external data.

### 6.1 Constants

| Dependency | Location | Class | Note |
|---|---|---|---|
| `PRIMARY_DATASET="UIT-VSFC"`, `PRIMARY_DATASET_VERSION="1.0"`, `PRIMARY_TASK`, `PRIMARY_NUM_LABELS=3` | `preg1_protocol.py:35-39` | **3** | imported by 14 production modules and tests; never edit |
| `LABEL_MAPPING = {negative:0, neutral:1, positive:2}` | `:41` | **3** | SA-VLSP2016 mapping is B-4 |
| `PUBLISHED_SPLIT_SIZES`, `PUBLISHED_LABEL_COUNTS` | `:45-47` | **3** | |
| `CONFLICTING_GROUP_POLICY`, `OBSERVED_CONFLICTING_GROUPS` | `:60-80` | **3** | the *policy shape* informs B-7 |
| `DERIVED_TRAIN_SIZE=11424`, `DERIVED_TRAIN_LABEL_COUNTS`, `DERIVED_*_CSV_SHA256` | `:96-124` | **3** | |
| `SUPERSEDED_DATASET="SA-VLSP2016"` + rationale | `:130-137` | **3** | historical record; **its figures are not authority** (§8 preamble) |
| `INTERNAL_SPLIT_FRACTIONS={protocol-train:0.80, protocol-dev:0.20}` | `:334` | **3** | the 80/20 rationale cites UIT-VSFC imbalance |
| `SPLIT_SEED_TAG="UNMARK-PREG1-SPLIT-UITVSFC-v1"`, `SPLIT_SEED=17486` | `:344-345` | **3** | an external campaign needs its **own** tag |
| `OFFICIAL_TEST_SEALED`, `OFFICIAL_VALIDATION_ROLE="measurement-dev"` | `:321-328` | **3** | |
| Head / optimiser / epoch / seed constants (`EPOCHS`, `BATCH_SIZE`, `ADAMW_*`, `WEIGHT_DECAY_*`, `MEASUREMENT_SEEDS`, `HEAD_*`, `LOSS_*`) | `:206-490` | **1** by content, **2** by location | dataset-agnostic values in a module named for UIT-VSFC pre-G1. Import rather than restate; importing registers the module as a `preg1_protocol` importer (§7.6). |
| `ENCODER_CHECKPOINT`, `ENCODER_REVISION`, `MAX_LENGTH=256` (`FIXED_MAX_LENGTH`, `profiling.py:629`) | `:276-295` | **1** | 256-token coverage on SA-VLSP2016 is **unmeasured** (B-17) |
| `STAGE2_MEASUREMENT_CORRUPTION_SEED=19225` | `stage2_head_campaign.py:176` | **2** | derived from the Stage-1 validation tag; reuse is B-16 |
| `CONTAMINATION_SCREEN_INPUTS` | `stage1/protocol.py:492` | **3** | changing it changes Stage-I identity |
| `CORPUS_MANIFEST_DIGEST` | `stage1/finalists.py:113` | **3** | the post-screen UVW corpus |
| `SPLIT_SEED_TAG` (Stage-1 UVW, a different constant) | `stage1/protocol.py:359` | **1** | unrelated to downstream data |

### 6.2 Functions and types

| Dependency | Location | Class | Note |
|---|---|---|---|
| `load_derived_pool(path, text_col, label_col, id_col, *, expected_sha256=UIT…, expected_rows=11424, expected_label_counts=UIT…)` | `preg1_split.py:181` | **2** / **4** | the defaults are UIT-VSFC. The external loader should be new, not called with overrides. |
| `_label_name` (reads `LABEL_MAPPING`), `build_manifest` (stamps `PRIMARY_DATASET`, `SPLIT_SEED`), `materialize_split` | `preg1_split.py:163,378,472` | **3** | pattern reusable, code not |
| `load_membership` (`preg1-split-v1` schema; only protocol-train and protocol-dev) | `preg1_head.py:270` | **3** | |
| `Preg1Role` (no TEST member) | `preg1_head.py:144` | **3** / **5** | the UIT-VSFC seal is *structural absence*. A campaign that eventually scores TEST cannot reuse it, and must not add a member to it. |
| `RepresentationKey` (`role: Preg1Role`, `pathway: SystemPathway`) | `preg1_head.py:343` | **5** | |
| `Stage2RepresentationKey`, `Stage2HeadRunStore`, `Stage2CampaignManifest`, `stage2_representation_key_for` | `stage2_head_campaign.py:237`; `stage2_campaign.py` | **5** | arm enum limited to A/B; **accepts a foreign `dataset` string** (C4); key builder hard-codes `PRIMARY_DATASET` |
| `ScfRepresentationKey`, `scf_representation_key_for`, `ScfHeadRunStore` | `stage2_scf_campaign.py` | **5** | same construction gap (C4); protocol-bound to the UIT-VSFC post-hoc freeze |
| `score_predictions` | `preg1_head.py:806` | **2** | hard-coded 3 classes |
| `build_head` default `num_labels` | `preg1_head.py:675` | **2** | |
| `aggregate_stage2_campaign` (B−A deltas), `measure_stage2_head` | `stage2_head_campaign.py` | **5** | A/B-specific |
| `screen_contamination` | `stage1/corpus.py:402` | **3** | |
| `read_split` row-index fallback `f"{path.stem}-{index:06d}"` | `scripts/preg1_dataset_profile.py:61-75` | **5** | **row-order dependent.** Because `corrupt()` keys on `sample_id`, reusing this fallback would make the corruption realisation depend on file row order. |

### 6.3 Paths and namespaces

| Dependency | Location | Class |
|---|---|---|
| `drive_root / "stage1-inputs" / "uit-vsfc-derived" / {train,validation}.csv`; `preg1-uit-vsfc-internal-split` | `scripts/baselines/restore_stage2.py:120-130` | **3** |
| `/content/drive/MyDrive/UNMARK/UNMARK-BACKUP/...` namespaces | Audits 061–063; `restore-baseline-protocol.json:303`; `stage1-adapter-finalists.json:84-85` | **3** — the external campaign needs a **new** sibling namespace |
| `--uitvsfc-derived-train`, `--uitvsfc-official-validation` CLI flags | `scripts/stage1_runner.py:825-828` | **3** |

### 6.4 Guards a new module must satisfy (found during this audit)

| Guard | Scope | Consequence |
|---|---|---|
| `test_the_importer_list_is_complete` | rglob over `unmark/`, `scripts/` | Any new importer of `preg1_protocol` **must** be appended to `IMPORTERS` (the Audit-071 precedent). |
| `test_no_dataset_or_benchmark_name_is_baked_in` bans `"vlsp"`, `"uit-"`, `"http"` | **only** `contracts.py`, `metrics.py`, `pathways.py`, `evaluation/__init__.py` | External modules must **not** be re-exported from `unmark/evaluation/__init__.py`. |
| no `RunProvenance(**mapping)` | rglob `unmark/`, `scripts/` | Build identity from the plan. |
| no `import wandb` | rglob `unmark/` | Telemetry only from `scripts/`. |
| no network imports | `unmark/{linguistics,corruption,orthography}` | Add an equivalent guard for the external package (requirement 4). |
| Test basenames | `tests/` has no `__init__.py` (`tests/baselines/restore/` precedent) | New test files need globally unique basenames, e.g. `test_external_*`. |

---

## 7. Minimal additive external-evaluation architecture (PROPOSAL — not implemented)

### 7.1 Placement, derived from repository precedent

The RESTORE baseline is the repository's existing model for a self-contained,
additive campaign: `unmark/baselines/restore/` + `scripts/baselines/` +
`tests/baselines/restore/` + `docs/spec/restore-baseline-protocol.json`.
Following it:

```text
unmark/evaluation/external/            # NEW package; NOT re-exported by unmark/evaluation/__init__.py
    __init__.py                        # torch-free; exports nothing dataset-named
    identity.py        # ExternalDatasetIdentity (frozen, spec-backed), file SHA pins, label map
    roles.py           # ExternalRole: PROTOCOL_TRAIN, PROTOCOL_DEV, TEST_INPUTS  (no TEST_LABELS)
    ingest.py          # path-supplied reading; explicit id construction; never downloads
    membership.py      # split freeze via profiling.stratified_group_split; overlap/duplicate checks
    corruption.py      # thin: calls corrupt()/apply_stage2_corruption only; no new semantics
    cache.py           # ExternalRepresentationKey: VALIDATES dataset identity at construction
    heads.py           # head runner over cached tensors; calls preg1_head primitives verbatim
    system.py          # frozen fusion/calibration definition, loaded from a committed spec (after P0)
    predict.py         # PRE-TEST: produces a SealedPredictions artifact; cannot read labels
    seal.py            # seal digest, atomic write, "is the seal committed at HEAD?" check
unmark/evaluation/external_scoring/    # NEW, SEPARATE package on purpose
    __init__.py
    score.py           # TEST-SCORE: sealed predictions + test labels -> metrics. Nothing else.
scripts/external/
    ext_audit_dataset.py      # E1
    ext_freeze_membership.py  # E2
    ext_extract.py            # E5
    ext_train_heads.py        # E6
    ext_predict_and_seal.py   # E8-E9
    ext_score_test.py         # E10
tests/
    test_external_*.py        # unique basenames
docs/spec/
    external-sa-vlsp2016-dataset-identity.json   # E1 output, committed
    external-sa-vlsp2016-protocol.json           # E2 freeze, committed before any training
```

Scoring gets its own package so the "score-only code cannot load models" rule can
be enforced by an import-graph test rather than by convention.

### 7.2 Requirement coverage

| # | Requirement | Mechanism |
|---|---|---|
| 1–2 | UIT-VSFC code and specs untouched | Everything is new. A test checks historical paths byte-for-byte against `git show 93227e1:<path>` (T-22). |
| 3 | Explicit identity | `ExternalDatasetIdentity(name, release, access: DatasetAccess, files, label_map, schema)` loaded from the committed spec; no defaults. |
| 4 | Supplied by path, never downloaded | `ingest.py` takes paths only; an AST test bans `urllib`, `requests`, `socket`, `http`, `datasets`, `huggingface_hub`. |
| 5 | SHA-256 recorded | `profiling.file_sha256` against spec pins; mismatch raises before parsing. |
| 6 | Explicit, deterministic IDs | Constructed from a rule frozen at E2 (B-10). The row-index fallback is **forbidden** and tested. |
| 7 | Membership durably frozen | `membership.json` + id files; sha256 pinned in the protocol spec. |
| 8 | Overlap rejected | `SplitLeakage` on any id in more than one role. |
| 9 | Duplicates detectable | `analyse_duplicates` over canonical digests within and across roles. The *policy* is B-7/B-8. |
| 10 | Label map frozen | Spec-pinned; ingest refuses unknown labels; digest in every key. |
| 11–13 | Authoritative corruption, six conditions, id-keyed | `corruption.py` only forwards to `apply_stage2_corruption`; conditions come from `STAGE2_UNMARK_CONDITIONS`; T-8–T-10. |
| 14 | Macro-F1 primary, accuracy secondary | `metrics.macro_f1` / `accuracy` with explicit `num_labels`. |
| 15 | Outputs bind identity | `ExternalRepresentationKey` + head artifact bind dataset identity digest, protocol version, role, condition, seed, ordered-id and label digests, checkpoint sha and fusion id, Stage-2 HEAD. |
| 16 | Seal before score | §7.4. |
| 17–20 | No test selection, recalibration or retuning | §7.4–§7.5. |

### 7.3 Key design choices

**A new key type that validates, rather than reusing `Stage2RepresentationKey`.**
`ExternalRepresentationKey.__post_init__` compares `dataset`, `dataset_release`
and `dataset_identity_sha256` with the frozen external identity, and refuses any
role outside `ExternalRole`. This closes the construction gap C4 found in both
existing keys. It is also written so a UIT-VSFC key payload cannot parse (closed
schema, as `ScfRepresentationKey.from_dict` already does).

**`TEST_INPUTS` is a role, and there is no `TEST_LABELS` role.** The
pre-test packages can name test *inputs* for prediction, but no function in them
can name or open test *labels*. Labels are a separate file, pinned by its own
sha256, and only `external_scoring` receives that path.

**A tensor's role comes from its key, never from an argument** (the pattern
`Stage2BoundRepresentations` already uses). The head trainer accepts only
`PROTOCOL_TRAIN` and `PROTOCOL_DEV` keys, so a `TEST_INPUTS` tensor is refused at
the type boundary.

### 7.4 Two-stage TEST discipline, enforced structurally

```text
PRE-TEST  (unmark/evaluation/external/*)
  inputs  : protocol-train, protocol-dev (text+labels), test (ids+text ONLY)
  allowed : extraction, head training, dev checkpoint selection, fusion/calibration
            freeze from the committed spec, test forward pass
  output  : SealedPredictions{ per-branch x per-seed x condition logits and argmax,
            ordered test ids, system-definition digest, calibration digest,
            checkpoint shas, protocol version, execution HEAD } + seal_sha256

TEST-SCORE  (unmark/evaluation/external_scoring/score.py)
  inputs  : SealedPredictions file, seal record, test-labels file
  allowed : verify seal, verify ids, compute macro-F1/accuracy/per-class
  output  : immutable score report; every seed and branch reported
```

How each rule is enforced:

1. **Pre-test cannot read test labels.** (a) The test-inputs file must be
   label-free. E1 must produce, or E2 must verify, a label-stripped inputs file
   with its own sha256 (B-6 decides whether this is derivable). Ingest refuses a
   test-inputs file that contains the label column. (b) An AST test forbids
   `external/*` from importing `external_scoring` or referring to the
   test-labels spec field.
2. **Score-only cannot load a model or build an optimiser.** An import-graph
   test: `external_scoring/score.py` may import only `json`, `hashlib`,
   `pathlib`, `dataclasses`, `typing` and `unmark.evaluation.metrics`. It is
   forbidden `torch`, `transformers`, `unmark.stage1`, `unmark.modeling`,
   `unmark.evaluation.stage2_*`, `unmark.evaluation.external.heads` and
   `.system`. A companion test asserts that no name `build_head`,
   `build_optimizer`, `AdamW`, `argmax over seeds`, `load_state_dict` or
   `reconstruct_adapter` appears in its AST.
3. **Score-only accepts only sealed predictions.** `score.py` recomputes the
   seal digest over the canonical JSON of `SealedPredictions`. It refuses a
   mismatch, a missing branch, seed or condition, an id list different from the
   frozen test ids, or a system or calibration digest different from the
   committed protocol spec.
4. **Seal before labels, checked against git and not trusted to the operator.**
   `score.py` refuses unless the seal record is present **at `git HEAD`**, read
   with `git show HEAD:<seal-record>`. That is the same read-only technique
   `tests/test_preg1_import_contract.py` (`git_blob`) already uses. The author's
   own commit of the seal therefore becomes the gate that opens scoring. Scoring
   also refuses if the test-labels sha256 was not pinned in the committed E2
   spec.
5. **No best-seed, cross-branch or test-driven choice.** The score report
   schema has **no** `winner`, `best_seed` or `selected_branch` field (closed
   schema, as in `validate_scf_head_artifact`). It must contain every seed and
   every declared branch.
6. **No recalibration.** Calibration parameters live only in the committed
   protocol spec. `score.py` has no parameter that accepts a bias or weight. The
   sealed artifact carries post-calibration predictions **and** the calibration
   digest, and the scorer checks that digest against the spec.
7. **No post-test retuning.** Once a score report exists,
   `external/predict.py` refuses to write a new seal for the same protocol
   version. A new seal requires a new, separately justified protocol version,
   and that event is visible in git history.

### 7.5 Relationship to the existing UIT-VSFC lineage

* No historical module gains a parameter, a member, a branch or a default.
* Frozen Stage-I checkpoints are consumed, if B-11 allows, only through
  `verify_finalist_checkpoint` / `verify_scf_checkpoint` and
  `reconstruct_adapter`, all called unchanged.
* The Stage-2 forward pass is reused through duck typing, exactly as Audit 071
  did.
* The design deliberately leaves the historical and V2-SCF key gap (C4)
  unrepaired. Both are committed and bound to existing evidence. The external
  package avoids them instead of changing them. Whether to harden them later is
  a separate author decision.

### 7.6 Obligations the implementation will inherit

* Register each new `preg1_protocol` importer in
  `tests/test_preg1_import_contract.py::IMPORTERS`.
* Pin the Stage-2 execution HEAD separately from any Stage-1 source HEAD, as in
  Audit 071.
* Supply a real syllable-inventory classifier to every input preparation. The
  Audit-059 guard stays live.

---

## 8. BLOCKED SCIENTIFIC DECISIONS — DO NOT IMPLEMENT YET

**About earlier SA-VLSP2016 figures.** `decisions.md` D-PREG1-001
(`decisions.md:2564`) once recorded several claims from published reports:
"5,100 train / 1,050 test", "balanced", source pool counts, a signed user
agreement, and no separate dev split (D-PREG1-004, `:2958`). **None was audited
against data by this repository.** D-PREG1-001b superseded that selection before
any SA-VLSP2016 file was profiled. These claims are hypotheses, not protocol
authority. Everything below stays open until the evidence named in its row
exists.

| ID | Decision | Why it is blocked | Evidence required |
|---|---|---|---|
| **P0** | **Commit the final ViUnMark system definition** | §4.3: no fusion weights, calibration, branch recipes, U/W semantics, SYS1/SYS2-2 or R4/R6 definitions exist in the repo, and the committed Stage-2 spec forbids ensembles | A committed spec and audit defining every branch (checkpoint, readout, loss including any weighting, seeds), fusion space, weights, calibration form, order and stacking, with the UIT-VSFC evidence hashes that produced them |
| B-1 | Canonical SA-VLSP2016 release identity | the access model differs by distributor (D-PREG1-002b) | distributor, release name and version, `DatasetAccess` value, licence status, per-file sha256 and byte size of the copy actually obtained |
| B-2 | File format and schema | unaudited | encoding, delimiter or format, column names, header presence, a row-parse report with no errors or with enumerated errors, a quoting and newline audit |
| B-3 | Exact train and test counts | the "5,100 / 1,050" figures are unaudited | row counts per file after parsing; empty-text and malformed-row counts |
| B-4 | Label names and integer mapping | index order decides what "class 1" means | the raw label vocabulary per file and a frozen `name → int` map with justification. **Do not assume `negative/neutral/positive = 0/1/2`.** |
| B-5 | Whether an official dev split exists | decides whether the UIT-VSFC three-role design (protocol-train / protocol-dev / measurement-dev) maps at all | the release's file inventory and documentation |
| B-6 | Whether official TEST labels are bundled | decides whether a label-stripped inputs file can be derived locally, or labels arrive separately | release inventory. **Record whether the file has a label column without reading the label values.** |
| B-7 | Exact-duplicate policy | UIT-VSFC excluded whole conflicting-label canonical groups from protocol-train only | `analyse_duplicates` report: within-split and cross-split canonical duplicate groups, conflicting-label groups, and train↔test overlap counts from an integrity-only pass |
| B-8 | Near-duplicate and group policy | only exact canonical grouping exists in repo | a decision on whether a near-duplicate metric is needed, and its threshold, *before* any score exists |
| B-9 | Internal train/dev split (if B-5 = none) | 80/20 was justified by UIT-VSFC's ~4% neutral | the label distribution of train; the fractions; a **new** seed tag (e.g. `UNMARK-EXT-SPLIT-SAVLSP2016-v1`); stratification/grouping confirmation |
| B-10 | Sample-ID construction | `corrupt()` keys on `sample_id`, so this fixes every corruption realisation | presence and uniqueness of a native id column; if absent, a rule that is stable under row reordering and unique despite duplicate texts. **The row-index fallback is excluded.** |
| B-11 | **Stage-I: reuse frozen UNMARK-A / V2-SCF, or retrain** | Stage-I is dataset-independent (UVW-2026, no labels). The frozen screen covered only UIT-VSFC. A new screen changes `CONTAMINATION_SCREEN_INPUTS` and `CORPUS_MANIFEST_DIGEST`, so it defines a new Stage-I identity. Training Stage-I on SA-VLSP2016 protocol-train text would change the method itself. | An author decision between: (a) reuse frozen checkpoints and report the screen gap; (b) re-screen UVW against SA-VLSP2016 train+dev and retrain under a new additive preparation path; (c) other. Plus an integrity-only exact-canonical overlap count of SA-VLSP2016 train/dev against UVW-2026 documents. |
| B-12 | Whether Stage-II mirrors the final UIT-VSFC recipe | the final recipe is unrecorded (P0); UNMARK-A was author-selected on UIT-VSFC official validation | P0 complete; an explicit statement that component choices made on UIT-VSFC transfer as fixed method choices and are **not** re-selected on SA-VLSP2016 |
| B-13 | Calibration policy on the target dataset | the calibration form is unrecorded; the target's label balance is unaudited | P0 complete; B-4 and the train label distribution; a rule written before any dev score |
| B-14 | UIT-VSFC "+1.25 class-1 bias": transfer, re-estimate on protocol-dev under a frozen rule, or remove | the value and its class index are unrecorded; "class 1" may not mean the same label under B-4 | P0 (its derivation), B-4, and a pre-registered re-estimation rule if chosen |
| B-15 | Changed-row expectations per condition | orthographic exposure and eligibility are dataset-specific; the Audit-059 dead-tone failure produced *zero* changes silently | a `profile_split` / `observe_orthography` report with the pinned inventory. For P25–STRIP_ALL on train and dev: changed-text rows and changed-`tone_ids` rows, and confirmation that `FULL` changes nothing |
| B-16 | Measurement corruption seed | 19225 was frozen for UIT-VSFC Stage-2 (D-S2-002) | a decision to reuse 19225, or a new derived tag, recorded before any degraded score |
| B-17 | 256-token coverage and truncation | measured for UIT-VSFC only | a `length_coverage` report on SA-VLSP2016 train (not test) |
| B-18 | Gate → UNMARK-A, Scale → V2-SCF naming | author-supplied, not in the repo | a committed statement binding each published name to a checkpoint sha256 and fusion id |
| B-19 | Whether a measurement-dev role exists externally | UIT-VSFC's official validation served as measurement-dev | B-5; a role table for the external protocol |
| B-20 | Licence and redistribution | agreement-based access is possible | licence text or agreement terms; confirmation that no SA-VLSP2016 text enters git, reports or audits |

---

## 9. Proposed future test matrix (not implemented)

**H** = historical safety · **D** = dataset identity · **C** = corruption ·
**K** = cache · **R** = role and leakage · **S** = sealing and scoring.

| ID | Test | Kind | Mechanism |
|---|---|---|---|
| T-1 | Existing full suite stays green | H | run the suite; compare pass count with the baseline in §12 |
| T-2 | No historical frozen protocol constant changed | H | import and compare `preg1_protocol`, `stage1/protocol`, `stage2_head_campaign` constants with literal expected values |
| T-3 | Deterministic split | D | build membership twice, and under shuffled input, with identical id sets and assignment digest |
| T-4 | Row-order-independent sample identity | D | shuffle the source rows; ids per text-and-label record are unchanged; the row-index fallback is absent (AST) |
| T-5 | Overlap rejected | R | an id in two roles raises `SplitLeakage` |
| T-6 | Label-map identity | D | an unknown label raises; the map digest equals the spec; the map is not `preg1_protocol.LABEL_MAPPING` by object identity |
| T-7 | Dataset SHA mismatch | D | a one-byte change fails before any parse |
| T-8 | Corruption determinism | C | the same `(text, condition, seed, id)` gives identical output across calls and process order |
| T-9 | `FULL` leaves canonical text uncorrupted | C | `corrupted_text == canon(text)` for every row |
| T-10 | Degraded conditions reuse authoritative semantics | C | AST: `external/corruption.py` calls only `apply_stage2_corruption` / `corrupt`; output equals a direct `corrupt()` call |
| T-11 | Cache keys cannot cross datasets | K | a key with a UIT-VSFC identity raises at **construction**; a UIT-VSFC key payload does not parse |
| T-12 | Cache keys cannot cross roles or splits | K | `require_compatible` fails on role and on ordered-id digest |
| T-13 | Cache keys cannot cross checkpoints or fusions | K | differing sha256 or fusion id fails |
| T-14 | External Stage-I cannot consume TEST | R | the preparation path has no test argument (AST); a test-inputs key is refused |
| T-15 | Stage-II trainer cannot consume TEST | R | a `TEST_INPUTS` bound tensor is refused by role read from its key |
| T-16 | Pre-test runner cannot read TEST labels | R, S | AST ban on importing `external_scoring` and the labels field; a test-inputs file with a label column is refused |
| T-17 | Score-only cannot load models | S | import-graph allowlist |
| T-18 | Score-only cannot create an optimiser | S | AST ban on `build_optimizer`, `torch.optim`, `AdamW` |
| T-19 | Score-only accepts only sealed vectors | S | a tampered payload, a missing seed, a wrong id order or an uncommitted seal all raise |
| T-20 | No best-seed selection | S | report schema is closed and has no `best_seed`; all seeds present |
| T-21 | No cross-branch selection from TEST | S | report schema has no `winner` or `selected_branch`; the system digest must equal the spec |
| T-22 | Historical UIT-VSFC paths unchanged | H | for each historical module and spec, `git show 93227e1:<path>` bytes equal the working tree |
| T-23 | External package not re-exported | H | `unmark/evaluation/__init__.py` does not import `external*` |
| T-24 | No network in external packages | D | import ban, as in `test_linguistics_eligibility.py` |
| T-25 | Tone channel live on external data | C | on a fixture row, `P100` `tone_ids` differ from `FULL` with the real classifier; `classifier=None` refuses |
| T-26 | Importer registry extended | H | `test_preg1_import_contract` passes with the new modules registered |
| T-27 | Scoring refuses an unpinned labels file | S | a labels sha256 absent from the committed E2 spec raises |

---

## 10. Proposed Colab execution phases (no execution now)

No runtime estimates are given. The repository has a few timings, but none for
an SA-VLSP2016 workload.

Namespace, following the existing pattern:
`/content/drive/MyDrive/UNMARK/UNMARK-BACKUP/external/sa-vlsp2016/<execution-head-prefix>/...`

| Phase | Purpose | Allowed data roles | Artifacts that must exist at the end |
|---|---|---|---|
| **P0** | commit the final ViUnMark definition | UIT-VSFC evidence only (already produced) | committed spec and audit for SYS2-2 and all branches (§8 P0) |
| **E1** | acquisition + immutable dataset audit | train: text, labels, stats. Test: **integrity only** (sha256, row count, schema, label-column *presence*, id uniqueness, canonical overlap counts). **No test label values, no test text in reports.** | `external-sa-vlsp2016-dataset-identity.json` (committed); file sha256s; schema report; train label distribution; duplicate report; length coverage (train); orthography and changed-row profile (train); UVW overlap count (B-11) |
| **E2** | external protocol freeze | none newly read | `external-sa-vlsp2016-protocol.json` (committed **before** any training): ids rule, membership sha256s, label map, roles, seeds, conditions, corruption seed, Stage-I decision, branch recipes, fusion and calibration **rule**, test-labels sha256 pin |
| **E3** | Stage-I Gate | **conditional on B-11.** Reuse: verify the UNMARK-A sha only. Retrain: UVW-2026 + a screen against protocol-train/dev only; never test. | verification evidence, or a new Stage-I run artifact with a new corpus manifest digest |
| **E4** | Stage-I Scale | as E3, for V2-SCF / `scale-calibrated-fusion-v1` | as E3 |
| **E5** | Stage-II representation extraction | protocol-train, protocol-dev (clean `FULL`); test **inputs** for later prediction under all six conditions | `ExternalRepresentationKey`-bound caches per branch × role × condition; extraction evidence JSON |
| **E6** | head training + dev checkpoint selection | protocol-train (train), protocol-dev (select) | 5 heads × every branch; closed-schema head artifacts; per-epoch history digests |
| **E7** | fusion and calibration freeze | protocol-dev only, and **only if** the E2 rule allows dev estimation | committed calibration record + system-definition digest |
| **E8** | TEST prediction without labels | test **inputs** only | per-branch × seed × condition logits and final predictions |
| **E9** | prediction seal | none | `SealedPredictions` + seal record; **author commits the seal record** |
| **E10** | TEST label-only scoring | test **labels** + the sealed artifact; **no model, no optimiser** | immutable score report (all seeds, all branches, six conditions) |
| **E11** | immutable evidence and reporting closeout | none newly read | closeout audit citing every sha256 above; labelled external evaluation; no retuning |

Gates between phases: E2 needs P0 and B-1–B-20 resolved. E5 needs E2 committed.
E8 needs E6 and E7 complete. E10 needs E9's seal record at `git HEAD`.

---

## 11. Files changed by this task

| File | Change |
|---|---|
| `docs/audits/072-sa-vlsp2016-external-evaluation-repository-reconnaissance.md` | **created** (this audit) |

No other file was created, modified, renamed or deleted. **Production code
changes: 0.**

---

## 12. Commands and checks actually run

All read-only with respect to tracked files. The Python import in C4 and the
test run may write git-ignored `__pycache__/` directories. The test run used
`-p no:cacheprovider`.

| ID | Command / check | Result |
|---|---|---|
| C1 | `git rev-parse --abbrev-ref HEAD`; `git rev-parse HEAD`; `git status --short`; `git log --oneline -15`; `git for-each-ref`; `git stash list` | §2 |
| C2 | `git grep -I -F` for `SYS2-2 SYS2_2 SYS1 SYS2 R4-W R4_W R6-U R6_U R6-W R6_W ViUnMark heterogeneous "logit fusion" Scale-U Scale-W 1.25 VLSP`; the same over `$(git rev-list --all)`; `grep -rIl` over untracked and ignored files | 0 hits for every final-system name at HEAD, across history and on disk; `1.25` hits are GRR only |
| C3 | `git grep` for `vsfc`, `uit-`, `PRIMARY_DATASET` over `unmark/corruption unmark/orthography unmark/linguistics` | no hits |
| C4 | In-memory Python: construct `Stage2RepresentationKey(arm="UNMARK-A", dataset="SA-VLSP2016", …)` and `ScfRepresentationKey(pathway_id="UNMARK-V2-SCF", dataset="SA-VLSP2016", …)`; also `role="official-test"` | both keys **accepted** the foreign dataset; `official-test` role **rejected** (`ValueError`) |
| C5 | `git grep -c` per file for `UIT-VSFC uit-vsfc uitvsfc PRIMARY_DATASET PRIMARY_NUM_LABELS dataset_version protocol-train protocol-dev official-validation official-test OFFICIAL_TEST LABEL_MAPPING SPLIT_SEED /content/drive` | §6 |
| C6 | Guard-scope inspection: `test_evaluation_harness.py:44-49,389`; `test_preg1_import_contract.py:33-100`; `test_stage1_provenance_contract.py:137`; `test_stage1_v2_scf.py:619`; `test_linguistics_eligibility.py:355` | §6.4 |
| C7 | `.venv/bin/python -m pytest -q -p no:cacheprovider` (full suite, ML-free environment) | **4845 passed, 259 skipped, 0 failed** (176.91 s). Skips are the existing torch- and CUDA-gated tests; torch is not installed in `.venv`. Nothing in this task can affect the count, because no code or test file changed. |

No dataset was downloaded or opened. No UIT-VSFC or SA-VLSP2016 file was read.
No official validation or TEST data was touched. Nothing was trained.

---

## 13. Final git status

```text
$ git status --short
?? docs/audits/072-sa-vlsp2016-external-evaluation-repository-reconnaissance.md

$ git diff --stat
(empty -- no tracked file modified)

HEAD unchanged: 93227e134b31b850763d9c1dd33ccc9abf3c9ae7
```

---

## 14. Final verdict

**E0 reconnaissance: COMPLETE. Implementation: NOT AUTHORISED.**

1. **The repository cannot define the system to be evaluated.** The final
   ViUnMark composition (branch weights, calibration, the Gate / Scale-U /
   Scale-W / native readouts, SYS1 / SYS2-2, R4-W / R6-U / R6-W) is absent from
   every commit and file. The committed Stage-2 spec forbids ensembles. Until P0
   commits that definition, "retrain the ViUnMark method on SA-VLSP2016" has no
   frozen referent.
2. **Most of the machinery is already reusable without changing it.** That
   includes corruption, orthography, eligibility, Stage-2 input construction and
   forward, adapter reconstruction and checkpoint verification, the FT head
   primitives, metrics, and the generic dataset-provenance, duplicate and
   stratified-group-split tools. None of them embeds UIT-VSFC in its logic. The
   few UIT-VSFC *defaults* (`build_head`, `score_predictions`,
   `load_derived_pool`) are avoided by passing explicit values or wrapping.
3. **Three existing structures must not be reused for external data.** The
   `Preg1Role` TEST seal is structural absence. Both existing cache-key families
   accept a foreign dataset at construction. The profiler's sample-id fallback is
   row-order dependent. The proposed `unmark/evaluation/external/` and separate
   `external_scoring/` packages avoid all three without editing them.
4. **Stage-I is a genuine fork.** The frozen adapters never saw downstream data
   and were never screened against SA-VLSP2016. Reusing them and retraining them
   are different experiments. That decision is the author's (B-11).
5. **Every SA-VLSP2016 fact is still unaudited.** No count, label map, split
   structure, test-label arrangement or id rule may be written into code until
   E1 produces the evidence listed in §8.

No commit and no push was performed. This phase stops here.
