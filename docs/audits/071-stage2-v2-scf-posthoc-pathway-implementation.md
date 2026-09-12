# Audit 071 - Stage-2 Post-Hoc V2-SCF Pathway Implementation

**Scope:** audit the repository, then implement the **minimal additive** Stage-2
pathway required to train and measure the post-hoc UNMARK-v2 C1 / V2-SCF
candidate correctly, under the **corrected historical Stage-2 head protocol**,
without mutating the frozen UNMARK-A / UNMARK-B dual-finalist contract.
**Date:** 2026-09-12
**Type:** additive implementation + protocol freeze + tests. **No historical
production module, protocol constant, scientific hyperparameter, cache schema,
head artifact schema, historical audit or Drive artifact is modified.** One test
registry is extended to cover the new modules (§4).

---

## 1. Executive verdict

**V2_SCF_STAGE2_PATHWAY_IMPLEMENTED=PASS. REAL STAGE-2 TRAINING: NOT AUTHORISED.**

Seven files were added. **One existing file was modified, and it is a test
registry, not production code** (§4). No production module, protocol constant,
scientific hyperparameter, cache schema or head artifact schema was touched. The
V2-SCF Stage-2 pathway is implemented, frozen before results exist, and fully
fail-closed against the one defect that would otherwise be silent: loading a
scale-calibrated adapter's weights into the historical mixture rule.

| | |
|---|---|
| Repository HEAD during this work | `8de83f0da8d2b38f312e2ff16cb78ebcf9e8e526` |
| Production files changed | **0** |
| Test-registry files changed | **1** |
| Files added | **7** |
| Historical A/B contract mutated | **NO** |
| V2-SCF added as a third `Stage2UnmarkArm` | **NO** |
| Protocol frozen before any V2-SCF Stage-2 result | **YES** |
| Real Stage-2 training performed | **NO** |
| Official validation read | **NO** |
| Official TEST read or routable | **NO** |
| Stage-1 retraining required | **NO** |
| Historical A/B caches or heads reusable as V2-SCF | **NO** |
| Real Stage-2 training authorised | **NO** — see §10 |

---

## 2. Repository state

```text
branch : main
HEAD   : 8de83f0da8d2b38f312e2ff16cb78ebcf9e8e526
status : clean before this work; afterwards seven untracked additions
         plus one modified test registry (tests/test_preg1_import_contract.py)
```

The local documentation/implementation HEAD happens to be **the same commit** as
the Stage-1 V2-SCF source HEAD. That coincidence is *not* relied on anywhere:
the cache key binds `stage1_source_repository_head` and
`stage2_repository_head` as **two separate fields**, and the Stage-2 execution
HEAD will be a later commit once the author commits this work. Nothing derives
one from the other.

**No commit and no push was performed. The author alone performs those.**

---

## 3. What was audited before anything was written

The APIs below were read in full before any design decision, so that no
signature was guessed:

| Module | What it establishes |
|---|---|
| `unmark.stage1.candidates` | closed candidate register; `candidate_for_stage("v2_scf")` gives `(align-clean-pooled-v1, scale-calibrated-fusion-v1)` and `FIRST_SCREEN_HARD_CAP` |
| `unmark.stage1.contracts` | `FusionIdentity`, `ObjectiveIdentity`, `SCALE_CALIBRATED_FUSION` |
| `unmark.stage1.reconstruct` | **the authoritative adapter constructor**: `recorded_fusion_id` -> `adapter_for_provenance` -> `reconstruct_adapter` -> `load_adapter_state(strict=True)`, plus `require_loadable_as` |
| `unmark.stage1.initialisation` | `fresh_adapter(hidden, seed, fusion_id)` -> `AdapterConfig(fusion_id=...)` |
| `unmark.modeling.adapter` | `scale_calibrated_fusion()` — the ONE implementation of the C1 equation |
| `unmark.stage1.trainer` | `RunProvenance.require_match` (compares `objective` **and** `fusion`), `verify_checkpoint`, `CHECKPOINT_SCHEMA_VERSION` |
| `unmark.stage1.finalists` | `verify_finalist_checkpoint` pattern, `CORPUS_MANIFEST_DIGEST`, inventory pins, `ADAPTER_STATE_KEYS` |
| `unmark.evaluation.stage2_dual_finalist` | frozen A/B arms, `prepare_stage2_unmark_input` (with the Audit-059 tone guard), `extract_stage2_unmark_representations` |
| `unmark.evaluation.stage2_head_campaign` | cache key, head runner, artifact, 2x5 plan, `STAGE2_MEASUREMENT_CORRUPTION_SEED = 19225` |
| `unmark.evaluation.stage2_campaign` | manifest / registry / runner shape |
| `unmark.evaluation.preg1_head` | `build_head`, `build_optimizer`, `deterministic_batches`, `select_checkpoint`, `require_full_schedule`, `score_predictions`, `Preg1Role` |

Two facts from that audit shaped everything that followed.

**(a) `Preg1Role` has no `OFFICIAL_TEST` member.** Official TEST is unreachable
by *construction*, not by a check. Reusing that enum is therefore the strongest
available TEST seal, and the V2-SCF modules reuse it rather than defining roles.

**(b) `reconstruct.py` already exists and already anticipated this exact
moment.** Its module docstring says it exists "so that the C1 checkpoints written
now remain correctly reconstructable later," and names `adapter_for_provenance`
as "the API a later Stage-2 loader calls." This audit is that later Stage-2
loader, and it calls exactly that API.

---

## 4. Files changed

**Added (7):**

| File | Lines | Role |
|---|---|---|
| `docs/spec/stage2-v2-scf-posthoc-protocol.json` | 646 | the machine-readable freeze |
| `unmark/evaluation/stage2_scf_pathway.py` | 1074 | identity, verification, fusion dispatch, frozen forward |
| `unmark/evaluation/stage2_scf_campaign.py` | 1476 | cache, head runner, artifacts, five-seed plan, registry |
| `unmark/evaluation/stage2_scf_measurement.py` | 326 | measurement — separated and gated |
| `tests/test_stage2_scf_pathway.py` | 844 | pathway tests |
| `tests/test_stage2_scf_campaign.py` | 1146 | campaign + measurement tests |
| `docs/audits/071-…md` | this file | the audit |

**Modified (1), and deliberately reported rather than glossed:**

`tests/test_preg1_import_contract.py` — three entries appended to `IMPORTERS`.

This is **not** an incidental edit; it is the repository's own guard firing
correctly. That file pins every module which imports `preg1_protocol` symbols so
that a commit cannot ship an importer whose protocol symbols are missing from the
committed tree — the exact defect that killed a Colab run after Audit 027. The
three new V2-SCF modules import protocol constants (rather than restating them,
which is what this audit wanted), so the suite failed with
`importer list is stale; missing [...]` until they were declared. Declaring them
**widens** the contract to cover the new modules; it weakens nothing, and no
production file is involved.

The sibling test that compares each importer against the **committed** protocol
module now reports 3 skips, because the new files are not yet at `HEAD`. Those
skips resolve into real checks as soon as the author commits.

---

## 5. Exact SCF checkpoint identity

Bound in `stage2_scf_pathway.V2_SCF_CHECKPOINT` and in the freeze artifact:

```text
pathway_id                 UNMARK-V2-SCF          (NOT a Stage2UnmarkArm)
stage                      v2_scf
source_repository_head     8de83f0da8d2b38f312e2ff16cb78ebcf9e8e526
stage artifact             v2_scf.json
stage artifact sha256      19a5bf4cb4b7793bee9e51292f62e7b42baf38c64026ad52fc177de40fb05695
checkpoint                 training-checkpoint-best.pt
checkpoint sha256          a32c0167817d457d5067c2a351f2d1b73b26229033f03727f43e2d79f59ef685
selected update            8000
held-out worst-case score  0.08116862134875966
objective_id               align-clean-pooled-v1        (the HISTORICAL objective)
fusion_id                  scale-calibrated-fusion-v1   (what makes C1 C1)
run_seed                   36930   (from protocol.V2_SCF_RUN_SEED, not restated)
init_seed                  51800   (derived via protocol.adapter_init_seed)
learning_rate              1e-4    (from protocol.V2_SCF_LEARNING_RATE)
r                          1.0     (from protocol.V2_SCF_R)
durable path (OPERATIONAL) stage1-training/8de83f0da8d2-g4-blackwell-unified-wandb-
                           20260912T010344Z/v2-scf/run-seed36930/_checkpoint/
                           training-checkpoint-best.pt
```

`run_seed`, `init_seed`, `learning_rate` and `r` are **read from the
repository's own V2-SCF run plan**, never typed in as literals, so the Stage-2
identity cannot drift from the Stage-1 plan that produced the checkpoint. The
durable Drive path is recorded as operational provenance and is verified against
nothing; the scientific binding is the sha256, checked against the bytes.

### Why `update=8000` is not a downstream choice

The checkpoint was selected **inside Stage-1**, by the locked held-out rule
(`selection.select_checkpoint`: lowest worst-case score over the fixed condition
grid, then lower `d_clean`, then earliest update). No downstream label, Macro-F1
or Stage-2 score took part, and nothing in the added code can revisit it —
`verify_scf_checkpoint` *compares* `global_update` against 8000 and refuses
anything else, including another update of the same run.

---

## 6. Exact fusion reconstruction mechanism

### The risk, precisely

A V2-SCF `adapter_state` holds the same eight tensors, with the same names and
the same shapes, as a historical adapter — a candidate fusion adds no parameter.
It therefore loads into the **historical** mixture rule under `strict=True` with
no error at all, producing a model that is numerically wrong and structurally
perfect. The equations differ only between the LayerNorm and the gate:

```text
historical-fusion-v1        z = g * f       + (1-g) * e
scale-calibrated-fusion-v1  scale = ||e|| / clamp(||f||, 1e-8)
                            f_cal = scale * f
                            z     = g * f_cal + (1-g) * e
```

### The mechanism

**The equation is not restated anywhere in the Stage-2 layer.** The adapter is
built through the Stage-1 dispatch that already exists:

```text
payload
  -> require_loadable_as(payload, "scale-calibrated-fusion-v1")   # refuse first
  -> reconstruct_adapter(payload, HIDDEN_SIZE)
       -> recorded_fusion_id(provenance)                          # read, not assumed
       -> fresh_adapter(hidden, seed, fusion_id)
            -> AdapterConfig(fusion_id="scale-calibrated-fusion-v1")
            -> OrthographyInputAdapter                            # config.is_scale_calibrated
       -> load_adapter_state(..., strict=True)
  -> post-check: adapter.config.fusion_id == scale-calibrated-fusion-v1
```

`unmark.modeling.adapter.scale_calibrated_fusion` stays the single
implementation of the C1 equation, reached the same way Stage-1 training reached
it. Two tests enforce the non-duplication: one asserts by AST that
`reconstruct_adapter` and `require_loadable_as` are called while
`OrthographyInputAdapter` and `AdapterConfig` are **not**; another asserts that
no `.norm(...)` computation appears anywhere in the Stage-2 pathway module.

### Four independent gates on the checkpoint

`verify_scf_checkpoint` refuses on any one of these, in order, each sufficient
alone:

1. **bytes** — `sha256_file` vs the frozen digest (runs before torch is imported);
2. **architecture** — `require_loadable_as(payload, scale-calibrated-fusion-v1)`,
   reading the checkpoint's own recorded `provenance.fusion`. A provenance with
   **no** fusion block is read as the historical claim and is refused here too;
3. **experiment** — `trainer.verify_checkpoint` against a freshly constructed
   `RunProvenance`, comparing all twelve scientific identity fields plus the
   `objective` and `fusion` blocks and the two derived objective weights;
4. **selection** — `global_update` vs the frozen 8000.

Gate 3 is the repository's own authoritative comparator, reused rather than
re-implemented, so this verifier cannot drift weaker than the contract a Stage-1
resume must satisfy. A live-module check also runs on every forward:
`require_frozen_scf_pathway` verifies `adapter.config.is_scale_calibrated` on the
**module**, not on a string in a binding.

---

## 7. Scientific decisions: reused vs newly introduced

### Reused UNCHANGED from the corrected historical Stage-2 protocol

Nothing in this list was re-derived, re-chosen or retuned:

| Value | Source |
|---|---|
| dataset UIT-VSFC 1.0, sentiment, 3 labels | `preg1_protocol` |
| protocol-train 9139 / protocol-dev 2285 | frozen split |
| representation FIRST_TOKEN, 768, FP32, no AMP | `STAGE2_FIRST_TOKEN_POOLING`, `STAGE2_REPRESENTATION_DTYPE` |
| encoder `vinai/phobert-base` @ `01daacda…` frozen, eval | `preg1_protocol` |
| head `Linear(768,3)`, Xavier-uniform weight, zero bias | `preg1_head.build_head` |
| AdamW, LR 0.01, wd 0.01 weight / 0.0 bias, betas (0.9,0.999), eps 1e-8 | `preg1_head.build_optimizer` |
| batch 128, 30 complete epochs, no early stopping, no scheduler, no clipping, no accumulation | `preg1_protocol`, `require_full_schedule` |
| unweighted cross entropy, no label smoothing | `nn.CrossEntropyLoss()` |
| the five frozen seeds `53148, 59945, 42941, 720, 9428` | `MEASUREMENT_SEEDS` |
| batch order `seed * 1000 + epoch` | `deterministic_batches` |
| selection: Macro-F1 -> Accuracy -> earliest epoch | `preg1_head.select_checkpoint` |
| clean-only training and selection | `require_clean_condition` |
| measurement conditions FULL/P25/P50/P75/P100/STRIP_ALL | `STAGE2_UNMARK_CONDITIONS` |
| measurement corruption seed **19225** | `STAGE2_MEASUREMENT_CORRUPTION_SEED` (D-S2-002) |
| roles PROTOCOL_TRAIN / PROTOCOL_DEV / OFFICIAL_VALIDATION | `Preg1Role` (no TEST member) |
| corrected tone-channel input path | `prepare_stage2_unmark_input` + `require_resolved_tone_channel` |
| the frozen forward pass | `extract_stage2_unmark_representations` |
| corpus digest and inventory pins | `finalists.CORPUS_MANIFEST_DIGEST`, `INVENTORY_*` |

Reusing seed 19225 rather than drawing a new realisation is deliberate: a new
one could not be shown not to have been cherry-picked.

A torch test asserts this reuse is real rather than claimed —
`train_scf_head` and the historical `train_stage2_head` produce **identical**
epoch scores, identical selected epoch, identical initial-head fingerprint and
identical selected-head digest on identical inputs at the same seed.

### Newly introduced (and why each was unavoidable)

| New thing | Why it could not be reused |
|---|---|
| `V2_SCF_PATHWAY_ID = "UNMARK-V2-SCF"` | the historical arm enum is frozen at two members; a third would change what the A/B campaign measured |
| `stage2-v2-scf-posthoc-protocol-v1` freeze | a post-hoc campaign must have its own frozen protocol, bound before results |
| `ScfCheckpointIdentity` / `verify_scf_checkpoint` | `verify_finalist_checkpoint` takes a `FinalistIdentity`; V2-SCF is not a finalist |
| `ScfRepresentationKey` | must bind `fusion_id`, the Stage-1 **and** Stage-2 commits, and the selected update — fields the A/B key has no place for |
| `ScfRepresentationCache` (own filenames) | so a path typo cannot cross the two artifact families |
| `train_scf_head` | the role types are pathway-bound; the A/B runner requires a `Stage2UnmarkArm` |
| `SCF_MEASUREMENT_AUTHORISATION` | the historical campaign had no explicit measurement boundary token |
| `posthoc_exploratory` on every artifact | the historical campaign was not post-hoc; these numbers must carry their status |

**No new seed, no new hyperparameter, no new selection rule, no new corruption
realisation, and no new metric was introduced.**

---

## 8. Why the historical dual-finalist code was not mutated

Five reasons, in descending order of severity.

1. **Adding a third `Stage2UnmarkArm` would silently rewrite the frozen
   protocol.** Audit 049 froze *exactly two* arms under option (c) — both
   carried, neither selected. `require_paired_campaign_plan`,
   `validate_stage2_campaign_manifest` and `aggregate_stage2_campaign` all
   enforce "both arms present, matched pairs unbroken". A third member would make
   the 2x5 plan a 3x5 plan, make every "both arms" check ambiguous, and make the
   ten completed corrected heads describe a campaign that no longer exists.

2. **The completed corrected clean campaign is evidence.** Audit 061 closed out
   ten heads and four caches at execution HEAD `cb78114e…`. Changing the schema,
   the key type or the arm universe would invalidate artifacts already on disk
   that no longer can be re-derived without re-running them.

3. **A post-hoc exploratory candidate must not be able to contaminate a closed
   comparison.** If V2-SCF were an arm, `aggregate_stage2_campaign` would compute
   paired deltas against it, and a reader could rank a post-hoc candidate into a
   campaign that deliberately declares no winner.

4. **Mutual unreadability is only achievable with separate schemas.** The
   requirement "historical A/B cache/head artifacts must be impossible to reuse
   as SCF" cannot be met by a shared key type with an extra field — a shared type
   parses both. Two closed schemas that each reject the other's payload can.

5. **`stage2_dual_finalist.py` must keep building the DEFAULT adapter.** Its
   loader calls `AdapterConfig(hidden_size=HIDDEN_SIZE)` — historical fusion, by
   omission. Adding a `fusion_id=` parameter "to support V2-SCF" would put a
   fusion switch on the path every UNMARK-A/B representation flows through. A
   test asserts by source inspection that the string `fusion_id` does not appear
   in that module at all.

**What was reused from the historical modules without changing them:**
`prepare_stage2_unmark_input`, `collate_stage2_unmark_batch`,
`require_resolved_tone_channel`, `extract_stage2_unmark_representations`,
`require_stage2_condition`, `require_batch_provenance`, `require_clean_condition`,
`require_role`, `require_head_only_optimizer`, `label_digest`, and every
`preg1_head` primitive. Reuse is by import and by duck-typing; not one line of
those modules was edited.

### How `extract_stage2_unmark_representations` is reused without a subclass

`FrozenScfPathway` is deliberately **not** a subclass of `FrozenUnmarkPathway` —
inheriting it would carry a `Stage2UnmarkArm` and would pass every `isinstance`
check the historical code performs. It is instead *structurally* compatible where
the forward pass actually touches it (`.encoder`, `.adapter`, `.require_frozen()`),
so the accepted Audit-050 forward runs over it unchanged, with the V2-SCF
pathway's own freeze check supplying `require_frozen`. One forward pass exists in
the repository; both families use it.

---

## 9. Test results

**Full suite, project `.venv` (ML-free — torch is not installed there):**

```text
before this work   4732 passed, 242 skipped
after  this work   4842 passed, 262 skipped, 0 failed   (162.64s)
```

`+110 passed, +20 skipped`. The 20 new skips are the torch-gated V2-SCF tests
(17) plus the 3 committed-tree checks for the new modules, which skip until the
author commits them.

**The 17 torch-gated tests were NOT left unverified.** A throwaway CPU-torch
environment was built in the session scratchpad — `torch 2.14.0+cpu`, entirely
outside the project and leaving `.venv` untouched — and both V2-SCF suites were
executed there with real tensors:

```text
tests/test_stage2_scf_pathway.py + tests/test_stage2_scf_campaign.py
    124 passed, 0 skipped, 0 failed        (real torch)
```

So every V2-SCF test in this audit has actually executed, including the adapter
construction, the frozen-parameter checks, the `[B, 768]` FP32 representation
check, and the head-protocol equivalence proof.

**Historical Stage-2 regression suite, also under real torch:**

```text
test_stage2_dual_finalist_infra.py, test_stage2_head_campaign.py,
test_stage2_head_campaign_torch.py, test_stage2_campaign.py,
test_stage2_campaign_torch.py, test_stage2_tone_channel_regression.py
    247 passed, 4 skipped (CUDA-gated), 0 failed
```

The historical Stage-2 contract is untouched and fully green with torch present.

### One failure was hit during development, and it was the right one

The first full run after adding the modules failed
`test_preg1_import_contract.py::test_the_importer_list_is_complete` with
`importer list is stale; missing [the three new modules]`. That guard exists
because a past commit shipped importers whose protocol symbols were left behind,
killing a Colab run with `ImportError`. It fired exactly as designed, and the fix
was to declare the new importers (§4) — not to weaken the check.

### Required proofs and where each is proven

| Required proof | Test |
|---|---|
| SCF loader refuses historical fusion | `test_scf_adapter_builder_refuses_a_historical_fusion_checkpoint`, `..._with_no_fusion_block`, `test_scf_checkpoint_historical_fusion_fails_closed` |
| historical A/B loaders unchanged | `test_historical_dual_finalist_module_is_untouched_by_this_work`, `test_historical_head_campaign_still_declares_exactly_two_arms`, `test_historical_verifier_refuses_an_scf_provenance` |
| SCF checkpoint digest mismatch fails closed | `test_scf_checkpoint_digest_mismatch_fails_closed`, `test_scf_checkpoint_sha_mismatch_fails_closed_in_binding` |
| SCF selected update mismatch fails closed | `test_scf_checkpoint_update_mismatch_fails_closed`, `..._in_binding` |
| SCF source HEAD mismatch fails closed | `test_scf_checkpoint_source_head_mismatch_fails_closed`, `..._in_binding` |
| cache identity differs from A/B and from another checkpoint/fusion | `test_scf_cache_key_and_historical_key_are_mutually_unreadable`, `..._refuses_the_historical_fusion`, `..._refuses_another_checkpoint`, `..._refuses_another_selected_update`, `..._refuses_another_stage1_source_head`, `..._differs_when_the_stage2_commit_differs`, plus both cache-directory rejection tests |
| SCF pathway really constructs scale-calibrated fusion | `test_scf_pathway_really_constructs_scale_calibrated_fusion`, `test_scf_adapter_forward_differs_from_the_historical_adapter`, `test_a_historical_fusion_adapter_on_the_scf_pathway_fails_closed`, `test_scf_pathway_builds_the_adapter_through_stage1_dispatch` |
| all encoder/adapter params frozen | `test_every_encoder_and_adapter_parameter_is_frozen`, `test_an_unfrozen_parameter_fails_closed`, `test_a_training_mode_module_fails_closed` |
| representation is FIRST_TOKEN `[B,768]`, detached FP32 | `test_representation_is_first_token_768_detached_fp32` |
| corrected tone classifier present | `test_scf_reuses_the_corrected_stage2_input_path`, `test_classifierless_preparation_still_fails_closed`, `test_frozen_protocol_requires_the_syllable_inventory_classifier` |
| campaign plan is exactly five SCF seeds | `test_campaign_plan_is_exactly_five_scf_seeds`, `..._refuses_a_dropped_seed`, `..._an_extra_seed`, `..._a_duplicated_seed`, `..._a_historical_arm_run` |
| exactly 30 epochs required | `test_thirty_epochs_are_required`, `test_no_early_stopping_or_scheduler_reaches_the_scf_head`, `test_scf_head_training_refuses_an_unfrozen_seed_and_a_retuned_lr` |
| no best-seed selection | `test_no_best_seed_selection_exists`, `test_no_best_seed_helper_can_be_reached_by_name`, `test_aggregate_refuses_a_report_missing_a_seed`, `test_aggregate_reports_all_five_seeds_and_no_winner` |
| official validation cannot enter head training/selection | `test_training_extraction_plan_has_no_measurement_role`, `test_clean_campaign_cache_slots_exclude_official_validation`, `test_campaign_manifest_refuses_an_official_validation_cache_slot`, `test_campaign_manifest_refuses_a_measurement_role_in_a_clean_slot`, `test_scf_head_training_refuses_an_official_validation_tensor` |
| official TEST cannot be represented or routed | `test_official_test_has_no_role_member`, `test_no_scf_module_can_name_or_route_official_test`, `test_scf_modules_take_no_split_argument`, `test_scf_safety_flags_declaring_test_are_all_false` |
| V2-SCF is not a third arm | `test_scf_pathway_id_is_not_a_historical_arm`, `test_historical_arm_universe_is_still_exactly_two`, `test_scf_pathway_refuses_a_historical_arm_name` |
| the head protocol is the historical one, unchanged | `test_scf_head_training_matches_the_historical_head_protocol_exactly`, `test_frozen_protocol_spec_reuses_the_corrected_historical_head_protocol`, `test_campaign_reuses_the_locked_head_primitives` |

---

## 10. Remaining Colab / runtime gates

G1 was satisfied locally (§9); **G2-G8 were not executed here** and each must
pass in the GPU runtime before any real Stage-2 work.

**G1 — torch-gated tests: SATISFIED LOCALLY, still worth re-running on GPU.**
All 17 were executed against real CPU torch and passed (§9). They have **not**
been run on CUDA, and the CUDA-gated tests in the historical suite (4 skips)
remain for the GPU runtime. Re-run both V2-SCF suites in Colab before any real
extraction, so the tensor contracts are confirmed on the actual device.

**G2 — checkpoint materialisation and verification.** The checkpoint is on
Drive, not in this checkout. `verify_scf_checkpoint(path)` must be run against
the real file and must return evidence whose `checkpoint_sha256` is
`a32c0167…`, `update` is `8000` and `fusion_id` is `scale-calibrated-fusion-v1`.

**G3 — corpus digest agreement.** `expected_scf_run_provenance` binds
`CORPUS_MANIFEST_DIGEST = 250859a5…`, the one prepared corpus every Stage-1 run
used. If the real V2-SCF checkpoint's provenance records a different corpus
digest, gate 3 of the verifier will refuse it by name. **That refusal is the
correct outcome, not a bug to work around** — it would mean C1 trained on
different data than the historical runs and the pairing argument would not hold.
Report it rather than relaxing the check.

**G4 — pinned inventory.** `resolve_inventory()` fails closed unless the pinned
Vietnamese syllable inventory is provisioned
(`scripts/fetch_vietnamese_syllable_inventory.py`). The same inventory must be
wired into input preparation as `make_classifier(load_inventory())` — see G5.

**G5 — the corrected tone channel, at runtime.** `classifier=None` is refused by
`require_resolved_tone_channel`, but only when preparation is actually called
with the real classifier does the tone channel carry signal. The Audit-059
defect was a runtime wiring mistake, not a code defect; the notebook must pass a
real classifier, and the operator should confirm that `FULL` and `P100`
`tone_ids` differ on a sample row before caching anything.

**G6 — real PhoBERT identity.** `require_stage2_encoder_identity` must pass
against the real `vinai/phobert-base` at revision `01daacda…`.

**G7 — Stage-2 execution HEAD.** Caches and heads bind
`stage2_repository_head`. That must be the commit the author creates from this
work, and it must not be confused with the Stage-1 source HEAD (which happens to
equal the current checkout's HEAD).

**G8 — stage artifact.** `verify_scf_stage_artifact(v2_scf.json)` should be run
where the file is reachable, to confirm `19a5bf4c…`, update 8000 and score
0.08116862134875966 against the committed freeze.

---

## 11. Is real Stage-2 training authorised?

**NO.**

The infrastructure is implemented and frozen, but authorisation is a separate
act and the following are all still open:

* **G2-G8 above have not been executed.** G1 is satisfied on CPU torch, but no
  real checkpoint has been verified, the real PhoBERT identity has not been
  checked, and nothing has run on CUDA.
* **The protocol freeze is not yet committed.** `require_frozen_scf_protocol_spec`
  reads `docs/spec/stage2-v2-scf-posthoc-protocol.json` from the working tree.
  Until the author commits it, the freeze is not durable evidence that the
  protocol predated the results, which is the entire point of freezing it first.
* **The Stage-2 execution HEAD does not exist yet.** Every cache key must bind
  it, and it is created by the author's commit.
* **Author authorisation has not been given.** This audit asks for it; it does
  not assume it.

When those clear, the order is fixed and the code enforces it: extract the two
**clean** caches (protocol-train, protocol-dev) -> run the five-seed clean
campaign -> all five heads complete -> and only then, at an explicitly
authorised boundary, measurement.

**Measurement remains a later phase.** `require_scf_measurement_authorised`
refuses unless *both* the five head artifacts exist on disk **and** the caller
passes `SCF_MEASUREMENT_AUTHORISATION` verbatim. Official validation is read
nowhere else, and official TEST has no role, no argument and no pathway.

---

## 12. Scientific status statement

This is **post-hoc exploratory** UNMARK-v2 work. Official UIT-VSFC validation
has already been seen historically, so nothing produced under this pathway is
untouched confirmatory evaluation, and every artifact it writes — cache key,
head artifact, campaign manifest, measurement report — carries
`posthoc_exploratory: true`. Official UIT-VSFC TEST remains **SEALED**. No
Stage-2 hyperparameter may be retuned after seeing V2-SCF results; the freeze
records that prohibition and the runner enforces the values it names.

**No commit and no push was performed during this work.**
