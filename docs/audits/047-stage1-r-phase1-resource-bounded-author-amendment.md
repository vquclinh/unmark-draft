# Audit 047 - Stage-1 r-phase1 Resource-Bounded Author Amendment

**Scope:** add a fail-closed handoff path for the stopped `r-phase1` sweep.
**Date:** 2026-09-05

## 1. Scientific State

The Stage-1 `r-phase1` plan originally specified five candidates
`r in {0.25, 0.5, 1, 2, 4}` at the frozen LR, with the normal 20,000-update
initial cap. The fused resource-bounded run was intentionally stopped after all
five candidates had:

- validation at update 6500;
- durable `training-checkpoint-last.pt` state at update 6500;
- training progress to update 6500.

No completed 20,000-update `r_phase1.json` is claimed or fabricated here.

## 2. Amendment

The adopted downstream value is:

| | |
|---|---|
| fixed LR | `0.0001` |
| selected r | `1.0` |
| amendment kind | `author_r_override_after_resource_bounded_validation_review` |
| observed cutoff | `6500` |
| original planned cap | `20000` |
| comparison window | `[4000, 4500, 5000, 5500, 6000, 6500]` |
| primary criterion | lower median validation/score |
| secondary diagnostic | lower median `d_clean` |
| stability diagnostics | `score_range`, `score_std` |

This is a resource-bounded author amendment after partial-curve observation. It
does not say that the old completed-sweep selector finished, and it does not
claim `r=1` is globally optimal.

## 3. Observed-Window Result

Resource-bounded order over the observed window:

1. `r=1`
2. `r=0.5`
3. `r=2`
4. `r=0.25`
5. `r=4`

The historical `r=1` LR-pilot tail after update 6500 is not selection evidence.
It is excluded by the reissue helper and by the artifact representation, which
sets `historical_tail_after_cutoff_used = false`.

## 4. Implementation

`unmark.stage1.artifact.validate_selection_artifact` still supports the normal
locked `r-phase1` artifact path: without an override, it recomputes
`select_r(candidates, frozen_lr)` and refuses edited winners or edited evidence.

For `r_phase1` only, the validator now accepts a closed-schema
`selection_override` block when:

- `kind` is exactly
  `author_r_override_after_resource_bounded_validation_review`;
- the original locked selector name and original planned 20,000-update cap are
  preserved;
- the observed cutoff and comparison window are exactly the documented values;
- top-level `official_test_used` and `downstream_score_used` are false;
- all five candidate summaries are present;
- each summary carries the validation points for the observed window;
- medians, means, range, standard deviation and cutoff score recompute exactly
  from those points;
- the documented resource-bounded rule selects `r=1`;
- the top-level selected candidate matches that override;
- the `r=1` fused run matches the historical LR-pilot `lr=0.0001,r=1` control
  over the tested window with max absolute validation-metric difference `0`.

`unmark/stage1/r_phase1_amendment.py` is the reusable helper core. It loads the
five read-only last checkpoints, verifies their provenance against the expected
source campaign, verifies update-6500 durability and the complete comparison
window, recomputes the summaries, validates the control equivalence, builds the
new artifact under the current repository HEAD, and immediately validates the
artifact through the same consumer that `final-main` calls.

`docs/colab/regenerate_r_phase1_resource_bounded_author_override_cell.py` is the
copyable Colab cell. It reissues the LR handoff under the same current HEAD,
then reissues `r_phase1.json` from the stopped checkpoints. It does not start
`final-main`.

## 5. Preserved Protocol

Unchanged:

- frozen `vinai/phobert-base` at
  `01daacda68afe13d83023d16ec647239e344a1e6`;
- batch size `128`;
- `max_length = 256`;
- fp32 deterministic CUDA policy;
- corruption seed, validation seed and protocol identity;
- official UIT-VSFC TEST seal;
- downstream score exclusion;
- final-main three-seed design.

## 6. Tests

Targeted validation:

```text
pytest -q tests/test_stage1_runner_contract.py \
  tests/test_stage1_r_phase1_amendment.py \
  tests/test_stage1_artifact_identity.py \
  tests/test_stage1_fused_r_phase1.py \
  tests/test_stage1_schedule.py \
  tests/test_stage1_runner_cli_contract.py \
  tests/test_stage1_repository_provenance.py \
  tests/test_stage1_device_contract.py

177 passed, 1 skipped in 0.81s
```

No Stage-1 training command was run. Existing checkpoints are not modified by
the tests.
