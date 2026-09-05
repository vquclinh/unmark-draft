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
- fused monitoring telemetry records `run_start`, update-6500
  `train_progress`, update-6500 `validation` and update-6500 checkpoint events
  for all five candidates under the historical source campaign HEAD;
- each summary carries the validation points for the observed window;
- medians, means, range, standard deviation and cutoff score recompute exactly
  from those points;
- the documented resource-bounded rule selects `r=1`;
- the top-level selected candidate matches that override;
- `selected_r` is exactly `1.0` and `fixed_learning_rate` is exactly `0.0001`.
  Every check above proves only that the artifact is *internally* consistent,
  which a coherently edited artifact also is; these two pins are what tie the
  amendment kind to the one decision this audit records. It is deliberately not
  a general-purpose r override;
- the `r=1` fused run matches the historical LR-pilot `lr=0.0001,r=1` control
  over the tested window with max absolute validation-metric difference `0`,
  and that control's `run_seed` is exactly the selection seed `21230`. `lr` and
  `r` alone do not name the control run, so the seed is read from the
  authoritative `RunProvenance.run_seed` field rather than inferred from a file
  or directory name. It is recorded in the control-equivalence evidence and
  re-checked whenever the finished artifact is validated.

`unmark/stage1/r_phase1_amendment.py` is the reusable helper core. It loads the
five read-only last checkpoints, verifies their provenance against the expected
source campaign, verifies update-6500 durability and the complete comparison
window, verifies fused monitoring telemetry, recomputes the summaries, validates
the control equivalence, builds the new artifact under the current repository
HEAD, and immediately validates the artifact through the same consumer that
`final-main` calls.

`docs/colab/regenerate_r_phase1_resource_bounded_author_override_cell.py` is the
copyable Colab cell. Before anything is read or written it requires an explicit
full 40-hex implementation commit, supplied by the human in
`IMPLEMENTATION_COMMIT` or via `UNMARK_IMPLEMENTATION_COMMIT`, which has no
default. Branch names, tags, `HEAD`, short SHAs and the unreplaced placeholder
are all refused: the artifact records the producing commit as provenance, so a
ref that can resolve to different code on different days cannot be accepted. The
commit is fetched and checked out by SHA and `git rev-parse HEAD` must equal it.
The same rule applies to
`docs/colab/regenerate_lr_pilot_author_override_cell.py`, which this cell
`exec()`s and which would otherwise check the repository back out onto a moving
ref mid-run; it reuses the already-verified commit through
`INJECTED_IMPLEMENTATION_COMMIT`.

The cell then establishes scientific identity **independently of the
artifacts**: it verifies the pinned inventory against its expected revision,
SHA-256 and size, verifies the prepared corpus with the repository's own
`verify_prepared_corpus` (re-hashing `chunks.jsonl` and `manifest.json` and
checking the membership digest), and builds the campaign identity from that
evidence plus the verified HEAD via `CampaignIdentity.from_inputs`. Only then is
`lr_pilot.json` required to agree with it. Reading the identity out of
`lr_pilot.json` and then validating `lr_pilot.json` against that same identity
would prove only that the file is self-consistent.

`chunks.jsonl` is runtime-only: Colab deletion wipes `/content`, so when the
payload is absent the cell rebuilds it from the immutable shards with
`concatenate_shards` onto local SSD and verifies it against the SHA-256 bound by
`COMPLETE.json`. That writes only to `/content`, never to Drive.

It works exclusively under `/content/drive/MyDrive/UNMARK/UNMARK-BACKUP`, backs
up any previous handoff artifact before replacement, deletes nothing, and does
not start `final-main`.

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
  tests/test_stage1_device_contract.py \
  tests/test_stage1_r_phase1_reissue_cell.py

224 passed, 1 skipped in 0.89s
```

Whole-repository suite:

```text
pytest -q

4063 passed, 107 skipped in 150.76s
```

No Stage-1 training command was run. Existing checkpoints are not modified by
the tests.
