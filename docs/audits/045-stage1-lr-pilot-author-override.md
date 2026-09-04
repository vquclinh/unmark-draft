# Audit 045 - Stage-1 LR Pilot Author Override

**Scope:** record and implement the author's post-hoc LR-pilot override after
reviewing W&B validation curves.
**Date:** 2026-09-04

## 1. Inputs

The completed LR-pilot evidence is local under `wandb/`:

| File | Role |
|---|---|
| `wandb/run-lr0.0001.json` | full evaluation trajectory for `lr=1e-4` |
| `wandb/run-lr0.0003.json` | full evaluation trajectory for `lr=3e-4` |
| `wandb/run-lr0.001.json` | full evaluation trajectory for `lr=1e-3` |
| `wandb/all_3.png` | W&B plot view used for the curve review |
| `wandb/lr_pilot.json` | regenerated handoff artifact |

No UIT-VSFC official TEST data was opened or used.

## 2. Locked-Rule Outcome

The original LR selector minimizes the selected checkpoint's
`validation/score`, then lower `d_clean`, then lower LR.

Under that rule, `lr=0.0003` wins:

| LR | selected update | selected score |
|---|---:|---:|
| `1e-4` | 14500 | 0.09000698585438581 |
| `3e-4` | 500 | 0.08824253183326698 |
| `1e-3` | 13500 | 0.10682767755677106 |

The margin between `3e-4` and `1e-4` is small, and the winning `3e-4` point is
the first post-update evaluation.

## 3. Curve Review

The author reviewed the W&B validation curves and rejected the `3e-4` winner as
too transient. The key observed sequence for `3e-4` was:

| update | validation/score |
|---:|---:|
| 500 | 0.08824253183326698 |
| 1000 | 0.10005928909487942 |
| 1500 | 0.20531023229990536 |
| 2000 | 0.30160508679935255 |

In contrast, `1e-4` had the better later validation trajectory under stability
summaries computed from the complete run JSON:

| Threshold | LR | best | median | mean |
|---:|---|---:|---:|---:|
| `>=2000` | `1e-4` | 0.09000698585438581 | 0.10404378789856562 | 0.11973577680257691 |
| `>=2000` | `3e-4` | 0.11848742018251854 | 0.1360465747737326 | 0.167967636219052 |
| `>=2000` | `1e-3` | 0.10682767755677106 | 0.11786325569806227 | 0.11908537473010306 |
| `>=5000` | `1e-4` | 0.09000698585438581 | 0.10138155619581189 | 0.10570159833330355 |
| `>=5000` | `3e-4` | 0.11848742018251854 | 0.13348664890414486 | 0.15120857326512446 |
| `>=5000` | `1e-3` | 0.10682767755677106 | 0.11711214874358086 | 0.1183406738895201 |

This audit does not pretend those stability summaries were part of the original
locked selector. They are recorded as the basis for the author's override.

## 4. Implementation

`unmark/stage1/artifact.py` now accepts one closed-schema LR-pilot override:

`author_lr_override_after_validation_curve_review`.

The downstream validator still recomputes the old locked-rule winner from the
recorded candidates. It then accepts the override only if the artifact preserves
that superseded winner and selects exactly one real completed LR-pilot
candidate. This lets `r-phase1` continue from the existing evidence without
rerunning the three LR candidates, while keeping the post-hoc deviation visible
in the handoff artifact.

The regenerated `wandb/lr_pilot.json` therefore has:

| Field | Value |
|---|---|
| `selected.learning_rate` | `0.0001` |
| `selection_override.superseded_locked_rule_winner.learning_rate` | `0.0003` |
| `selection_override.reason` | author chose `1e-4` after W&B curve review |

## 5. Tests

Targeted validation:

```text
pytest -q tests/test_stage1_artifact_identity.py
36 passed
```

The added tests cover:

- a valid author override selecting `1e-4`;
- refusal when the override does not preserve the old locked-rule winner;
- refusal when top-level `selected` does not match the overridden candidate.
