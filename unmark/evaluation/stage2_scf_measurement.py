"""Post-hoc **V2-SCF** Stage-2 measurement: a LATER phase, behind a hard gate.

Audit 071. This module is separate from `stage2_scf_campaign` for one reason:
**measurement is the only place official UIT-VSFC validation is read**, and a
boundary that matters should be a file you have to import on purpose, not a
keyword argument on a function you were already calling.

Nothing here may run until two independent conditions both hold:

1. the clean five-head campaign is **complete** -- every frozen seed has a
   committed, validated head artifact. `ScfCampaignRegistry.require_campaign_complete`
   is what checks it, reading artifacts off disk rather than trusting a flag;
2. the caller passes :data:`SCF_MEASUREMENT_AUTHORISATION` **verbatim**. There is
   no default, no boolean, and no environment variable: crossing this boundary is
   a deliberate act that shows up in the calling code and in review.

**What measurement may do:** produce descriptive per-seed, per-condition numbers
for the six frozen conditions under the one frozen degradation realisation.

**What it may not do, and has no mechanism for:** choose a best seed, drop a
seed, rank V2-SCF against UNMARK-A or UNMARK-B, change any head, re-select any
checkpoint, retune any Stage-2 hyperparameter, or touch official TEST -- which
cannot be named here at all, because the role enum has no member for it.

**Scientific status travels with every number.** Official validation has already
been seen historically, so every report this module builds carries
`posthoc_exploratory: true` and `confirmatory: false`. A V2-SCF measurement is
exploratory evidence about a candidate, never a blind confirmatory result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from unmark.evaluation.metrics import per_class_scores
from unmark.evaluation.preg1_head import score_predictions
from unmark.evaluation.preg1_protocol import PRIMARY_NUM_LABELS
from unmark.evaluation.stage2_dual_finalist import (
    STAGE2_EXCLUDED_CONDITIONS,
    STAGE2_UNMARK_CONDITIONS,
    require_stage2_condition,
)
from unmark.evaluation.stage2_head_campaign import (
    STAGE2_CLEAN_CONDITION,
    STAGE2_DEGRADED_CONDITIONS,
    STAGE2_MEASUREMENT_CORRUPTION_SEED,
    label_digest,
)
from unmark.evaluation.stage2_scf_campaign import (
    SCF_CAMPAIGN_SEEDS,
    SCF_MEASUREMENT_ROLE,
    V2_SCF_CAMPAIGN_SCHEMA_VERSION,
    ScfBoundRepresentations,
    ScfCampaignManifest,
    ScfCampaignRegistry,
    ScfExtractionRequest,
)
from unmark.evaluation.stage2_scf_pathway import (
    OFFICIAL_TEST_USED,
    OFFICIAL_VALIDATION_PREVIOUSLY_SEEN,
    POSTHOC_EXPLORATORY,
    V2_SCF_PATHWAY_ID,
    ScfPathwayViolation,
    require_frozen_scf_protocol_spec,
)

SCF_MEASUREMENT_AUTHORISATION = "v2-scf-posthoc-measurement-boundary-authorised"
"""The exact string a caller must pass to cross the measurement boundary.

A string rather than a boolean on purpose: `authorised=True` is what a caller
writes by reflex when a function complains, and it reads identically whether the
author considered the boundary or not. This has to be typed, and it names what is
being authorised.
"""

SCF_MEASUREMENT_SCHEMA_VERSION = "stage2-v2-scf-measurement-v1"

SCF_MEASUREMENT_MAY_SELECT = False
SCF_MEASUREMENT_MAY_RANK_AGAINST_ARMS = False
SCF_MEASUREMENT_BEST_SEED_RULE = None
SCF_MEASUREMENT_WINNER = None
SCF_OFFICIAL_TEST_REACHABLE = False


def require_scf_measurement_authorised(
    manifest: ScfCampaignManifest,
    registry: ScfCampaignRegistry,
    *,
    authorisation: str,
) -> dict[str, Any]:
    """The measurement boundary. **Both** conditions, or nothing runs.

    Returns a gate record naming what was checked, so an execution log shows the
    boundary being crossed deliberately rather than a function quietly returning.
    """
    require_frozen_scf_protocol_spec()
    if authorisation != SCF_MEASUREMENT_AUTHORISATION:
        raise ScfPathwayViolation(
            "the V2-SCF measurement boundary requires the explicit authorisation string "
            f"{SCF_MEASUREMENT_AUTHORISATION!r}; got {authorisation!r}. Official UIT-VSFC "
            "validation is read only at an authorised boundary, after the clean five-head "
            "campaign is complete."
        )
    # Reads the five head artifacts off disk; a flag cannot satisfy this.
    registry.require_campaign_complete(manifest)
    status = registry.campaign_status(manifest)
    return {
        "kind": "v2_scf_measurement_boundary",
        "schema_version": SCF_MEASUREMENT_SCHEMA_VERSION,
        "pathway_id": V2_SCF_PATHWAY_ID,
        "clean_campaign_complete": True,
        "completed_seeds": status["completed_seeds"],
        "manifest_digest": manifest.digest,
        "posthoc_exploratory": POSTHOC_EXPLORATORY,
        "official_validation_previously_seen": OFFICIAL_VALIDATION_PREVIOUSLY_SEEN,
        "official_test_used": OFFICIAL_TEST_USED,
        "may_select": SCF_MEASUREMENT_MAY_SELECT,
        "may_rank_against_historical_arms": SCF_MEASUREMENT_MAY_RANK_AGAINST_ARMS,
    }


def scf_measurement_extraction_plan(
    manifest: ScfCampaignManifest,
    registry: ScfCampaignRegistry,
    *,
    corruption_seed: int,
    authorisation: str,
) -> tuple[ScfExtractionRequest, ...]:
    """measurement-dev x the six frozen conditions. **Six items, gated.**

    `corruption_seed` has no default and exactly one accepted value: the frozen
    :data:`STAGE2_MEASUREMENT_CORRUPTION_SEED`, reused unchanged from the
    corrected historical protocol. Keeping it required *and* checked is
    deliberate -- a default would let an execution use the frozen realisation
    without naming it, and accepting any integer would let a caller substitute a
    different degradation realisation and report it as the protocol's.
    """
    require_scf_measurement_authorised(manifest, registry, authorisation=authorisation)
    if isinstance(corruption_seed, bool) or not isinstance(corruption_seed, int):
        raise ScfPathwayViolation(
            "the Stage-2 measurement corruption seed must be an explicit integer; it is "
            "frozen and must not acquire a default"
        )
    if corruption_seed != STAGE2_MEASUREMENT_CORRUPTION_SEED:
        raise ScfPathwayViolation(
            f"the Stage-2 measurement corruption seed is frozen to "
            f"{STAGE2_MEASUREMENT_CORRUPTION_SEED} (D-S2-002, reused unchanged by this "
            f"post-hoc campaign) and may not be overridden; got {corruption_seed!r}."
        )
    return tuple(
        ScfExtractionRequest(
            pathway_id=V2_SCF_PATHWAY_ID,
            role=SCF_MEASUREMENT_ROLE.value,
            condition=condition,
            corruption_seed=(
                None if condition == STAGE2_CLEAN_CONDITION else corruption_seed
            ),
            purpose="measurement_reporting",
        )
        for condition in STAGE2_UNMARK_CONDITIONS
    )


@dataclass(frozen=True)
class ScfConditionScore:
    pathway_id: str
    seed: int
    condition: str
    macro_f1: float
    accuracy: float
    per_class_f1: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "pathway_id": self.pathway_id,
            "seed": self.seed,
            "condition": self.condition,
            "macro_f1": self.macro_f1,
            "accuracy": self.accuracy,
            "per_class_f1": list(self.per_class_f1),
        }


def measure_scf_head(
    head: Any,
    measurement: ScfBoundRepresentations,
    measurement_labels: Sequence[int],
    *,
    seed: int,
) -> ScfConditionScore:
    """Score a **frozen** selected head on measurement-dev. Reporting only.

    The role is read from the tensor's own key, so a protocol-dev tensor cannot be
    scored here as if it were measurement, and a measurement tensor cannot reach
    checkpoint selection. This returns numbers and holds no reference to any
    selection state; there is nothing here that could change a head.
    """
    import torch

    measurement.require_role(SCF_MEASUREMENT_ROLE, "V2-SCF measurement")
    require_stage2_condition(measurement.condition)
    if seed not in SCF_CAMPAIGN_SEEDS:
        raise ScfPathwayViolation(f"seed {seed!r} is not a frozen campaign seed")
    if measurement.key.label_digest != label_digest(measurement_labels):
        raise ScfPathwayViolation(
            "measurement labels do not match the cached digest"
        )
    head.eval()
    with torch.no_grad():
        predictions = head(measurement.values).argmax(dim=1).tolist()
    labels = [int(v) for v in measurement_labels]
    f1, acc = score_predictions(predictions, labels)
    classes = per_class_scores(predictions, labels, num_labels=PRIMARY_NUM_LABELS)
    return ScfConditionScore(
        pathway_id=measurement.key.pathway_id,
        seed=seed,
        condition=measurement.condition,
        macro_f1=f1,
        accuracy=acc,
        per_class_f1=tuple(c.f1 for c in classes),
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _stdev(values: Sequence[float]) -> float:
    import statistics

    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def aggregate_scf_campaign(scores: Sequence[ScfConditionScore]) -> dict[str, Any]:
    """Descriptive V2-SCF report. **All five seeds required; none is ranked.**

    Returns per-condition mean and sd over the five seeds plus the two frozen
    robustness summaries, and returns **no winner, no best seed and no comparison
    against the historical arms** -- there is no argument by which one could be
    requested. The report carries its own scientific status so a number cannot be
    lifted out of it and presented as confirmatory.
    """
    require_frozen_scf_protocol_spec()
    pathways = {s.pathway_id for s in scores}
    if pathways != {V2_SCF_PATHWAY_ID}:
        raise ScfPathwayViolation(
            f"a V2-SCF report contains exactly {V2_SCF_PATHWAY_ID!r}, got "
            f"{sorted(pathways)}. A post-hoc candidate is reported on its own; it is not "
            "ranked into the closed historical A/B campaign."
        )

    conditions: dict[str, Any] = {}
    for condition in STAGE2_UNMARK_CONDITIONS:
        rows = [s for s in scores if s.condition == condition]
        if not rows:
            continue
        if sorted(r.seed for r in rows) != sorted(SCF_CAMPAIGN_SEEDS):
            raise ScfPathwayViolation(
                f"{condition} does not cover the frozen five seeds; a seed may never be "
                "dropped because a result looks bad"
            )
        conditions[condition] = {
            "macro_f1_mean": _mean([r.macro_f1 for r in rows]),
            "macro_f1_std": _stdev([r.macro_f1 for r in rows]),
            "accuracy_mean": _mean([r.accuracy for r in rows]),
            "accuracy_std": _stdev([r.accuracy for r in rows]),
            "per_class_f1_mean": [
                _mean([r.per_class_f1[i] for r in rows]) for i in range(PRIMARY_NUM_LABELS)
            ],
            "per_seed": [
                {"seed": r.seed, "macro_f1": r.macro_f1, "accuracy": r.accuracy}
                for r in sorted(rows, key=lambda s: SCF_CAMPAIGN_SEEDS.index(s.seed))
            ],
        }

    summaries: dict[str, Any] = {}
    strip_all = conditions.get("STRIP_ALL")
    if strip_all:
        summaries["strip_all_macro_f1_mean"] = strip_all["macro_f1_mean"]
    degraded = [
        conditions[c]["macro_f1_mean"]
        for c in STAGE2_DEGRADED_CONDITIONS
        if c in conditions
    ]
    if len(degraded) == len(STAGE2_DEGRADED_CONDITIONS):
        summaries["degraded_equal_weight_macro_f1_mean"] = _mean(degraded)

    return {
        "schema_version": SCF_MEASUREMENT_SCHEMA_VERSION,
        "campaign_schema_version": V2_SCF_CAMPAIGN_SCHEMA_VERSION,
        "pathway_id": V2_SCF_PATHWAY_ID,
        "conditions": conditions,
        "robustness_summaries": summaries,
        "excluded_conditions": list(STAGE2_EXCLUDED_CONDITIONS),
        "measurement_corruption_seed": STAGE2_MEASUREMENT_CORRUPTION_SEED,
        "posthoc_exploratory": POSTHOC_EXPLORATORY,
        "confirmatory": False,
        "official_validation_previously_seen": OFFICIAL_VALIDATION_PREVIOUSLY_SEEN,
        "official_test_used": OFFICIAL_TEST_USED,
        "best_seed": SCF_MEASUREMENT_BEST_SEED_RULE,
        "winner": SCF_MEASUREMENT_WINNER,
        "ranks_against_historical_arms": SCF_MEASUREMENT_MAY_RANK_AGAINST_ARMS,
        "note": (
            "POST-HOC EXPLORATORY. Official UIT-VSFC validation was already seen "
            "historically, so these numbers are not an untouched confirmatory result. "
            "All five seeds are reported; no seed is selected, and V2-SCF is not ranked "
            "against the closed UNMARK-A/UNMARK-B campaign."
        ),
    }


__all__ = [
    "SCF_MEASUREMENT_AUTHORISATION",
    "SCF_MEASUREMENT_BEST_SEED_RULE",
    "SCF_MEASUREMENT_MAY_RANK_AGAINST_ARMS",
    "SCF_MEASUREMENT_MAY_SELECT",
    "SCF_MEASUREMENT_SCHEMA_VERSION",
    "SCF_MEASUREMENT_WINNER",
    "SCF_OFFICIAL_TEST_REACHABLE",
    "ScfConditionScore",
    "aggregate_scf_campaign",
    "measure_scf_head",
    "require_scf_measurement_authorised",
    "scf_measurement_extraction_plan",
]
