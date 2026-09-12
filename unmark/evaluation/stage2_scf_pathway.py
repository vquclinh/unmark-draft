"""Post-hoc **V2-SCF** Stage-2 pathway: identity, verification, frozen forward.

**Additive.** Nothing here changes what `UNMARK-A` or `UNMARK-B` mean. The
historical dual-finalist contract (Audit 049/050/051) is a *closed* two-arm
protocol, and `Stage2UnmarkArm` stays a two-member enum: V2-SCF is **not** a
third arm, it is a separate post-hoc campaign with its own identity, its own
caches, its own heads and its own freeze artifact.

**Scientific status.** This is POST-HOC EXPLORATORY work. Official UIT-VSFC
validation has already been seen historically, so nothing produced under this
pathway is untouched confirmatory evaluation. Official TEST remains SEALED and
is structurally unreachable: the role enum reused here (`Preg1Role`) has no
`OFFICIAL_TEST` member, and no function in this module takes a split argument.

---

**THE correctness risk this module exists to close.**

A V2-SCF checkpoint's `adapter_state` is *shape-compatible* with the historical
adapter -- the same eight tensors, the same names, the same shapes, because a
candidate fusion adds no parameter. It therefore loads into the historical
mixture rule under `strict=True` with **no error at all**, and the resulting
model is numerically wrong while looking structurally perfect. The two equations
differ only in what happens between the LayerNorm and the gate:

    historical-fusion-v1        z = g * f       + (1-g) * e
    scale-calibrated-fusion-v1  scale = ||e|| / clamp(||f||, 1e-8)
                                f_cal = scale * f
                                z     = g * f_cal + (1-g) * e

The equation is **not restated in code here**, and must never be. The adapter is
built through the repository's own Stage-1 dispatch --
`reconstruct.reconstruct_adapter`, which reads the checkpoint's recorded
`provenance.fusion`, resolves it through `initialisation.fresh_adapter` ->
`AdapterConfig(fusion_id=...)` -> `OrthographyInputAdapter`, and then loads the
tensors with `strict=True`. `unmark.modeling.adapter.scale_calibrated_fusion` is
the one implementation of that equation in this repository, and this pathway
reaches it the same way Stage-1 training did.

`require_loadable_as` is called *before* the adapter is built, so a historical
checkpoint handed to this loader is refused on the strength of its own recorded
architecture rather than discovered later as an unexplainable number.

---

**What is reused rather than re-implemented.**

* `unmark.stage1.reconstruct` -- the authoritative fusion dispatch and the
  `strict=True` state load;
* `unmark.stage1.trainer.verify_checkpoint` -- THE checkpoint gate, which
  compares the whole `RunProvenance` including `objective` **and** `fusion`;
* `unmark.stage1.candidates.candidate_for_stage` -- the closed candidate
  register, so the objective/fusion pair this pathway expects is read from the
  same register Stage-1 trained under instead of being restated;
* `stage2_dual_finalist.extract_stage2_unmark_representations` -- the accepted
  Audit-050 frozen forward pass. It is duck-typed over `.encoder`, `.adapter`
  and `.require_frozen()`, so this pathway runs **the same forward** rather than
  a second copy that could drift from it;
* `prepare_stage2_unmark_input` / `collate_stage2_unmark_batch` -- input
  construction, which is arm-agnostic and carries the Audit-059 tone-channel
  guard that fails closed when no syllable-inventory classifier is supplied.

Torch and transformers are imported lazily, so every contract here is testable
in the ML-free local environment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from unmark.evaluation.contracts import EvaluationContractViolation
from unmark.evaluation.preg1_protocol import (
    ENCODER_CHECKPOINT,
    ENCODER_REVISION,
    PRIMARY_DATASET,
    PRIMARY_DATASET_VERSION,
    PRIMARY_NUM_LABELS,
    PRIMARY_TASK,
)
from unmark.evaluation.stage2_dual_finalist import (
    STAGE2_FIRST_TOKEN_POOLING,
    STAGE2_REPRESENTATION_DTYPE,
    STAGE2_UNMARK_ARM_NAMES,
    STAGE2_UNMARK_CONDITIONS,
    extract_stage2_unmark_representations,
    load_stage2_phobert_components,
    require_stage2_encoder_identity,
)
from unmark.stage1.candidates import candidate_for_stage
from unmark.stage1.checkpoint import sha256_file
from unmark.stage1.contracts import Stage1ContractViolation
from unmark.stage1.finalists import (
    ADAPTER_STATE_KEYS,
    ADAPTER_TENSOR_COUNT,
    CORPUS_MANIFEST_DIGEST,
    FINALISTS,
    INVENTORY_SHA256,
    INVENTORY_SOURCE_REVISION,
    INVENTORY_SIZE_BYTES,
    PRECISION,
    resolve_inventory,
)
from unmark.stage1.protocol import (
    ADAPTER_TRAINABLE_PARAMETERS,
    CORRUPTION_SEED,
    HIDDEN_SIZE,
    HISTORICAL_OBJECTIVE_ID,
    SCALE_CALIBRATED_FUSION_ID,
    STAGE1_PROTOCOL_VERSION,
    V2_SCF_LEARNING_RATE,
    V2_SCF_R,
    V2_SCF_RUN_SEED,
    V2_SCF_STAGE,
    adapter_init_seed,
)
from unmark.stage1.reconstruct import (
    PROVENANCE_KEY,
    recorded_fusion_id,
    reconstruct_adapter,
    require_loadable_as,
)
from unmark.stage1.trainer import (
    CHECKPOINT_SCHEMA_VERSION,
    RunProvenance,
    TrainerContractViolation,
    verify_checkpoint,
)


class ScfPathwayViolation(EvaluationContractViolation):
    """Raised when a V2-SCF Stage-2 identity or freeze condition is violated.

    A subclass of `EvaluationContractViolation` so existing Stage-2 callers and
    tests that catch the evaluation contract error still catch these, and so a
    caller that wants to distinguish a V2-SCF failure from a historical one can.
    """


# ---------------------------------------------------------------------------
# Campaign identity -- deliberately NOT a Stage2UnmarkArm
# ---------------------------------------------------------------------------
V2_SCF_STAGE2_SCHEMA_VERSION = "stage2-v2-scf-posthoc-v1"

V2_SCF_PATHWAY_ID = "UNMARK-V2-SCF"
"""The post-hoc pathway identity.

**Not an arm.** `require_stage2_unmark_arm(V2_SCF_PATHWAY_ID)` raises, which is
asserted by test: the historical protocol is exactly two arms, and adding a
third would rewrite what the frozen A/B campaign measured. Every V2-SCF artifact
carries this string where a historical artifact carries `arm`, so the two schemas
cannot be read as each other in either direction.
"""

V2_SCF_POSTHOC_PROTOCOL_VERSION = "stage2-v2-scf-posthoc-protocol-v1"

V2_SCF_PROTOCOL_SPEC_PATH = (
    Path(__file__).resolve().parents[2] / "docs/spec/stage2-v2-scf-posthoc-protocol.json"
)

# --- scientific-status declarations, asserted by test and bound in the freeze
POSTHOC_EXPLORATORY = True
"""This campaign is exploratory. It is not untouched confirmatory evaluation."""

OFFICIAL_VALIDATION_PREVIOUSLY_SEEN = True
"""Official UIT-VSFC validation has already been seen historically. Recorded so
no report produced from this pathway can imply a blind measurement."""

OFFICIAL_TEST_USED = False
OFFICIAL_TEST_ROLE_EXISTS = False
V2_SCF_SELECTION_AGAINST_AB_IMPLEMENTED = False
V2_SCF_BEST_SEED_SELECTION_IMPLEMENTED = False
V2_SCF_STAGE2_HYPERPARAMETER_RETUNED = False
"""No Stage-2 hyperparameter is retuned for V2-SCF. Every value is inherited
from the corrected historical Stage-2 head protocol, unchanged."""


# ---------------------------------------------------------------------------
# The one frozen Stage-1 source this pathway may consume
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ScfCheckpointIdentity:
    """The single authoritative V2-SCF Stage-1 result, as external evidence.

    Every field describes a file this repository does not contain. They are
    compared, never inferred: `verify_scf_checkpoint` hashes the real file and
    matches the recorded provenance against a freshly constructed identity, so a
    checkpoint cannot define which experiment it belongs to.
    """

    pathway_id: str
    stage: str
    source_repository_head: str
    run_seed: int
    update: int
    learning_rate: float
    r: float
    objective_id: str
    fusion_id: str
    checkpoint_name: str
    checkpoint_sha256: str
    stage_artifact_name: str
    stage_artifact_sha256: str
    held_out_worst_case_score: float
    durable_checkpoint_path: str

    @property
    def init_seed(self) -> int:
        """Derived through the protocol's own domain separation, never stored."""
        return adapter_init_seed(self.run_seed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pathway_id": self.pathway_id,
            "stage": self.stage,
            "source_repository_head": self.source_repository_head,
            "run_seed": self.run_seed,
            "init_seed": self.init_seed,
            "update": self.update,
            "learning_rate": self.learning_rate,
            "r": self.r,
            "objective_id": self.objective_id,
            "fusion_id": self.fusion_id,
            "checkpoint_name": self.checkpoint_name,
            "checkpoint_sha256": self.checkpoint_sha256,
            "stage_artifact_name": self.stage_artifact_name,
            "stage_artifact_sha256": self.stage_artifact_sha256,
            "held_out_worst_case_score": self.held_out_worst_case_score,
            "durable_checkpoint_path": self.durable_checkpoint_path,
        }


V2_SCF_CHECKPOINT = ScfCheckpointIdentity(
    pathway_id=V2_SCF_PATHWAY_ID,
    stage=V2_SCF_STAGE,
    source_repository_head="8de83f0da8d2b38f312e2ff16cb78ebcf9e8e526",
    run_seed=V2_SCF_RUN_SEED,
    update=8000,
    learning_rate=V2_SCF_LEARNING_RATE,
    r=V2_SCF_R,
    objective_id=HISTORICAL_OBJECTIVE_ID,
    fusion_id=SCALE_CALIBRATED_FUSION_ID,
    checkpoint_name="training-checkpoint-best.pt",
    checkpoint_sha256=(
        "a32c0167817d457d5067c2a351f2d1b73b26229033f03727f43e2d79f59ef685"
    ),
    stage_artifact_name="v2_scf.json",
    stage_artifact_sha256=(
        "19a5bf4cb4b7793bee9e51292f62e7b42baf38c64026ad52fc177de40fb05695"
    ),
    held_out_worst_case_score=0.08116862134875966,
    durable_checkpoint_path=(
        "stage1-training/"
        "8de83f0da8d2-g4-blackwell-unified-wandb-20260912T010344Z/"
        "v2-scf/run-seed36930/_checkpoint/training-checkpoint-best.pt"
    ),
)
"""The Stage-1 C1 result this post-hoc Stage-2 campaign is bound to.

`update=8000` is the checkpoint the **locked Stage-1 held-out rule** selected
within the single first-screen run (`selection.select_checkpoint`: lowest
worst-case score, then lower `d_clean`, then earliest update). No downstream
label, Macro-F1 or Stage-2 score took part in choosing it, and none may.

`durable_checkpoint_path` is OPERATIONAL: it records where the file lives so an
operator does not have to rediscover it, and nothing verifies against it. The
scientific binding is `checkpoint_sha256`, which is checked against the bytes.
"""


def require_scf_pathway_id(value: str) -> str:
    """Parse the post-hoc pathway identity. Fails closed on anything else.

    Deliberately rejects `UNMARK-A` and `UNMARK-B` by name: a historical arm
    reaching a V2-SCF artifact is the mistake this pathway is built to prevent,
    and it should fail here rather than produce a plausible-looking number.
    """
    if value == V2_SCF_PATHWAY_ID:
        return value
    if value in STAGE2_UNMARK_ARM_NAMES:
        raise ScfPathwayViolation(
            f"{value!r} is a frozen historical Stage-2 arm, not the post-hoc "
            f"{V2_SCF_PATHWAY_ID} pathway. The historical dual-finalist campaign is "
            "closed and exactly two arms; V2-SCF is a separate post-hoc campaign and "
            "its artifacts may never be read as one of the arms', or the other way round."
        )
    raise ScfPathwayViolation(
        f"unknown Stage-2 pathway {value!r}; this module implements exactly "
        f"{V2_SCF_PATHWAY_ID}"
    )


def require_scf_candidate_registration() -> Any:
    """The registered Stage-1 candidate for `v2_scf`, checked against the pins.

    Read from the closed candidate register rather than restated, so this
    pathway cannot expect an `(objective, fusion)` pair that Stage-1 could not
    have trained.
    """
    candidate = candidate_for_stage(V2_SCF_STAGE)
    expected = (V2_SCF_CHECKPOINT.objective_id, V2_SCF_CHECKPOINT.fusion_id)
    if candidate.identity != expected:
        raise ScfPathwayViolation(
            f"the Stage-1 register gives stage {V2_SCF_STAGE!r} identity "
            f"{candidate.identity}, but this Stage-2 pathway is frozen to {expected}. "
            "The candidate register is the single source of truth for what C1 is; a "
            "disagreement means one of them drifted and neither may be trusted."
        )
    if not candidate.fusion.is_scale_calibrated:
        raise ScfPathwayViolation(
            f"stage {V2_SCF_STAGE!r} is registered with fusion "
            f"{candidate.fusion.fusion_id!r}, which is not the scale-calibrated one"
        )
    return candidate


def require_historical_arms_untouched() -> None:
    """The historical two-arm contract is still exactly two arms, unchanged.

    Called by the V2-SCF freeze check. A post-hoc campaign that silently grew the
    frozen A/B universe would invalidate the very protocol it claims to reuse, so
    this pathway refuses to operate if that has happened.
    """
    if STAGE2_UNMARK_ARM_NAMES != ("UNMARK-A", "UNMARK-B"):
        raise ScfPathwayViolation(
            f"the historical Stage-2 arm universe is {list(STAGE2_UNMARK_ARM_NAMES)}, not "
            "the frozen ('UNMARK-A', 'UNMARK-B'). V2-SCF is additive and may not be "
            "executed against a mutated dual-finalist contract."
        )
    if len(FINALISTS) != 2:
        raise ScfPathwayViolation(
            f"the Stage-1 finalist universe holds {len(FINALISTS)} entries, not 2"
        )
    scf_digests = {V2_SCF_CHECKPOINT.checkpoint_sha256}
    collisions = [f.key for f in FINALISTS if f.checkpoint_sha256 in scf_digests]
    if collisions:
        raise ScfPathwayViolation(
            f"the V2-SCF checkpoint digest is also finalist {collisions}'s; a post-hoc "
            "candidate that hashes to a frozen finalist is not a candidate"
        )


# ---------------------------------------------------------------------------
# The committed freeze artifact
# ---------------------------------------------------------------------------
def require_frozen_scf_protocol_spec() -> dict[str, Any]:
    """Load `docs/spec/stage2-v2-scf-posthoc-protocol.json` and cross-check it.

    The runner must not outlive the protocol it implements. Every value checked
    here is one that, if the artifact and the code disagreed, would let a cache
    key or a head artifact claim a freeze that was never made.
    """
    require_historical_arms_untouched()
    require_scf_candidate_registration()

    if not V2_SCF_PROTOCOL_SPEC_PATH.is_file():
        raise ScfPathwayViolation(
            f"the frozen V2-SCF Stage-2 protocol is missing: {V2_SCF_PROTOCOL_SPEC_PATH}. "
            "The protocol is frozen BEFORE results exist; without the artifact there is "
            "nothing to be bound by."
        )
    payload = json.loads(V2_SCF_PROTOCOL_SPEC_PATH.read_text(encoding="utf-8"))
    if payload.get("schema_version") != V2_SCF_POSTHOC_PROTOCOL_VERSION:
        raise ScfPathwayViolation(
            f"frozen V2-SCF protocol is {payload.get('schema_version')!r} but this runner "
            f"implements {V2_SCF_POSTHOC_PROTOCOL_VERSION!r}"
        )

    def pinned(section: str, key: str) -> Any:
        block = payload.get(section)
        if not isinstance(block, Mapping):
            raise ScfPathwayViolation(
                f"frozen V2-SCF protocol has no {section!r} section"
            )
        entry = block.get(key)
        if not isinstance(entry, Mapping) or "value" not in entry:
            raise ScfPathwayViolation(
                f"frozen V2-SCF protocol does not pin {section}.{key}; a missing frozen "
                "field is a drifted artifact, not an unset default"
            )
        return entry["value"]

    expectations: tuple[tuple[str, str, Any], ...] = (
        ("scientific_status", "posthoc_exploratory", POSTHOC_EXPLORATORY),
        (
            "scientific_status",
            "official_validation_previously_seen",
            OFFICIAL_VALIDATION_PREVIOUSLY_SEEN,
        ),
        ("scientific_status", "official_test_used", OFFICIAL_TEST_USED),
        ("stage1_source", "stage", V2_SCF_CHECKPOINT.stage),
        (
            "stage1_source",
            "source_repository_head",
            V2_SCF_CHECKPOINT.source_repository_head,
        ),
        (
            "stage1_source",
            "stage_artifact_sha256",
            V2_SCF_CHECKPOINT.stage_artifact_sha256,
        ),
        ("stage1_source", "checkpoint_sha256", V2_SCF_CHECKPOINT.checkpoint_sha256),
        ("stage1_source", "selected_update", V2_SCF_CHECKPOINT.update),
        ("stage1_source", "fusion_id", V2_SCF_CHECKPOINT.fusion_id),
        ("stage1_source", "objective_id", V2_SCF_CHECKPOINT.objective_id),
        (
            "stage1_source",
            "held_out_worst_case_score",
            V2_SCF_CHECKPOINT.held_out_worst_case_score,
        ),
        ("pathway", "pathway_id", V2_SCF_PATHWAY_ID),
        ("pathway", "is_a_third_historical_arm", False),
        ("pathway", "historical_arm_count", len(STAGE2_UNMARK_ARM_NAMES)),
        ("dataset", "name", PRIMARY_DATASET),
        ("dataset", "version", PRIMARY_DATASET_VERSION),
        ("dataset", "task", PRIMARY_TASK),
        ("dataset", "num_labels", PRIMARY_NUM_LABELS),
        ("frozen_representation", "encoder_checkpoint", ENCODER_CHECKPOINT),
        ("frozen_representation", "encoder_revision", ENCODER_REVISION),
        ("frozen_representation", "hidden_size", HIDDEN_SIZE),
        (
            "frozen_representation",
            "adapter_trainable_parameters",
            ADAPTER_TRAINABLE_PARAMETERS,
        ),
        ("pooling", "strategy", STAGE2_FIRST_TOKEN_POOLING),
        ("splits", "official_test", "SEALED"),
    )
    for section, key, want in expectations:
        got = pinned(section, key)
        if got != want:
            raise ScfPathwayViolation(
                f"frozen V2-SCF protocol pins {section}.{key} as {got!r} but this runner "
                f"implements {want!r}. The artifact and the implementation must agree, or "
                "an execution would bind a freeze that was never made."
            )
    return payload


# ---------------------------------------------------------------------------
# Checkpoint verification
# ---------------------------------------------------------------------------
def expected_scf_run_provenance(
    identity: ScfCheckpointIdentity = V2_SCF_CHECKPOINT, *, inventory: Any
) -> RunProvenance:
    """The Stage-1 identity the V2-SCF checkpoint must match.

    Built from the **plan**, exactly as `finalists.expected_run_provenance` is:
    the seeds come from the repository's own V2-SCF run plan, `init_seed` from
    `adapter_init_seed`, the corruption seed and backbone/protocol/precision
    defaults from `protocol`, and the corpus digest and inventory from the
    Audit-048 pins that describe the one prepared corpus every Stage-1 run used.

    The two fields that make this *not* UNMARK-A despite sharing its seed, LR and
    `r` are `fusion` -- the scale-calibrated identity -- and `repository_head`.
    """
    candidate = require_scf_candidate_registration()
    return RunProvenance(
        run_seed=identity.run_seed,
        init_seed=identity.init_seed,
        corruption_seed=CORRUPTION_SEED,
        learning_rate=identity.learning_rate,
        r=identity.r,
        corpus_manifest_digest=CORPUS_MANIFEST_DIGEST,
        repository_head=identity.source_repository_head,
        objective=candidate.objective,
        fusion=candidate.fusion,
        inventory=inventory,
    )


def verify_scf_stage_artifact(
    path: str | Path, identity: ScfCheckpointIdentity = V2_SCF_CHECKPOINT
) -> dict[str, Any]:
    """Fail closed unless the file at `path` IS the frozen `v2_scf.json`.

    Verifies the stage artifact's own digest and then the claims inside it that
    the Stage-2 freeze binds: the stage, the source HEAD, the objective, the
    selected update, the selected held-out worst-case score, and the two
    safety declarations a first-screen artifact always carries.

    Reading the stage artifact is **evidence, not selection**: the checkpoint was
    already chosen inside Stage-1 by the locked held-out rule, and nothing here
    can revisit that choice.
    """
    path = Path(path)
    if not path.is_file():
        raise ScfPathwayViolation(f"V2-SCF stage artifact is missing: {path}")
    digest = sha256_file(path)
    if digest != identity.stage_artifact_sha256:
        raise ScfPathwayViolation(
            f"stage artifact sha256 mismatch: {path} is {digest}, the frozen identity is "
            f"{identity.stage_artifact_sha256}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("stage") != identity.stage:
        raise ScfPathwayViolation(
            f"{path} records stage {payload.get('stage')!r}, expected {identity.stage!r}"
        )
    if payload.get("repository_head") != identity.source_repository_head:
        raise ScfPathwayViolation(
            f"{path} records repository_head {payload.get('repository_head')!r}, expected "
            f"{identity.source_repository_head!r}"
        )
    objective = payload.get("objective")
    if not isinstance(objective, Mapping) or objective.get("objective_id") != identity.objective_id:
        raise ScfPathwayViolation(
            f"{path} records objective {objective!r}, expected objective_id "
            f"{identity.objective_id!r}"
        )
    for flag in ("official_test_used", "downstream_score_used"):
        if payload.get(flag) is not False:
            raise ScfPathwayViolation(
                f"{path} records {flag}={payload.get(flag)!r}; a Stage-1 artifact that "
                "claims otherwise cannot be the source of a post-hoc Stage-2 campaign"
            )
    stage_block = payload.get(identity.stage)
    if not isinstance(stage_block, Mapping):
        raise ScfPathwayViolation(
            f"{path} carries no {identity.stage!r} first-screen block"
        )
    selected = stage_block.get("selected")
    if not isinstance(selected, Mapping):
        raise ScfPathwayViolation(f"{path} first-screen block records no selected point")
    if selected.get("update") != identity.update:
        raise ScfPathwayViolation(
            f"{path} selected update is {selected.get('update')!r}, expected "
            f"{identity.update!r}"
        )
    if selected.get("score") != identity.held_out_worst_case_score:
        raise ScfPathwayViolation(
            f"{path} selected held-out worst-case score is {selected.get('score')!r}, "
            f"expected {identity.held_out_worst_case_score!r}"
        )
    return {
        "kind": "stage2_v2_scf_stage_artifact_evidence",
        "stage": identity.stage,
        "stage_artifact_sha256": digest,
        "source_repository_head": identity.source_repository_head,
        "selected_update": identity.update,
        "held_out_worst_case_score": identity.held_out_worst_case_score,
        "objective_id": identity.objective_id,
        "fusion_id": identity.fusion_id,
        "official_test_used": False,
        "downstream_score_used": False,
    }


def verify_scf_checkpoint(
    path: str | Path,
    identity: ScfCheckpointIdentity = V2_SCF_CHECKPOINT,
    *,
    inventory: Any = None,
) -> dict[str, Any]:
    """Fail closed unless the file at `path` IS the frozen V2-SCF checkpoint.

    Four independent gates, in order, each of which alone would catch the wrong
    file:

    1. **bytes** -- `sha256_file` against the frozen digest;
    2. **architecture** -- `require_loadable_as(payload, scale-calibrated-fusion-v1)`,
       which reads the checkpoint's own recorded `provenance.fusion`. This is the
       gate that refuses a historical adapter whose tensors would otherwise load
       here without complaint;
    3. **experiment** -- `trainer.verify_checkpoint` against a freshly built
       `RunProvenance`, comparing all twelve scientific identity fields plus the
       objective and fusion blocks and the two derived objective weights;
    4. **selection** -- `global_update` against the frozen selected update.

    Never writes. Returns an evidence record of identifiers and hashes only.
    """
    path = Path(path)
    if not path.is_file():
        raise ScfPathwayViolation(f"V2-SCF checkpoint is missing: {path}")
    require_frozen_scf_protocol_spec()
    if inventory is None:
        inventory = resolve_inventory()

    digest = sha256_file(path)
    if digest != identity.checkpoint_sha256:
        raise ScfPathwayViolation(
            f"sha256 mismatch for the V2-SCF checkpoint: {path} is {digest}, the frozen "
            f"identity is {identity.checkpoint_sha256}. A digest mismatch is a different "
            "file, never a tolerable difference."
        )

    import torch  # lazy: the identity contract itself is checkable without torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ScfPathwayViolation(
            f"{path} did not deserialise to a checkpoint mapping; got "
            f"{type(payload).__name__}"
        )

    # GATE 2, before anything is built. The tensors are shape-compatible across
    # fusions, so this is the only point at which a historical checkpoint can be
    # told apart from a scale-calibrated one cheaply and unambiguously.
    try:
        require_loadable_as(payload, identity.fusion_id)
    except Stage1ContractViolation as error:
        raise ScfPathwayViolation(
            f"{path} is not a {identity.fusion_id!r} checkpoint: {error}"
        ) from error

    expected = expected_scf_run_provenance(identity, inventory=inventory)
    try:
        verify_checkpoint(dict(payload), expected)
    except TrainerContractViolation as error:
        raise ScfPathwayViolation(
            f"{path} is not the frozen V2-SCF Stage-1 result: {error}"
        ) from error

    if payload.get("global_update") != identity.update:
        raise ScfPathwayViolation(
            f"{path} global_update {payload.get('global_update')!r} != the frozen "
            f"selected update {identity.update}. The Stage-2 campaign is bound to ONE "
            "Stage-1 checkpoint; another update of the same run is a different model."
        )

    provenance = payload[PROVENANCE_KEY]
    recorded_inventory = provenance.get("inventory")
    if not isinstance(recorded_inventory, Mapping):
        raise ScfPathwayViolation(
            f"{path} provenance records no inventory identity (D-S1A-008)"
        )
    for field_name, want in (
        ("sha256", INVENTORY_SHA256),
        ("source_revision", INVENTORY_SOURCE_REVISION),
        ("size_bytes", INVENTORY_SIZE_BYTES),
    ):
        got = recorded_inventory.get(field_name)
        if got != want:
            raise ScfPathwayViolation(
                f"{path} provenance.inventory.{field_name} is {got!r}, expected {want!r}"
            )

    adapter_state = payload.get("adapter_state")
    if not isinstance(adapter_state, Mapping) or not adapter_state:
        raise ScfPathwayViolation(f"{path} carries no adapter_state")
    keys = tuple(sorted(str(k) for k in adapter_state))
    if keys != ADAPTER_STATE_KEYS:
        raise ScfPathwayViolation(
            f"{path} adapter_state keys {list(keys)} != the locked adapter contract "
            f"{list(ADAPTER_STATE_KEYS)}"
        )
    parameters = 0
    for name, tensor in adapter_state.items():
        if not isinstance(tensor, torch.Tensor):
            raise ScfPathwayViolation(
                f"{path} adapter_state[{name!r}] is {type(tensor).__name__}, not a tensor"
            )
        if tensor.dtype is not torch.float32:
            raise ScfPathwayViolation(
                f"{path} adapter_state[{name!r}] dtype is {tensor.dtype}, not float32"
            )
        if not bool(torch.isfinite(tensor).all()):
            raise ScfPathwayViolation(
                f"{path} adapter_state[{name!r}] contains NaN or Inf"
            )
        parameters += int(tensor.numel())
    if parameters != ADAPTER_TRAINABLE_PARAMETERS:
        raise ScfPathwayViolation(
            f"{path} adapter has {parameters} parameters, not the locked "
            f"{ADAPTER_TRAINABLE_PARAMETERS}"
        )

    execution = payload.get("execution")
    return {
        "kind": "stage2_v2_scf_checkpoint_evidence",
        "pathway_id": identity.pathway_id,
        "stage": identity.stage,
        "source_repository_head": identity.source_repository_head,
        "run_seed": identity.run_seed,
        "init_seed": identity.init_seed,
        "corruption_seed": CORRUPTION_SEED,
        "update": identity.update,
        "cap": payload.get("cap"),
        "learning_rate": identity.learning_rate,
        "r": identity.r,
        "lambda_align": expected.weights.lambda_align,
        "lambda_clean": expected.weights.lambda_clean,
        "objective_id": identity.objective_id,
        "fusion_id": recorded_fusion_id(provenance),
        "checkpoint_sha256": digest,
        "checkpoint_bytes": path.stat().st_size,
        "corpus_manifest_digest": CORPUS_MANIFEST_DIGEST,
        "inventory": dict(recorded_inventory),
        "backbone_checkpoint": ENCODER_CHECKPOINT,
        "backbone_revision": ENCODER_REVISION,
        "protocol_version": STAGE1_PROTOCOL_VERSION,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "adapter_tensor_count": len(adapter_state),
        "adapter_trainable_parameters": parameters,
        "adapter_dtype": PRECISION,
        "all_finite": True,
        # Operational, NOT campaign identity -- reported, never enforced.
        "execution_fingerprint": dict(execution) if isinstance(execution, Mapping) else execution,
    }


@dataclass(frozen=True)
class ScfCheckpointBinding:
    """Verified checkpoint evidence bound to the post-hoc V2-SCF pathway."""

    pathway_id: str
    stage: str
    source_repository_head: str
    run_seed: int
    update: int
    fusion_id: str
    objective_id: str
    checkpoint_sha256: str
    checkpoint_path: str
    evidence: Mapping[str, Any] = field(repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": V2_SCF_STAGE2_SCHEMA_VERSION,
            "pathway_id": self.pathway_id,
            "stage": self.stage,
            "source_repository_head": self.source_repository_head,
            "run_seed": self.run_seed,
            "update": self.update,
            "fusion_id": self.fusion_id,
            "objective_id": self.objective_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_path": self.checkpoint_path,
            "stage1_evidence_kind": self.evidence.get("kind"),
        }


def bind_verified_scf_checkpoint(
    checkpoint_path: str | Path,
    evidence: Mapping[str, Any],
    identity: ScfCheckpointIdentity = V2_SCF_CHECKPOINT,
) -> ScfCheckpointBinding:
    """Bind verifier evidence to the frozen V2-SCF identity. Fails closed.

    `verify_scf_checkpoint` stays the authoritative verifier; this only refuses
    evidence that does not name the one checkpoint this campaign may consume.
    """
    require_scf_pathway_id(str(evidence.get("pathway_id", "")))
    required = {
        "kind",
        "pathway_id",
        "stage",
        "source_repository_head",
        "run_seed",
        "update",
        "fusion_id",
        "objective_id",
        "checkpoint_sha256",
        "checkpoint_schema_version",
        "adapter_tensor_count",
        "adapter_trainable_parameters",
        "adapter_dtype",
        "all_finite",
    }
    missing = sorted(required - set(evidence))
    if missing:
        raise ScfPathwayViolation(
            f"V2-SCF checkpoint evidence is missing {missing}"
        )
    expected: dict[str, Any] = {
        "kind": "stage2_v2_scf_checkpoint_evidence",
        "pathway_id": identity.pathway_id,
        "stage": identity.stage,
        "source_repository_head": identity.source_repository_head,
        "run_seed": identity.run_seed,
        "update": identity.update,
        "fusion_id": identity.fusion_id,
        "objective_id": identity.objective_id,
        "checkpoint_sha256": identity.checkpoint_sha256,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "adapter_tensor_count": ADAPTER_TENSOR_COUNT,
        "adapter_trainable_parameters": ADAPTER_TRAINABLE_PARAMETERS,
        "adapter_dtype": PRECISION,
        "all_finite": True,
    }
    differences = [name for name, want in expected.items() if evidence.get(name) != want]
    if differences:
        detail = ", ".join(
            f"{name}: got {evidence.get(name)!r}, expected {expected[name]!r}"
            for name in differences
        )
        raise ScfPathwayViolation(
            f"checkpoint evidence does not bind to {identity.pathway_id}: {detail}"
        )
    return ScfCheckpointBinding(
        pathway_id=identity.pathway_id,
        stage=identity.stage,
        source_repository_head=identity.source_repository_head,
        run_seed=identity.run_seed,
        update=identity.update,
        fusion_id=identity.fusion_id,
        objective_id=identity.objective_id,
        checkpoint_sha256=identity.checkpoint_sha256,
        checkpoint_path=str(Path(checkpoint_path)),
        evidence=evidence,
    )


# ---------------------------------------------------------------------------
# The frozen pathway
# ---------------------------------------------------------------------------
@dataclass
class FrozenScfPathway:
    """A frozen encoder plus a frozen **scale-calibrated** Stage-1 adapter.

    Deliberately NOT a subclass of `FrozenUnmarkPathway`: that class carries a
    `Stage2UnmarkArm`, and inheriting it would make a V2-SCF pathway pass every
    `isinstance` check the historical code performs. It is instead structurally
    compatible where it must be -- `.encoder`, `.adapter`, `.require_frozen()` --
    so `extract_stage2_unmark_representations` runs the accepted Audit-050
    forward over it without a second implementation existing anywhere.
    """

    binding: ScfCheckpointBinding
    encoder: Any
    adapter: Any
    hidden_size: int = HIDDEN_SIZE

    @property
    def pathway_id(self) -> str:
        return self.binding.pathway_id

    @property
    def checkpoint_sha256(self) -> str:
        return self.binding.checkpoint_sha256

    @property
    def fusion_id(self) -> str:
        return self.binding.fusion_id

    def require_frozen(self, *, check_values: bool = False) -> None:
        require_frozen_scf_pathway(self, check_values=check_values)


def _freeze_module(module: Any) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    module.eval()


def build_scf_adapter(payload: Mapping[str, Any]) -> Any:
    """Build the authoritative scale-calibrated adapter from a checkpoint payload.

    **The equation is not written here.** `reconstruct_adapter` is the Stage-1
    construction API: it reads `provenance.fusion`, dispatches through
    `fresh_adapter` -> `AdapterConfig(fusion_id=...)` -> `OrthographyInputAdapter`,
    and loads the tensors with `strict=True`. The guard above it refuses any
    payload whose recorded architecture is not the scale-calibrated one, so this
    function cannot silently return a historical adapter holding C1's weights.
    """
    try:
        require_loadable_as(payload, SCALE_CALIBRATED_FUSION_ID)
    except Stage1ContractViolation as error:
        raise ScfPathwayViolation(
            f"refusing to build a V2-SCF adapter from this checkpoint: {error}"
        ) from error
    adapter = reconstruct_adapter(payload, HIDDEN_SIZE)
    config = getattr(adapter, "config", None)
    if getattr(config, "fusion_id", None) != SCALE_CALIBRATED_FUSION_ID:
        raise ScfPathwayViolation(
            f"the reconstructed adapter reports fusion {getattr(config, 'fusion_id', None)!r}, "
            f"not {SCALE_CALIBRATED_FUSION_ID!r}. The Stage-1 dispatch is the only "
            "construction path and it must produce the recorded architecture."
        )
    if not getattr(config, "is_scale_calibrated", False):
        raise ScfPathwayViolation(
            "the reconstructed adapter does not report scale-calibrated fusion"
        )
    return adapter


def load_frozen_scf_pathway(
    checkpoint_path: str | Path,
    *,
    encoder: Any | None = None,
    inventory: Any = None,
    cache_dir: str | Path | None = None,
    identity: ScfCheckpointIdentity = V2_SCF_CHECKPOINT,
) -> FrozenScfPathway:
    """Load the post-hoc V2-SCF pathway from an explicit checkpoint path.

    Mirrors `load_frozen_unmark_pathway` step for step, with exactly two
    differences, both of them the point of this module: the checkpoint is
    verified against the V2-SCF identity rather than a finalist, and the adapter
    is built through the Stage-1 fusion dispatch rather than by constructing a
    default `AdapterConfig` -- which would be the historical mixture rule.
    """
    require_frozen_scf_protocol_spec()
    if encoder is None:
        _, encoder = load_stage2_phobert_components(cache_dir=cache_dir)
    require_stage2_encoder_identity(encoder)

    evidence = verify_scf_checkpoint(checkpoint_path, identity, inventory=inventory)
    binding = bind_verified_scf_checkpoint(checkpoint_path, evidence, identity)

    import torch

    payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    adapter = build_scf_adapter(payload)

    encoder_device = next(encoder.parameters()).device
    adapter.to(device=encoder_device)
    _freeze_module(encoder)
    _freeze_module(adapter)

    pathway = FrozenScfPathway(binding=binding, encoder=encoder, adapter=adapter)
    pathway.require_frozen(check_values=True)
    return pathway


def require_frozen_scf_pathway(
    pathway: FrozenScfPathway, *, check_values: bool = False
) -> None:
    """Every encoder/adapter parameter frozen, FP32, eval -- and scale-calibrated.

    The last clause is the one the historical checker cannot make: it verifies
    that the **live module** implements the scale-calibrated rule, not merely
    that a string in a binding says so.
    """
    import torch

    require_stage2_encoder_identity(pathway.encoder)
    if not isinstance(pathway.binding, ScfCheckpointBinding):
        raise ScfPathwayViolation(
            f"pathway carries {type(pathway.binding).__name__}, not a ScfCheckpointBinding"
        )
    require_scf_pathway_id(pathway.binding.pathway_id)
    if pathway.binding.fusion_id != SCALE_CALIBRATED_FUSION_ID:
        raise ScfPathwayViolation(
            f"pathway binding records fusion {pathway.binding.fusion_id!r}, expected "
            f"{SCALE_CALIBRATED_FUSION_ID!r}"
        )

    config = getattr(pathway.adapter, "config", None)
    if getattr(config, "fusion_id", None) != SCALE_CALIBRATED_FUSION_ID:
        raise ScfPathwayViolation(
            f"the loaded adapter implements fusion {getattr(config, 'fusion_id', None)!r}, "
            f"not {SCALE_CALIBRATED_FUSION_ID!r}. The V2-SCF weights are shape-compatible "
            "with the historical mixture rule, so this is the check that stops them "
            "being evaluated under the wrong equation."
        )
    if not getattr(config, "is_scale_calibrated", False):
        raise ScfPathwayViolation("the loaded adapter is not scale-calibrated")

    hidden = getattr(getattr(pathway.encoder, "config", None), "hidden_size", None)
    if hidden != HIDDEN_SIZE or pathway.hidden_size != HIDDEN_SIZE:
        raise ScfPathwayViolation(
            f"V2-SCF hidden size must be {HIDDEN_SIZE}, got encoder={hidden!r} "
            f"pathway={pathway.hidden_size!r}"
        )

    devices = set()
    for module_name, module in (("encoder", pathway.encoder), ("adapter", pathway.adapter)):
        if getattr(module, "training", True):
            raise ScfPathwayViolation(f"{module_name} must be in eval mode")
        parameter_count = 0
        for name, parameter in module.named_parameters():
            parameter_count += 1
            devices.add(str(parameter.device))
            if parameter.requires_grad:
                raise ScfPathwayViolation(
                    f"{module_name}.{name} requires grad; encoder and adapter are frozen "
                    "for the whole of Stage-2 and only the head may train"
                )
            if parameter.dtype.is_floating_point and parameter.dtype is not torch.float32:
                raise ScfPathwayViolation(
                    f"{module_name}.{name} dtype is {parameter.dtype}, expected torch.float32"
                )
            if check_values and parameter.dtype.is_floating_point and not bool(
                torch.isfinite(parameter).all()
            ):
                raise ScfPathwayViolation(f"{module_name}.{name} contains NaN or Inf")
        if parameter_count == 0:
            raise ScfPathwayViolation(f"{module_name} exposes no parameters")
    if len(devices) != 1:
        raise ScfPathwayViolation(
            f"encoder and adapter parameters span devices {sorted(devices)}"
        )
    if "meta" in next(iter(devices)):
        raise ScfPathwayViolation(
            "parameters are on the meta device; a real CPU/GPU device is required"
        )

    adapter_keys = tuple(sorted(pathway.adapter.state_dict()))
    if adapter_keys != ADAPTER_STATE_KEYS:
        raise ScfPathwayViolation(
            f"adapter state keys drifted: {list(adapter_keys)} != {list(ADAPTER_STATE_KEYS)}"
        )
    adapter_parameters = sum(int(p.numel()) for p in pathway.adapter.parameters())
    if adapter_parameters != ADAPTER_TRAINABLE_PARAMETERS:
        raise ScfPathwayViolation(
            f"adapter has {adapter_parameters} parameters, expected "
            f"{ADAPTER_TRAINABLE_PARAMETERS}"
        )


def extract_scf_representations(
    pathway: FrozenScfPathway, batch: Mapping[str, Any], **encoder_kwargs: Any
) -> Any:
    """Frozen V2-SCF first-token representations, detached FP32 `[B, 768]`.

    **Delegates.** The forward pass is the accepted Audit-050 one and is not
    re-implemented: same `base_word_embeddings`, same authoritative position ids,
    same `<s>` pooling, same detach-to-FP32. The only thing that differs is which
    adapter the pathway carries, which is exactly the variable under study.
    """
    if not isinstance(pathway, FrozenScfPathway):
        raise ScfPathwayViolation(
            f"expected a FrozenScfPathway, got {type(pathway).__name__}"
        )
    pathway.require_frozen()
    return extract_stage2_unmark_representations(pathway, batch, **encoder_kwargs)


def scf_head_trainable_parameters(pathway: FrozenScfPathway, head: Any) -> tuple[Any, ...]:
    """Head parameters only; encoder and adapter can never leak into training."""
    pathway.require_frozen()
    frozen_ids = {
        id(parameter)
        for module in (pathway.encoder, pathway.adapter)
        for parameter in module.parameters()
    }
    selected = tuple(parameter for parameter in head.parameters() if parameter.requires_grad)
    leaked = [parameter for parameter in selected if id(parameter) in frozen_ids]
    if leaked:
        raise ScfPathwayViolation(
            "head parameter enumeration includes frozen encoder/adapter parameter objects"
        )
    if not selected:
        raise ScfPathwayViolation("the V2-SCF Stage-2 head has no trainable parameters")
    return selected


__all__ = [
    "OFFICIAL_TEST_ROLE_EXISTS",
    "OFFICIAL_TEST_USED",
    "OFFICIAL_VALIDATION_PREVIOUSLY_SEEN",
    "POSTHOC_EXPLORATORY",
    "STAGE2_FIRST_TOKEN_POOLING",
    "STAGE2_REPRESENTATION_DTYPE",
    "STAGE2_UNMARK_CONDITIONS",
    "V2_SCF_BEST_SEED_SELECTION_IMPLEMENTED",
    "V2_SCF_CHECKPOINT",
    "V2_SCF_PATHWAY_ID",
    "V2_SCF_POSTHOC_PROTOCOL_VERSION",
    "V2_SCF_PROTOCOL_SPEC_PATH",
    "V2_SCF_SELECTION_AGAINST_AB_IMPLEMENTED",
    "V2_SCF_STAGE2_HYPERPARAMETER_RETUNED",
    "V2_SCF_STAGE2_SCHEMA_VERSION",
    "FrozenScfPathway",
    "ScfCheckpointBinding",
    "ScfCheckpointIdentity",
    "ScfPathwayViolation",
    "bind_verified_scf_checkpoint",
    "build_scf_adapter",
    "expected_scf_run_provenance",
    "extract_scf_representations",
    "load_frozen_scf_pathway",
    "require_frozen_scf_pathway",
    "require_frozen_scf_protocol_spec",
    "require_historical_arms_untouched",
    "require_scf_candidate_registration",
    "require_scf_pathway_id",
    "scf_head_trainable_parameters",
    "verify_scf_checkpoint",
    "verify_scf_stage_artifact",
]
