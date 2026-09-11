"""Stage orchestration: build the model, run a schedule, persist selection.

**Imports torch, transformers and the prepared corpus lazily.** Nothing here
runs at import time, and Audit 029 executes none of it.

`execute_stage` is the only place a scientific Stage-1 run is launched, and
`smoke_check` is deliberately a *separate* function that constructs no optimizer
and calls no `.backward()` -- so "the smoke path cannot update a parameter" is
structural rather than a promise.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Sequence

from unmark.modeling.contracts import HISTORICAL_FUSION_ID
from unmark.stage1.candidates import (
    HISTORICAL_STAGES,
    StageCandidate,
    candidate_for_stage,
)
from unmark.stage1.contracts import (
    CorruptionRatePolicy,
    GEOMETRY_RELATIONAL_OBJECTIVE,
    GRID_CONSISTENCY_OBJECTIVE,
    OverflowBehaviour,
    Stage1ContractViolation,
    TruncationPolicy,
)
from unmark.stage1.checkpoint import VerifiedCorpus
from unmark.stage1.trainer import load_training_checkpoint, resolve_run_cap, resume_cap
from unmark.stage1.manifest import CHUNKS_NAME
from unmark.stage1.protocol import (
    BATCH_SIZE,
    CORRUPTION_SEED,
    ENCODER_CHECKPOINT,
    ENCODER_REVISION,
    EXTENDED_MAX_UPDATES,
    HIDDEN_SIZE,
    INITIAL_MAX_UPDATES,
    LR_PILOT_R,
    MAX_LENGTH,
    STAGE1_PROTOCOL_VERSION,
    V2_GC_STAGE,
    V2_GRD_STAGE,
    V2_SCF_STAGE,
    lambdas_for_r,
)
from unmark.stage1.artifact import CampaignIdentity
from unmark.stage1.telemetry import NullSink, TelemetrySink, phase
from unmark.stage1.selection import (
    Candidate,
    PlannedRun,
    select_learning_rate,
    select_r,
)
from unmark.stage1.trainer import RunProvenance, train_run, verify_model_contract

FIRST_SCREEN_STAGES: tuple[str, ...] = (V2_GC_STAGE, V2_SCF_STAGE, V2_GRD_STAGE)
"""Post-hoc single-run candidate stages. Each writes its own artifact leg and
performs NO selection -- there is one run, and the checkpoint within it is still
chosen by the locked held-out rule. A further candidate joins by being
registered."""

HISTORICAL_SMOKE_STAGE = HISTORICAL_STAGES[0]
"""Default candidate for `smoke_check`: the historical objective.

A smoke that names no candidate smokes the objective every existing checkpoint
was trained under, so the pre-repair default behaviour is preserved exactly."""

_FULL_SHA = re.compile(r"^[0-9a-fA-F]{40}$")
"""A commit identity is the full sha. A branch name or abbreviation is not one."""

TRUNCATION = TruncationPolicy(max_length=MAX_LENGTH, on_overflow=OverflowBehaviour.FAIL)
"""`FAIL` is a guard: after correct pre-chunking nothing can overflow."""


def load_prepared_chunks(directory: Path) -> tuple[dict[str, str], dict[str, str]]:
    """`(train_text_by_chunk_id, dev_text_by_chunk_id)` from the prepared corpus."""
    path = Path(directory) / CHUNKS_NAME
    if not path.is_file():
        raise Stage1ContractViolation(f"prepared chunks not found: {path}")
    train: dict[str, str] = {}
    dev: dict[str, str] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            target = train if row["partition"] == "train" else dev
            target[row["chunk_id"]] = row["text"]
    if not train or not dev:
        raise Stage1ContractViolation(
            f"prepared corpus has {len(train)} train and {len(dev)} dev chunks; both "
            "partitions must be non-empty"
        )
    return train, dev


def build_backbone(revision: str):
    """Tokenizer + the pinned FROZEN encoder. `(tokenizer, encoder, hidden_size)`.

    Stage-scope immutable state (D-S1B-017): loaded once per stage command and
    shared by every nominal run. It deliberately does **not** build an adapter —
    that is per-run, deterministic and CPU-first under D-S1B-016.
    """
    from transformers import AutoModel, AutoTokenizer

    if revision != ENCODER_REVISION:
        raise Stage1ContractViolation(
            f"backbone revision {revision!r} is not the locked {ENCODER_REVISION!r}"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        ENCODER_CHECKPOINT, revision=revision, use_fast=False
    )
    encoder = AutoModel.from_pretrained(ENCODER_CHECKPOINT, revision=revision)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    encoder.eval()
    hidden = int(encoder.config.hidden_size)
    if hidden != HIDDEN_SIZE:
        raise Stage1ContractViolation(f"hidden size {hidden} != locked {HIDDEN_SIZE}")
    return tokenizer, encoder, hidden


def require_frozen_backbone_unchanged(encoder, expected_hash: str, label: str) -> None:
    """The shared encoder is unchanged after a nominal run (D-S1B-017).

    Full `state_dict` — parameters *and* persistent buffers — because parameters
    alone would miss a mutated buffer. Plus: still zero trainable parameters, and
    no gradient was ever accumulated onto it.
    """
    from unmark.stage1.initialisation import module_state_hash

    observed = module_state_hash(encoder)
    if observed != expected_hash:
        raise Stage1ContractViolation(
            f"after {label} the shared frozen backbone CHANGED "
            f"({expected_hash[:12]}... -> {observed[:12]}...). It is shared across "
            "nominal runs, so every later candidate would be contaminated."
        )
    trainable = [n for n, p in encoder.named_parameters() if p.requires_grad]
    if trainable:
        raise Stage1ContractViolation(
            f"after {label} the frozen backbone has {len(trainable)} trainable "
            f"parameter(s), e.g. {trainable[:3]}"
        )
    with_grad = [n for n, p in encoder.named_parameters() if p.grad is not None]
    if with_grad:
        raise Stage1ContractViolation(
            f"after {label} the frozen backbone carries gradients, e.g. {with_grad[:3]}"
        )


def build_objective(revision: str, fusion_id: str = HISTORICAL_FUSION_ID):
    """Frozen pinned encoder + the locked adapter. **Lazy torch/transformers.**

    `fusion_id` selects the adapter architecture and defaults to the historical
    one, so every existing caller keeps its exact behaviour. It exists so the
    real-model smoke can exercise a candidate's actual adapter rather than
    checking a candidate objective over the historical mixture rule.
    """
    from transformers import AutoModel, AutoTokenizer

    from unmark.modeling.adapter import OrthographyInputAdapter, UnmarkEncoder
    from unmark.modeling.config import AdapterConfig
    from unmark.stage1.objective import Stage1Objective

    if revision != ENCODER_REVISION:
        raise Stage1ContractViolation(
            f"backbone revision {revision!r} is not the locked {ENCODER_REVISION!r}"
        )
    tokenizer = AutoTokenizer.from_pretrained(
        ENCODER_CHECKPOINT, revision=revision, use_fast=False
    )
    encoder = AutoModel.from_pretrained(ENCODER_CHECKPOINT, revision=revision)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    encoder.eval()
    hidden = int(encoder.config.hidden_size)
    if hidden != HIDDEN_SIZE:
        raise Stage1ContractViolation(f"hidden size {hidden} != locked {HIDDEN_SIZE}")
    adapter = OrthographyInputAdapter(
        AdapterConfig(hidden_size=hidden, fusion_id=fusion_id)
    )
    return tokenizer, UnmarkEncoder(encoder=encoder, adapter=adapter), Stage1Objective


def build_candidate_objective(unmark_encoder, candidate: StageCandidate, weights):
    """THE Stage-1 objective constructor. **Lazy torch. One dispatch, one place.**

    Used by `execute_stage` for a real run and by `smoke_check` for the
    no-update real-model check, so a smoke can never validate a different loss
    from the one the run would optimise (Audit 065 BLOCKER 4). Before this
    existed, `smoke_check` built `Stage1Objective` unconditionally and could not
    exercise a candidate objective at all.

    Dispatch is on the candidate's **objective identity**, resolved by
    `candidates.candidate_for_stage` from the stage name -- never on an argument,
    a flag or a string compared here. Adding C1/C2 adds a branch to the register,
    not to every call site.

    `weights` supplies `lambda_align` and `lambda_clean`, which keep their
    historical meaning and their historical source (`lambdas_for_r`). A candidate
    whose objective needs more than those two upgrades them to its own weights
    type; `lambda_grid` is never passed in, because it is locked in `protocol`
    and `GridConsistencyWeights` refuses any other value.
    """
    from unmark.stage1.contracts import GridConsistencyWeights
    from unmark.stage1.objective import Stage1Objective
    from unmark.stage1.objective_grid import GridConsistencyObjective
    from unmark.stage1.objective_relational import RelationalDistillationObjective

    # The adapter that was actually built must implement the fusion this
    # candidate declares. Both are derived from the same `StageCandidate`, so a
    # mismatch means two derivations drifted -- which would train C1's objective
    # over the historical architecture, or the reverse, and record a provenance
    # that describes neither.
    built = getattr(unmark_encoder.adapter.config, "fusion_id", None)
    if built != candidate.fusion.fusion_id:
        raise Stage1ContractViolation(
            f"{candidate.stage}: the adapter implements fusion {built!r} but the "
            f"candidate declares {candidate.fusion.fusion_id!r}. The architecture "
            "and the provenance must be the same statement."
        )
    if candidate.objective is GRID_CONSISTENCY_OBJECTIVE:
        # The SAME adapter, the SAME frozen backbone, and the two historical
        # weights read straight off the plan -- so a V2-GC run cannot optimise
        # one pair of lambdas while its checkpoint claims another.
        return GridConsistencyObjective(
            unmark_encoder,
            GridConsistencyWeights(
                lambda_align=weights.lambda_align,
                lambda_clean=weights.lambda_clean,
            ),
        )
    if candidate.objective is GEOMETRY_RELATIONAL_OBJECTIVE:
        # C2 takes the plain historical weights: `lambda_grd` and the whole
        # relational specification are locked in `protocol` and carried by the
        # objective identity, so there is nothing about the third term that a
        # caller could pass in or get wrong.
        return RelationalDistillationObjective(unmark_encoder, weights)
    objective = Stage1Objective(unmark_encoder, weights)
    return objective


def continuation_permitted(
    candidate: StageCandidate, leg_cap: int, result_cap: int
) -> bool:
    """May this run enter the historical 20k -> 40k continuation leg?

    Two independent conditions, and the candidate's is checked first:

    * its screening budget must ALLOW the precommitted continuation at all --
      a first-screen candidate is hard-capped and never continues, whatever its
      held-out curve did (Audit 065 BLOCKER 1);
    * the leg that just ran must be the initial one and the locked rule must
      have asked for the extension, exactly as before.

    Pure and module-level so the decision is testable without torch, a corpus or
    a GPU. `resolve_budget` already refuses to raise `result.cap` for a
    hard-capped candidate, so for those this is a second, independent refusal.
    """
    if not candidate.budget.allows_precommitted_continuation:
        return False
    return leg_cap == INITIAL_MAX_UPDATES and result_cap == EXTENDED_MAX_UPDATES


def execute_stage(
    *,
    stage: str,
    schedule: Sequence[PlannedRun],
    prepared_corpus: Path,
    verified: "VerifiedCorpus",
    output_dir: Path,
    cache_root: Path,
    revision: str,
    repository_head: str,
    resume: bool = False,
    telemetry: TelemetrySink | None = None,
) -> int:
    """Run every planned run of one stage and persist the selection artifact.

    With `resume`, each planned run continues from its own verified training
    checkpoint if one exists, and starts fresh if it does not -- so a stage
    interrupted after two of five runs redoes neither of the two.
    """
    # A scientific run must be able to name the commit that produced it. The
    # runner derives this from Git and refuses a disagreeing assertion; this is
    # the structural backstop, so a future caller cannot reintroduce a `None` or
    # a free-text head by passing one in (Audit 031 B5 / Audit 032 MAJ2).
    if not isinstance(repository_head, str) or not _FULL_SHA.match(repository_head):
        raise Stage1ContractViolation(
            f"repository_head {repository_head!r} is not a full 40-character commit "
            "sha; a Stage-1 artifact that cannot name the code that produced it is "
            "not reproducible. It is derived from Git, never supplied."
        )

    # WHICH objective this stage trains AND how far its run may go, both decided
    # ONCE from the stage name by the candidate register and then carried by
    # provenance into every checkpoint and artifact. The three historical stages
    # keep the historical objective and the locked precommitted budget exactly;
    # the V2-GC stage, which has its own name and its own output namespace, gets
    # the grid term and a 20 000-update hard cap. A stage cannot pick either
    # through an argument, so no flag, config key or typo can give a historical
    # run a different loss or a candidate a longer budget. An unregistered stage
    # fails closed rather than defaulting into the historical one.
    candidate = candidate_for_stage(stage)
    objective_identity = candidate.objective

    # OPERATIONAL ONLY (Audit 040). Defaults to a no-op sink, so a caller that
    # does not opt in gets exactly the pre-telemetry code path.
    sink = telemetry if telemetry is not None else NullSink()
    sink.emit(
        "stage_start", stage=stage, candidate_count=len(schedule),
        repository_head=repository_head, protocol_version=STAGE1_PROTOCOL_VERSION,
        resume=bool(resume), **objective_identity.to_dict(),
        **candidate.fusion.to_dict(),
        candidate_id=candidate.stage,
        budget_policy=candidate.budget.to_dict(),
        hard_max_updates=candidate.budget.hard_max_updates,
        wandb_project=candidate.wandb_project,
    )

    from unmark.linguistics import make_classifier, try_load_inventory
    from unmark.modeling.adapter import UnmarkEncoder
    from unmark.stage1.device import (
        current_fingerprint,
        enforce_numerical_policy,
        require_deterministic_cublas_workspace,
        resolve_scientific_device,
        verify_numerical_policy,
    )
    from unmark.stage1.initialisation import (
        expected_fresh_init_hash,
        fresh_adapter,
        module_state_hash,
        trainable_state,
        trainable_state_hash,
    )
    from unmark.stage1.fused import (
        R_PHASE1_EXECUTION_FUSED,
        R_PHASE1_EXECUTION_SEQUENTIAL,
        resolve_r_phase1_execution,
        train_fused_r_phase1,
    )
    from unmark.stage1.preflight import verify_scientific_inputs
    from unmark.stage1.preparation import (
        PreparationPool,
        preparation_provenance,
        resolve_preparation_workers,
        worker_config,
    )
    from unmark.stage1.protocol import adapter_init_seed
    from unmark.stage1.validation import HeldOutExample, at_update, evaluate, prepare_condition_batch
    from unmark.stage1.protocol import VALIDATION_CONDITIONS

    # Every mandatory external scientific input, BEFORE the encoder is fetched
    # or loaded. The second real smoke (Audit 030 §W) discovered the missing
    # pinned syllable inventory only after the model was already resident.
    with phase(sink, "inventory_preflight"):
        inputs = verify_scientific_inputs()
    print(f"scientific inputs VERIFIED: eligibility {inputs.report['eligibility_policy']}")
    print(f"  inventory {inputs.inventory.source_name} @ {inputs.inventory.source_revision[:12]} "
          f"sha256 {inputs.inventory.sha256[:12]} ({inputs.report['inventory_shape']['unique_stripped_form_count']} stripped forms)")

    # --- SCIENTIFIC EXECUTION CONTRACT (D-S1B-015) ---------------------------
    # Before any model work: CUDA or nothing, and a numerical policy that is
    # enforced and then re-asserted, so a global setting changed elsewhere in the
    # process cannot reach a run whose artifact claims fp32 and determinism.
    with phase(sink, "cuda_policy"):
        require_deterministic_cublas_workspace()
        device = resolve_scientific_device()
        enforce_numerical_policy()
        verify_numerical_policy()
        execution = current_fingerprint(device)
    print(f"scientific execution VERIFIED: {execution.backend} {execution.gpu_name} "
          f"(cc {execution.compute_capability}), torch {execution.torch_version}, "
          f"CUDA {execution.cuda_version}")
    print(f"  deterministic={execution.deterministic_algorithms} "
          f"cudnn.deterministic={execution.cudnn_deterministic} "
          f"cudnn.benchmark={execution.cudnn_benchmark} "
          f"cublas={execution.cublas_workspace_config} "
          f"matmul={execution.float32_matmul_precision}")

    # The 2.2 GB read that went silent for minutes in the first lr-pilot attempt.
    with phase(sink, "corpus_load", prepared_corpus=str(prepared_corpus)):
        train_text, dev_text = load_prepared_chunks(prepared_corpus)
    sink.emit("corpus_loaded", train_chunks=len(train_text), dev_chunks=len(dev_text))
    # STAGE-SCOPE IMMUTABLE STATE. The frozen encoder is loaded once, placed once
    # and shared by every nominal run -- it is pinned, immutable backbone state,
    # and shuttling ~135M parameters per candidate would buy nothing. It is the
    # ONLY model state permitted to cross nominal runs (D-S1B-017).
    with phase(sink, "backbone_load", checkpoint=ENCODER_CHECKPOINT, revision=revision):
        tokenizer, frozen_encoder, hidden_size = build_backbone(revision)
        frozen_encoder.to(device)
    with phase(sink, "backbone_verify"):
        encoder_state_hash = module_state_hash(frozen_encoder)
    print(f"frozen backbone VERIFIED on {device}: state_dict sha256 "
          f"{encoder_state_hash[:12]}... ({hidden_size}d)")
    with phase(sink, "classifier_build"):
        classifier = make_classifier(try_load_inventory())
    pad_token_id = tokenizer.pad_token_id
    held_out = [HeldOutExample(cid, text) for cid, text in sorted(dev_text.items())]

    # The held-out corruption realization is built ONCE and reused by every
    # candidate: the same examples under the same fixed conditions, so candidates
    # differ only by the model. Independent of any training seed.
    # Tokenising every held-out chunk under all four locked conditions: the
    # other long pre-update phase.
    with phase(sink, "validation_batch_build",
               held_out=len(held_out), conditions=list(VALIDATION_CONDITIONS)):
        prepared_by_condition = {
            condition: prepare_condition_batch(
                held_out, tokenizer, condition, truncation=TRUNCATION, classifier=classifier
            )
            for condition in VALIDATION_CONDITIONS
        }

    # The VERIFIED digest (Audit 030 F1). Until the hardening this read
    # `manifest["counts"][...]` -- a declaration accepted on trust, so a run
    # could record a digest describing data it had not trained on. `verified`
    # can only exist if every artifact COMPLETE.json binds was re-hashed from
    # disk and matched.
    manifest_digest = verified.chunk_membership_digest

    # THE single campaign identity, built once here and reused verbatim for the
    # stage artifact below (Audit 040 review). Building it early lets telemetry
    # report authoritative provenance at the START of a stage instead of only
    # after the last candidate finishes -- without creating a second identity
    # definition, which is exactly what Audit 031 B4 / 032 MAJ1 forbade.
    campaign = CampaignIdentity.from_inputs(
        repository_head=repository_head,
        corpus_manifest_digest=manifest_digest,
        encoder_revision=revision,
        inventory=inputs.inventory,
    )
    # The corpus pin comes from the VERIFIED corpus object, not from a constant
    # restated here: `verified` exists only if those bytes were re-hashed.
    sink.emit(
        "campaign_identity",
        corpus_dataset=verified.identity.corpus_dataset,
        corpus_revision=verified.identity.corpus_revision,
        **campaign.to_dict(),
    )

    candidates: list[Candidate] = []
    output_dir.mkdir(parents=True, exist_ok=resume)

    # ONE persistent preparation pool for the whole stage command. Under `spawn`
    # each worker reloads the pinned tokenizer and re-verifies the inventory, so
    # rebuilding it per batch would cost far more than it saves. It is closed on
    # normal completion, on exception and on fail-closed abort alike.
    preparation = worker_config(
        encoder_checkpoint=ENCODER_CHECKPOINT,
        encoder_revision=revision,
        corruption_policy=CorruptionRatePolicy(seed=CORRUPTION_SEED),
        truncation=TRUNCATION,
        unk_token_id=getattr(tokenizer, "unk_token_id", None),
    )
    preparation_workers = resolve_preparation_workers()
    provenance_of_preparation = preparation_provenance(preparation_workers)
    r_phase1_execution = (
        resolve_r_phase1_execution() if stage == "r_phase1"
        else R_PHASE1_EXECUTION_SEQUENTIAL
    )
    if stage == "r_phase1":
        provenance_of_preparation["candidate_execution"] = r_phase1_execution
    print(f"preparation: {provenance_of_preparation['preparation_backend']} x"
          f"{preparation_workers}, order_preserving="
          f"{provenance_of_preparation['order_preserving']}, "
          f"prefetch={provenance_of_preparation['prefetch']}")
    if stage == "r_phase1":
        print(f"r-phase1 execution: {r_phase1_execution}")

    with PreparationPool(preparation, preparation_workers) as preparation_pool:
      if stage == "r_phase1" and r_phase1_execution == R_PHASE1_EXECUTION_FUSED:
          candidates = train_fused_r_phase1(
              schedule=schedule,
              train_chunks=train_text,
              tokenizer=tokenizer,
              frozen_encoder=frozen_encoder,
              hidden_size=hidden_size,
              encoder_state_hash=encoder_state_hash,
              prepared_by_condition=prepared_by_condition,
              pad_token_id=pad_token_id,
              device=device,
              execution=execution,
              manifest_digest=manifest_digest,
              repository_head=repository_head,
              inventory=inputs.inventory,
              output_dir=output_dir,
              preparation_pool=preparation_pool,
              resume=resume,
              telemetry=sink,
          )
          schedule = ()
      for planned in schedule:
          # One checkpoint namespace per run, named by the run's own label, so two
          # runs in a stage can never overwrite each other's state.
          run_checkpoints = output_dir / f"run-{planned.label.replace('=', '')}" / "_checkpoint"
          lambda_align, lambda_clean = lambdas_for_r(planned.r)
          provenance = RunProvenance(
              run_seed=planned.seed,
              init_seed=adapter_init_seed(planned.seed),
              corruption_seed=CORRUPTION_SEED,
              learning_rate=planned.learning_rate,
              r=planned.r,
              corpus_manifest_digest=manifest_digest,
              repository_head=repository_head,
              inventory=inputs.inventory,
              objective=objective_identity,
              fusion=candidate.fusion,
          )
          # --- FRESH NOMINAL RUN (D-S1B-016 / D-S1B-017) ----------------------
          # A NEW adapter, initialised on CPU from this run's domain-separated
          # init seed, then moved to the already-resident encoder's device. New
          # Parameter objects and new storage every time: candidates 2..N used to
          # inherit the previous candidate's TRAINED weights (Audit 030 §AD).
          # The candidate identity every downstream telemetry event carries.
          # Derived from the PRODUCTION plan, never hard-coded: "candidate 2/3
          # LR=3e-4" is whatever `schedule` actually says it is.
          candidate_index = list(schedule).index(planned) + 1
          telemetry_identity = {
              "stage": stage,
              "candidate_index": candidate_index,
              "candidate_count": len(schedule),
              "label": planned.label,
              "lr": planned.learning_rate,
              "r": planned.r,
              "seed": planned.seed,
          }
          # The candidate's ARCHITECTURE. Historical and C3 get the historical
          # fusion; C1 gets the scale-calibrated one. The parameter set and every
          # RNG draw are identical either way, so this run starts from exactly the
          # weights `expected_fresh_init_hash` predicts and differs only in the
          # equation that mixes them.
          adapter = fresh_adapter(
              hidden_size, provenance.init_seed, candidate.fusion.fusion_id
          )
          fresh_hash = trainable_state_hash(trainable_state(adapter))
          adapter.to(device)
          unmark_encoder = UnmarkEncoder(encoder=frozen_encoder, adapter=adapter)
          # ONE dispatch, shared with the real-model smoke path, on the SAME
          # weights object the artifact records. Whichever objective is built,
          # the adapter, the frozen backbone and the parameter count are
          # identical: 3 551 232 trainable.
          objective = build_candidate_objective(
              unmark_encoder, candidate, provenance.weights
          )

          placed_hash = trainable_state_hash(trainable_state(adapter))
          if placed_hash != fresh_hash:
              raise Stage1ContractViolation(
                  f"{planned.label}: moving the adapter to {device} changed its state "
                  f"({fresh_hash[:12]}... -> {placed_hash[:12]}...); initialisation must "
                  "be hardware-independent"
              )
          expected = expected_fresh_init_hash(hidden_size, planned.seed)
          if fresh_hash != expected:
              raise Stage1ContractViolation(
                  f"{planned.label}: fresh adapter hash {fresh_hash[:12]}... != the "
                  f"{expected[:12]}... that run_seed {planned.seed} (init_seed "
                  f"{provenance.init_seed}) must produce"
              )
          print(f"  {planned.label}: fresh adapter init_seed {provenance.init_seed} "
                f"hash {fresh_hash[:12]}...")
          corruption = CorruptionRatePolicy(seed=CORRUPTION_SEED)

          def evaluate_fn(update: int, _obj=objective) -> Any:
              return at_update(
                  evaluate(_obj, prepared_by_condition, pad_token_id, batch_size=BATCH_SIZE),
                  update,
              )

          # A checkpoint is used only when the operator asked to resume, and
          # `train_run` verifies its identity before touching it. The leg's cap
          # comes from that verified checkpoint: passing INITIAL_MAX_UPDATES
          # unconditionally meant a 40k continuation checkpoint resumed under a
          # 20k budget (Audit 031 B3 / Audit 032 B2).
          carried = load_training_checkpoint(run_checkpoints) if resume else None
          leg_cap = resume_cap(carried) if carried is not None else INITIAL_MAX_UPDATES
          # The candidate's screening budget, applied BEFORE any run_start event
          # so a hard-capped candidate handed a 40k leg (or a checkpoint already
          # past its ceiling) stops here with a policy error rather than deep in
          # the loop. `train_run` re-asserts this; it is the structural gate.
          resolve_run_cap(provenance, leg_cap, carried)
          sink.emit(
              "run_start",
              initial_global_update=int(carried["global_update"]) if carried else 0,
              cap=leg_cap,
              repository_head=repository_head,
              protocol_version=STAGE1_PROTOCOL_VERSION,
              init_seed=provenance.init_seed,
              corruption_seed=CORRUPTION_SEED,
              batch_size=BATCH_SIZE,
              train_chunks=len(train_text),
              resumed=carried is not None,
              # The candidate's full scientific identity, on the event the W&B
              # bridge turns into a run config. OPERATIONAL transport only: every
              # value originates in the production plan above.
              candidate_id=candidate.stage,
              **objective_identity.to_dict(),
              **candidate.fusion.to_dict(),
              hard_max_updates=candidate.budget.hard_max_updates,
              # `objective_identity.to_dict()` already carries `relational`, so a
              # candidate's full specification -- space, metric, reduction,
              # balance and epsilon -- reaches the dashboard config unchanged.
              wandb_project=candidate.wandb_project,
              **telemetry_identity,
          )
          result = train_run(
              objective=objective,
              provenance=provenance,
              train_chunks=train_text,
              tokenizer=tokenizer,
              corruption_policy=corruption,
              truncation=TRUNCATION,
              evaluate_fn=evaluate_fn,
              pad_token_id=pad_token_id,
              classifier=classifier,
              cap=leg_cap,
              checkpoint_dir=run_checkpoints,
              execution=execution,
              preparation_pool=preparation_pool,
              resume=carried,
              telemetry=sink,
              telemetry_identity=telemetry_identity,
          )
          # ONE continuation, and only out of a completed INITIAL leg. Gating on
          # `result.continued` was wrong once a resumed run could already be on
          # the 40k leg: `continued` records "this trajectory passed 20k", which
          # a continuation resume also sets, so it would re-enter here. The leg
          # that just ran is the authority -- after the extended leg there is no
          # successor, which is how "no 60k/80k extension" stays structural.
          if continuation_permitted(candidate, leg_cap, result.cap):
              # SAME run, continued -- not a new candidate. The locked budget rule
              # requires preserving adapter, optimizer, visit, cursor and streams
              # across the 20k boundary. This passed `resume=None` until the Audit
              # 030 F3 hardening, which rebuilt the optimizer and restarted the
              # sampler at visit 0 -- a continuation in name only. It now resumes
              # from the checkpoint the first leg wrote at exactly `cap`, so the
              # continuation uses the same mechanism as a crash resume.
              carried = load_training_checkpoint(run_checkpoints)
              if carried is None:
                  raise Stage1ContractViolation(
                      "the 20k leg produced no checkpoint to continue from; a "
                      "continuation must preserve optimizer and sampler state"
                  )
              result = train_run(
                  objective=objective,
                  provenance=provenance,
                  train_chunks=train_text,
                  tokenizer=tokenizer,
                  corruption_policy=corruption,
                  truncation=TRUNCATION,
                  evaluate_fn=evaluate_fn,
                  pad_token_id=pad_token_id,
                  classifier=classifier,
                  cap=result.cap,
                  resume=carried,
                  checkpoint_dir=run_checkpoints,
                  execution=execution,
                  preparation_pool=preparation_pool,
                  telemetry=sink,
                  telemetry_identity=telemetry_identity,
              )
          # The shared backbone must come out of this run exactly as it went in.
          require_frozen_backbone_unchanged(frozen_encoder, encoder_state_hash, planned.label)

          candidates.append(
              Candidate(
                  label=planned.label,
                  learning_rate=planned.learning_rate,
                  r=planned.r,
                  selected=result.selected,
                  budget_limited=result.budget_limited,
              )
          )
          (output_dir / f"run-{planned.label.replace('=', '')}.json").write_text(
              json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
          )

    # The identity every downstream consumer validates against its OWN current
    # inputs. Recorded as one block so a consumer cannot check three fields and
    # forget the fourth (Audit 031 B4 / Audit 032 MAJ1). The two legacy
    # top-level keys are kept so existing readers and artifacts stay readable.
    # `campaign` was built once above; telemetry and the artifact therefore
    # cannot disagree about what campaign this is.
    artifact = {
        "stage": stage,
        "protocol_version": STAGE1_PROTOCOL_VERSION,
        "identity": campaign.to_dict(),
        # WHICH loss produced these candidates, and the grid weight it implies.
        # Recorded at the top level so the objective is recoverable from the
        # stage artifact alone, without opening a checkpoint.
        "objective": objective_identity.to_dict(),
        "repository_head": repository_head,
        "corpus_manifest_digest": manifest_digest,
        "candidates": [c.to_dict() for c in candidates],
        # OPERATIONAL provenance, deliberately separate from RunProvenance:
        # prepared output is byte-identical across worker counts, so this changes
        # wall-clock and nothing scientific, and is not resume-blocking.
        "preparation": provenance_of_preparation,
        "raw_text_persisted": False,
        "official_test_used": False,
        "downstream_score_used": False,
    }
    if stage == "lr_pilot":
        with phase(sink, "selection", stage=stage):
            artifact["selected"] = select_learning_rate(candidates).to_dict()
        # The PRODUCTION selection result, not a re-derivation.
        sink.emit("selection", stage=stage, selected=artifact["selected"])
    elif stage == "r_phase1":
        frozen = candidates[0].learning_rate
        with phase(sink, "selection", stage=stage):
            artifact["selected"] = select_r(candidates, frozen).to_dict()
        sink.emit("selection", stage=stage, selected=artifact["selected"])
    elif stage in FIRST_SCREEN_STAGES:
        # NO SELECTION. V2-GC is one run, so there is no candidate to choose
        # between -- and the checkpoint WITHIN it is still chosen by the locked
        # held-out rule in `select_checkpoint`, which this stage does not touch.
        # The grid term is a training term and a logged diagnostic; it selects
        # nothing, and no downstream label or Macro-F1 enters here.
        # NOTE: `candidate` here is the StageCandidate resolved from the stage
        # name; the screened run is `candidates[0]`. The two are deliberately
        # named apart -- conflating them is how a budget block could end up
        # describing the wrong thing.
        screened = candidates[0]
        artifact[stage] = {
            "note": (
                "post-hoc research candidate: the historical objective plus a "
                "token-grid consistency term, ONE run, no selection performed here"
            ),
            "objective": objective_identity.to_dict(),
            # The ENFORCED screening budget, not a restated constant. A reader
            # can tell from the artifact alone that no 40k leg was permitted.
            "budget": candidate.budget.to_dict(),
            "label": screened.label,
            "learning_rate": screened.learning_rate,
            "r": screened.r,
            "selected": screened.selected.to_dict(),
            "budget_limited": screened.budget_limited,
        }
    else:
        from unmark.stage1.selection import descriptive_summary

        artifact["final_main"] = {
            "note": "these three adapters ARE the final main Stage-1 models",
            "score": descriptive_summary([c.selected.score for c in candidates]),
            "d_clean": descriptive_summary([c.selected.d_clean for c in candidates]),
        }
    (output_dir / f"{stage}.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"\nWrote {output_dir}/{stage}.json")
    sink.emit(
        "stage_complete",
        stage=stage,
        artifact_path=str(output_dir / f"{stage}.json"),
        candidate_count=len(candidates),
        identity=campaign.to_dict(),
    )
    return 0


def _resident_bytes() -> int | None:
    """Process RSS, for the Audit 030 F4 measurement. Linux only; None elsewhere."""
    try:
        with open("/proc/self/statm", encoding="ascii") as handle:
            return int(handle.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except Exception:  # noqa: BLE001 - measurement only, never load-bearing
        return None


def smoke_check(
    *,
    prepared_corpus: Path,
    revision: str,
    repository_head: str | None,
    completion_dir: Path | None = None,
    stage: str = HISTORICAL_SMOKE_STAGE,
) -> int:
    """No-update real-model integration check, for ONE candidate.

    **Constructs no optimizer and calls no `.backward()`.** It reports the model
    contract and one forward pass, and cannot change a parameter. No checkpoint
    is selected, no update is taken, no official validation or TEST is touched,
    and no downstream label exists anywhere on this path.

    Verifies the prepared corpus **before the model is loaded** (Audit 030 F1),
    so the smoke exercises exactly the gate a training run will pass through.

    `stage` names the candidate to exercise and is resolved by the **same**
    `candidates.candidate_for_stage` the real run uses, then built by the **same**
    `build_candidate_objective`. Smoking `v2_gc` therefore runs a real
    `GridConsistencyObjective` forward on real PhoBERT over the real adapter and
    a real prepared batch -- which Audit 065 BLOCKER 4 found was impossible,
    because this path constructed `Stage1Objective` unconditionally.

    Every loss term the candidate's objective produces is asserted finite before
    anything is reported, so a smoke cannot pass on a NaN.
    """
    import time

    import torch

    from unmark.linguistics import make_classifier, try_load_inventory
    from unmark.stage1.checkpoint import verify_prepared_corpus
    from unmark.stage1.data import (
        Stage1Example,
        batch_to_device,
        collate_stage1_batch,
        module_device,
        prepare_example,
    )
    from unmark.stage1.preflight import verify_scientific_inputs

    inputs = verify_scientific_inputs()
    print(f"scientific inputs VERIFIED: eligibility {inputs.report['eligibility_policy']}")
    print(f"  inventory sha256 {inputs.inventory.sha256}")

    completion = Path(completion_dir) if completion_dir else Path(prepared_corpus) / "_checkpoint"
    verified = verify_prepared_corpus(Path(prepared_corpus), completion)
    print(f"prepared corpus VERIFIED against {verified.completion_path}")
    print(f"  chunk_membership_digest {verified.chunk_membership_digest}")
    print(f"  counts {json.dumps(verified.counts, sort_keys=True)}")

    # Audit 030 F4 is measured here, on the real corpus, and nowhere else.
    started = time.monotonic()
    train_text, dev_text = load_prepared_chunks(prepared_corpus)
    print(f"  loaded {len(train_text)} train and {len(dev_text)} dev chunks in "
          f"{time.monotonic() - started:.1f}s")
    resident = _resident_bytes()
    if resident is not None:
        print(f"  process RSS after load: {resident / 1e9:.2f} GB")
    # `build_objective` still supplies the pinned tokenizer, the frozen backbone
    # and the locked adapter; the objective itself comes from the shared
    # candidate dispatch, so the class it returns is never the thing that decides
    # which loss is smoked.
    candidate = candidate_for_stage(stage)
    tokenizer, unmark_encoder, _historical_objective_cls = build_objective(
        revision, fusion_id=candidate.fusion.fusion_id
    )
    contract = verify_model_contract(unmark_encoder)
    objective = build_candidate_objective(
        unmark_encoder, candidate, lambdas_to_weights(LR_PILOT_R)
    )
    print(f"smoking candidate {candidate.stage!r}: objective "
          f"{candidate.objective.objective_id}")
    corruption = CorruptionRatePolicy(seed=CORRUPTION_SEED)
    classifier = make_classifier(try_load_inventory())

    sample = sorted(train_text)[:8]
    prepared = [
        prepare_example(
            Stage1Example(text=train_text[cid], sample_id=cid),
            tokenizer, corruption_policy=corruption, truncation=TRUNCATION,
            visit=0, classifier=classifier,
        )
        for cid in sample
    ]
    # The same one boundary `evaluate` and `train_run` use: the batch follows the
    # model, derived from the objective's own parameters. A no-op on CPU.
    usable = [p for p in prepared if p is not None]
    # Relational geometry is defined only BETWEEN examples, so a candidate with a
    # relational term needs a real batch. Checked before the forward, so the
    # smoke reports a policy error rather than a shape error from inside a loss.
    if candidate.objective.relational is not None and len(usable) < 2:
        raise Stage1ContractViolation(
            f"{candidate.stage!r} smokes a relational objective, which needs at least "
            f"2 examples to have an off-diagonal pair; the batch has {len(usable)}"
        )
    batch = batch_to_device(
        collate_stage1_batch(usable, tokenizer.pad_token_id),
        module_device(objective),
    )
    with torch.no_grad():
        result = objective(batch)
    losses = result.to_dict()

    # Every term the candidate produced must be a real number. A smoke that
    # prints NaN and returns 0 is not a check. `loss_grid`/`mean_distance_grid`
    # are present exactly when the candidate's objective has a grid term, so this
    # also proves the dispatch above really built what the register promised.
    required = ["loss", "loss_align", "loss_clean"]
    if candidate.objective.lambda_grid is not None:
        required += ["loss_grid", "mean_distance_grid"]
    if candidate.objective.lambda_grd is not None:
        required += ["loss_grd", "loss_rel_clean", "loss_rel_corrupt"]
    for key in required:
        value = losses.get(key)
        if not isinstance(value, (int, float)) or value != value or value in (
            float("inf"), float("-inf")
        ):
            raise Stage1ContractViolation(
                f"smoke for {candidate.stage!r} produced {key}={value!r}, which is not "
                "finite. The candidate's objective is not usable for training."
            )
    # PASSIVE C1 diagnostics, read off the adapter's last forward. No extra
    # encoder forward, no graph, no RNG: `scale_diagnostics()` returns `{}` for
    # any adapter that is not scale-calibrated.
    scale = unmark_encoder.adapter.scale_diagnostics()
    for name, value in scale.items():
        if value != value or value in (float("inf"), float("-inf")):
            raise Stage1ContractViolation(
                f"smoke for {candidate.stage!r} produced a non-finite scale "
                f"diagnostic {name}={value!r}"
            )
    print(json.dumps({
        "smoke": "STAGE1_NO_UPDATE_FORWARD_ONLY",
        "repository_head": repository_head,
        "candidate": candidate.to_dict(),
        "scale_diagnostics": scale,
        "model_contract": contract,
        "losses": losses,
        "loss_terms_verified_finite": required,
        "optimizer_constructed": False,
        "backward_called": False,
        "parameters_updated": 0,
        "checkpoint_selected": False,
        "official_validation_used": False,
        "official_test_used": False,
        "downstream_label_used": False,
    }, indent=2, sort_keys=True))
    return 0


def lambdas_to_weights(r: float):
    from unmark.stage1.contracts import ObjectiveWeights

    lambda_align, lambda_clean = lambdas_for_r(r)
    return ObjectiveWeights(lambda_align=lambda_align, lambda_clean=lambda_clean)
