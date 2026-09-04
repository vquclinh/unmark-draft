"""Operational accelerators for Stage-1 selection runs.

The fused r-phase1 path is an execution optimization, not a new protocol. It is
valid only for the locked r-phase1 schedule, whose candidates deliberately share
one run seed and one frozen learning rate. The main process consumes one sampler
batch, prepares it once, proves every active candidate would have consumed the
same `(chunk_id, visit)` pairs, then applies each candidate's independent
adapter, optimizer, checkpoint and validation state.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from unmark.stage1.contracts import (
    Stage1ContractViolation,
)
from unmark.stage1.protocol import (
    BATCH_SIZE,
    CORRUPTION_SEED,
    EVAL_EVERY_UPDATES,
    INITIAL_MAX_UPDATES,
    R_PHASE1_GRID,
    STAGE1_PROTOCOL_VERSION,
    adapter_init_seed,
)
from unmark.stage1.sampler import DeterministicSampler
from unmark.stage1.selection import (
    Candidate,
    PlannedRun,
    ValidationPoint,
    select_checkpoint,
)
from unmark.stage1.telemetry import TelemetrySink
from unmark.stage1.trainer import (
    CHECKPOINT_EVERY_UPDATES,
    MonitorWindow,
    RunProvenance,
    RunResult,
    checkpoint_payload,
    gradient_report,
    load_training_checkpoint,
    require_optimizer_parameter_identity,
    require_optimizer_state_device,
    require_resumable_leg,
    resolve_budget,
    resume_cap,
    save_training_checkpoint,
    verify_checkpoint,
    verify_model_contract,
)

R_PHASE1_EXECUTION_ENV = "UNMARK_STAGE1_R_PHASE1_EXECUTION"
R_PHASE1_EXECUTION_SEQUENTIAL = "sequential"
R_PHASE1_EXECUTION_FUSED = "fused"
R_PHASE1_EXECUTION_MODES = (R_PHASE1_EXECUTION_SEQUENTIAL, R_PHASE1_EXECUTION_FUSED)


def resolve_r_phase1_execution(
    env: Mapping[str, str] | None = None,
) -> str:
    """Resolve the operational r-phase1 execution mode."""
    environment = os.environ if env is None else env
    value = str(
        environment.get(R_PHASE1_EXECUTION_ENV, R_PHASE1_EXECUTION_SEQUENTIAL)
    ).strip().lower()
    if value in ("", "0", "false", "off", "no"):
        return R_PHASE1_EXECUTION_SEQUENTIAL
    if value not in R_PHASE1_EXECUTION_MODES:
        raise Stage1ContractViolation(
            f"{R_PHASE1_EXECUTION_ENV} must be one of "
            f"{list(R_PHASE1_EXECUTION_MODES)}, got {value!r}"
        )
    return value


def require_fused_r_phase1_schedule(schedule: Sequence[PlannedRun]) -> None:
    """Fail closed unless the schedule is exactly the fusable r-phase1 plan."""
    planned = list(schedule)
    if len(planned) != len(R_PHASE1_GRID):
        raise Stage1ContractViolation(
            f"fused r-phase1 requires {len(R_PHASE1_GRID)} candidates, got "
            f"{len(planned)}"
        )
    stages = {p.stage for p in planned}
    if stages != {"r_phase1"}:
        raise Stage1ContractViolation(
            f"fused execution is only defined for r_phase1, got {sorted(stages)}"
        )
    rates = {p.learning_rate for p in planned}
    seeds = {p.seed for p in planned}
    if len(rates) != 1 or None in rates:
        raise Stage1ContractViolation(
            f"fused r-phase1 requires one frozen learning rate, got {sorted(rates)}"
        )
    if len(seeds) != 1:
        raise Stage1ContractViolation(
            f"fused r-phase1 requires one shared selection seed, got {sorted(seeds)}"
        )
    values = sorted(float(p.r) for p in planned if p.r is not None)
    if values != sorted(R_PHASE1_GRID):
        raise Stage1ContractViolation(
            f"fused r-phase1 requires the locked r grid {list(R_PHASE1_GRID)}, "
            f"got {values}"
        )


@dataclass
class _Slot:
    planned: PlannedRun
    candidate_index: int
    checkpoint_dir: Path
    provenance: RunProvenance
    adapter: Any
    objective: Any
    optimizer: Any
    sampler: DeterministicSampler
    result: RunResult
    global_update: int
    telemetry_identity: dict[str, Any]
    model_contract: dict[str, Any]
    window: MonitorWindow
    started_from_checkpoint: bool
    last_loss: dict[str, float] | None = None
    finalized: bool = False


def train_fused_r_phase1(
    *,
    schedule: Sequence[PlannedRun],
    train_chunks: dict[str, str],
    tokenizer: Any,
    frozen_encoder: Any,
    hidden_size: int,
    encoder_state_hash: str,
    prepared_by_condition: dict[str, Sequence[Any]],
    pad_token_id: int,
    device: Any,
    execution: Any,
    manifest_digest: str,
    repository_head: str,
    inventory: Any,
    output_dir: Path,
    preparation_pool: Any,
    resume: bool,
    telemetry: TelemetrySink,
) -> list[Candidate]:
    """Train the locked r-phase1 grid with shared preparation work.

    Candidate state remains independent. Only the already-chosen batch
    preparation is shared, and only after every active candidate proves it would
    have consumed the same sampler pairs.
    """
    if not train_chunks:
        raise Stage1ContractViolation("no training chunks supplied")
    require_fused_r_phase1_schedule(schedule)

    from unmark.modeling.adapter import UnmarkEncoder
    from unmark.stage1.data import (
        batch_to_device,
        collate_stage1_batch,
        module_device,
    )
    from unmark.stage1.initialisation import (
        expected_fresh_init_hash,
        fresh_adapter,
        trainable_state,
        trainable_state_hash,
    )
    from unmark.stage1.objective import Stage1Objective
    from unmark.stage1.optim import build_optimizer
    from unmark.stage1.validation import at_update, evaluate

    chunk_ids = tuple(sorted(train_chunks))

    def evaluate_fn(slot: _Slot, update: int) -> ValidationPoint:
        return at_update(
            evaluate(
                slot.objective,
                prepared_by_condition,
                pad_token_id,
                batch_size=BATCH_SIZE,
            ),
            update,
        )

    slots: list[_Slot] = []
    for candidate_index, planned in enumerate(schedule, start=1):
        if planned.learning_rate is None or planned.r is None:
            raise Stage1ContractViolation(
                f"{planned.label}: fused r-phase1 requires concrete LR and r"
            )
        run_checkpoints = (
            output_dir / f"run-{planned.label.replace('=', '')}" / "_checkpoint"
        )
        provenance = RunProvenance(
            run_seed=planned.seed,
            init_seed=adapter_init_seed(planned.seed),
            corruption_seed=CORRUPTION_SEED,
            learning_rate=planned.learning_rate,
            r=planned.r,
            corpus_manifest_digest=manifest_digest,
            repository_head=repository_head,
            inventory=inventory,
        )
        telemetry_identity = {
            "stage": "r_phase1",
            "candidate_index": candidate_index,
            "candidate_count": len(schedule),
            "label": planned.label,
            "lr": planned.learning_rate,
            "r": planned.r,
            "seed": planned.seed,
        }

        adapter = fresh_adapter(hidden_size, provenance.init_seed)
        fresh_hash = trainable_state_hash(trainable_state(adapter))
        adapter.to(device)
        objective = Stage1Objective(
            UnmarkEncoder(encoder=frozen_encoder, adapter=adapter),
            provenance.weights,
        )
        placed_hash = trainable_state_hash(trainable_state(adapter))
        if placed_hash != fresh_hash:
            raise Stage1ContractViolation(
                f"{planned.label}: moving the adapter to {device} changed its "
                f"state ({fresh_hash[:12]}... -> {placed_hash[:12]}...)"
            )
        expected = expected_fresh_init_hash(hidden_size, planned.seed)
        if fresh_hash != expected:
            raise Stage1ContractViolation(
                f"{planned.label}: fresh adapter hash {fresh_hash[:12]}... != "
                f"{expected[:12]}..."
            )
        print(
            f"  {planned.label}: fused fresh adapter init_seed "
            f"{provenance.init_seed} hash {fresh_hash[:12]}..."
        )

        model_contract = verify_model_contract(objective.unmark_encoder)
        optimizer = build_optimizer(
            [(n, p) for n, p in objective.unmark_encoder.named_parameters()
             if p.requires_grad],
            provenance.learning_rate,
        )
        require_optimizer_parameter_identity(optimizer, adapter)
        sampler = DeterministicSampler(chunk_ids, seed=provenance.run_seed)
        carried = load_training_checkpoint(run_checkpoints) if resume else None
        leg_cap = resume_cap(carried) if carried is not None else INITIAL_MAX_UPDATES
        result = RunResult(provenance=provenance, cap=leg_cap)
        global_update = 0

        if carried is not None:
            verify_checkpoint(carried, provenance)
            require_resumable_leg(carried, leg_cap)
            if execution is not None and carried.get("execution") is not None:
                execution.require_compatible(carried["execution"])
            adapter.load_state_dict(carried["adapter_state"], strict=True)
            optimizer.load_state_dict(carried["optimizer_state"])
            require_optimizer_parameter_identity(optimizer, adapter)
            require_optimizer_state_device(optimizer, module_device(objective))
            sampler = DeterministicSampler.from_state(
                chunk_ids, carried["sampler_state"]
            )
            global_update = int(carried["global_update"])
            result.points = [
                ValidationPoint.from_dict(p) for p in carried.get("points", [])
            ]
            result.continued = leg_cap > INITIAL_MAX_UPDATES

        slot = _Slot(
            planned=planned,
            candidate_index=candidate_index,
            checkpoint_dir=run_checkpoints,
            provenance=provenance,
            adapter=adapter,
            objective=objective,
            optimizer=optimizer,
            sampler=sampler,
            result=result,
            global_update=global_update,
            telemetry_identity=telemetry_identity,
            model_contract=model_contract,
            window=MonitorWindow(),
            started_from_checkpoint=carried is not None,
        )
        if not any(p.update == 0 for p in slot.result.points):
            slot.result.points.append(evaluate_fn(slot, 0))
        slot.objective.train(True)
        _emit_run_context(
            telemetry,
            slot,
            repository_head=repository_head,
            train_chunks=len(train_chunks),
        )
        slots.append(slot)

    with _phase(telemetry, "r_phase1_fused_training"):
        while True:
            _promote_or_finalize_ready_slots(
                slots,
                frozen_encoder=frozen_encoder,
                encoder_state_hash=encoder_state_hash,
                output_dir=output_dir,
                telemetry=telemetry,
                repository_head=repository_head,
                train_chunks=len(train_chunks),
            )
            active = [s for s in slots if not s.finalized and s.global_update < s.result.cap]
            if not active:
                break
            _require_common_active_state(active)
            pairs = active[0].sampler.next_batch(BATCH_SIZE)
            tasks = [
                (chunk_id, visit, train_chunks[chunk_id])
                for chunk_id, visit in pairs
            ]
            for slot in active[1:]:
                observed = slot.sampler.next_batch(BATCH_SIZE)
                if observed != pairs:
                    raise Stage1ContractViolation(
                        "fused r-phase1 detected divergent sampler streams. "
                        f"{active[0].planned.label} and {slot.planned.label} "
                        "would not train on the same batch, so sharing "
                        "preparation would change the experiment."
                    )

            prepared = preparation_pool.prepare(tasks)
            for (chunk_id, _visit), item in zip(pairs, prepared):
                if item is None:
                    raise Stage1ContractViolation(
                        f"chunk {chunk_id!r} overflowed at training time; after "
                        "correct pre-chunking this cannot happen"
                    )
            batch = batch_to_device(
                collate_stage1_batch(prepared, pad_token_id),
                module_device(active[0].objective),
            )

            for slot in active:
                for item in prepared:
                    slot.window.tone_channel_differs += int(item.channels_differ)
                    slot.window.letter_channel_differs += int(
                        item.letter_channels_differ
                    )
                loss_result = slot.objective(batch)
                slot.optimizer.zero_grad(set_to_none=True)
                loss_result.loss.backward()
                grads = gradient_report(slot.objective.unmark_encoder)
                slot.window.batches += 1
                for name, norm in grads["adapter_group_grad_norms"].items():
                    lowered = name.lower()
                    if "tone" in lowered and norm > 0:
                        slot.window.tone_embedding_grad_batches += 1
                    if "letter" in lowered and norm > 0:
                        slot.window.letter_embedding_grad_batches += 1
                slot.optimizer.step()
                slot.global_update += 1
                slot.last_loss = {
                    "loss": float(loss_result.loss.detach()),
                    "loss_align": float(loss_result.loss_align.detach()),
                    "loss_clean": float(loss_result.loss_clean.detach()),
                }

            update = active[0].global_update
            cap = active[0].result.cap
            if update % EVAL_EVERY_UPDATES == 0 or update == cap:
                for slot in active:
                    _emit_run_context(
                        telemetry,
                        slot,
                        repository_head=repository_head,
                        train_chunks=len(train_chunks),
                    )
                    if slot.last_loss is not None:
                        telemetry.emit(
                            "train_progress",
                            global_update=slot.global_update,
                            cap=slot.result.cap,
                            batch_size=BATCH_SIZE,
                            visit=slot.sampler.visit,
                            position=slot.sampler.position,
                            execution_mode=R_PHASE1_EXECUTION_FUSED,
                            **slot.last_loss,
                            **slot.telemetry_identity,
                        )
                    point = evaluate_fn(slot, slot.global_update)
                    slot.result.points.append(point)
                    telemetry.emit(
                        "validation",
                        cap=slot.result.cap,
                        execution_mode=R_PHASE1_EXECUTION_FUSED,
                        **point.to_dict(),
                        **slot.telemetry_identity,
                    )
                    slot.objective.train(True)
                    if (
                        slot.global_update % CHECKPOINT_EVERY_UPDATES == 0
                        or slot.global_update == slot.result.cap
                    ):
                        is_best = (
                            select_checkpoint(slot.result.points).update
                            == slot.global_update
                        )
                        published = save_training_checkpoint(
                            slot.checkpoint_dir,
                            checkpoint_payload(
                                provenance=slot.provenance,
                                adapter_state=slot.adapter.state_dict(),
                                optimizer_state=slot.optimizer.state_dict(),
                                global_update=slot.global_update,
                                sampler_state=slot.sampler.state_dict(),
                                cap=slot.result.cap,
                                budget_limited=slot.result.budget_limited,
                                points=slot.result.points,
                                execution=execution.to_dict()
                                if execution is not None else None,
                            ),
                            is_best=is_best,
                        )
                        telemetry.emit(
                            "checkpoint",
                            update=slot.global_update,
                            cap=slot.result.cap,
                            is_best=is_best,
                            continued=slot.result.continued,
                            checkpoint_name=Path(published).name,
                            checkpoint_dir=str(slot.checkpoint_dir),
                            execution_mode=R_PHASE1_EXECUTION_FUSED,
                            **slot.telemetry_identity,
                        )
                    slot.window = MonitorWindow()

    return [
        Candidate(
            label=slot.planned.label,
            learning_rate=slot.planned.learning_rate,
            r=slot.planned.r,
            selected=slot.result.selected,
            budget_limited=slot.result.budget_limited,
        )
        for slot in slots
    ]


def _require_common_active_state(slots: Sequence[_Slot]) -> None:
    """Active fused candidates must share the exact cursor and budget leg."""
    base = slots[0]
    expected = (
        base.global_update,
        base.result.cap,
        base.sampler.visit,
        base.sampler.position,
    )
    for slot in slots[1:]:
        actual = (
            slot.global_update,
            slot.result.cap,
            slot.sampler.visit,
            slot.sampler.position,
        )
        if actual != expected:
            raise Stage1ContractViolation(
                "fused r-phase1 can resume only candidates at the same "
                f"update/cap/cursor. {base.planned.label} has {expected}; "
                f"{slot.planned.label} has {actual}. Resume this output "
                "sequentially or from a checkpoint boundary where the active "
                "candidates are aligned."
            )


def _promote_or_finalize_ready_slots(
    slots: Sequence[_Slot],
    *,
    frozen_encoder: Any,
    encoder_state_hash: str,
    output_dir: Path,
    telemetry: TelemetrySink,
    repository_head: str,
    train_chunks: int,
) -> None:
    from unmark.stage1.execute import require_frozen_backbone_unchanged

    for slot in slots:
        if slot.finalized or slot.global_update < slot.result.cap:
            continue
        old_cap = slot.result.cap
        resolve_budget(slot.result)
        if slot.result.cap > old_cap:
            continue
        require_frozen_backbone_unchanged(
            frozen_encoder, encoder_state_hash, slot.planned.label
        )
        (output_dir / f"run-{slot.planned.label.replace('=', '')}.json").write_text(
            json.dumps(slot.result.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _emit_run_context(
            telemetry,
            slot,
            repository_head=repository_head,
            train_chunks=train_chunks,
        )
        selected = slot.result.selected
        telemetry.emit(
            "run_end",
            global_update=slot.global_update,
            cap=slot.result.cap,
            continued_past_initial_budget=slot.result.continued,
            budget_limited=slot.result.budget_limited,
            evaluations=len(slot.result.points),
            selected_update=selected.update,
            selected_score=selected.score,
            selected_d_clean=selected.d_clean,
            selected_distances=dict(selected.distances),
            execution_mode=R_PHASE1_EXECUTION_FUSED,
            **slot.telemetry_identity,
        )
        slot.finalized = True


def _emit_run_context(
    sink: TelemetrySink,
    slot: _Slot,
    *,
    repository_head: str,
    train_chunks: int,
) -> None:
    sink.emit(
        "run_start",
        initial_global_update=slot.global_update,
        cap=slot.result.cap,
        repository_head=repository_head,
        protocol_version=STAGE1_PROTOCOL_VERSION,
        init_seed=slot.provenance.init_seed,
        corruption_seed=CORRUPTION_SEED,
        batch_size=BATCH_SIZE,
        train_chunks=train_chunks,
        resumed=slot.started_from_checkpoint,
        execution_mode=R_PHASE1_EXECUTION_FUSED,
        **slot.telemetry_identity,
    )


class _phase:
    """Minimal local phase helper; keeps fused.py independent of execute.py."""

    def __init__(self, sink: TelemetrySink, name: str) -> None:
        self.sink = sink
        self.name = name

    def __enter__(self) -> None:
        self.sink.emit(
            "stage_phase",
            phase=self.name,
            state="START",
            execution_mode=R_PHASE1_EXECUTION_FUSED,
        )

    def __exit__(self, exc_type, exc, tb) -> bool:
        state = "FAILED" if exc_type is not None else "DONE"
        self.sink.emit(
            "stage_phase",
            phase=self.name,
            state=state,
            execution_mode=R_PHASE1_EXECUTION_FUSED,
            error_type=exc_type.__name__ if exc_type is not None else None,
        )
        return False
