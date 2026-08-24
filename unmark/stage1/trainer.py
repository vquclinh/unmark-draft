"""The Stage-1 training loop. **Imports torch lazily; NOT executed in Audit 029.**

Implements exactly the locked protocol (D-S1B-003, D-S1B-004). Everything that
can be decided without tensors lives in `selection`, `sampler`, `optim` and
`protocol`, so this module is the loop and the bookkeeping only.

Three invariants the loop must never violate, each checked rather than assumed:

* the **encoder is frozen** -- it is never handed to the optimizer, stays in
  eval, and its gradient must remain absent;
* **resume is exact** -- adapter, optimizer, `visit` and the in-pass cursor are
  all checkpointed, so a resumed run is scientifically the run it continues;
* **the budget is precommitted** -- one continuation from 20 000 to 40 000 and
  then `BUDGET_LIMITED`, never an open-ended extension.
"""

from __future__ import annotations

import os

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Sequence

if TYPE_CHECKING:  # imported lazily: `trainer` must stay importable without PyYAML
    from unmark.stage1.preflight import InventoryIdentity

from unmark.stage1.contracts import (
    ObjectiveWeights,
    Stage1ContractViolation,
    TruncationPolicy,
)
from unmark.stage1.protocol import (
    ADAPTER_TRAINABLE_PARAMETERS,
    BATCH_SIZE,
    ENCODER_CHECKPOINT,
    ENCODER_REVISION,
    EVAL_EVERY_UPDATES,
    EXTENDED_MAX_UPDATES,
    GRADIENT_ACCUMULATION_STEPS,
    GRADIENT_CLIPPING,
    HIDDEN_SIZE,
    INITIAL_MAX_UPDATES,
    PRECISION,
    STAGE1_PROTOCOL_VERSION,
    lambdas_for_r,
)
from unmark.stage1.sampler import DeterministicSampler
from unmark.stage1.selection import ValidationPoint, budget_decision, select_checkpoint

CHECKPOINT_SCHEMA_VERSION = "stage1-checkpoint-v2"
"""**v2** (Audit 030 §AE): `adapter_state` is now the ADAPTER's own `state_dict`
— trainable parameters only — restored with `strict=True`, and the payload
carries `execution` and `init_seed`. v1 stored the whole `UnmarkEncoder`,
including the frozen encoder, and restored it with `strict=False`, so a key
mismatch silently restored nothing.

No migration path is offered and none is needed: **no scientific Stage-1 campaign
has ever run**, so no v1 checkpoint of scientific value exists. A v1 payload now
fails closed rather than being reinterpreted under v2 semantics."""


class TrainerContractViolation(Stage1ContractViolation):
    """Raised when the training environment contradicts the locked protocol."""


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RunProvenance:
    """Everything a resume must match. A mismatch fails closed."""

    run_seed: int
    """Seeds the data order — `DeterministicSampler(seed=run_seed)`. **Unchanged**
    by D-S1B-016: the existing realization is preserved exactly."""
    init_seed: int
    """Seeds the adapter's initial weights, domain-separated from `run_seed` via
    `protocol.adapter_init_seed` (D-S1B-016). Part of scientific identity because
    it determines the state the run starts from; a GPU name is not, and lives in
    the operational execution fingerprint instead."""
    corruption_seed: int
    learning_rate: float
    r: float
    corpus_manifest_digest: str
    repository_head: str | None
    backbone_checkpoint: str = ENCODER_CHECKPOINT
    backbone_revision: str = ENCODER_REVISION
    protocol_version: str = STAGE1_PROTOCOL_VERSION
    precision: str = PRECISION
    inventory: "InventoryIdentity | None" = None
    """The pinned Vietnamese syllable inventory this run resolved eligibility with.

    **D-S1A-008**, whose status line reads "BLOCKING for scientific Stage-1
    training and the PRE-TRAIN audit": the inventory decides which spans are
    eligible and therefore every corruption denominator and channel projection,
    so "a training run whose artifact cannot name the inventory it used is not
    reproducible in the sense the project has held itself to everywhere else".

    Optional on the dataclass because the diagnostic and unit-test paths
    construct provenance without one; a scientific run cannot, because
    `execute_stage` obtains it from `verify_scientific_inputs`, which fails
    closed rather than returning.
    """

    DERIVED_KEYS: ClassVar[tuple[str, ...]] = ("lambda_align", "lambda_clean")
    """Keys `to_dict()` emits that are **not** constructor parameters.

    They are computed from `r` by `lambdas_for_r`, and recorded so an artifact
    states the weights its objective actually used instead of making a reader
    recompute them. Because they are derived, `to_dict()` is deliberately not
    constructor-round-trippable -- see `to_dict`.
    """

    @property
    def weights(self) -> ObjectiveWeights:
        lambda_align, lambda_clean = lambdas_for_r(self.r)
        return ObjectiveWeights(lambda_align=lambda_align, lambda_clean=lambda_clean)

    def to_dict(self) -> dict[str, Any]:
        """The **artifact** form. Deliberately NOT constructor-round-trippable.

        It carries the two `DERIVED_KEYS` on top of the constructor fields, so
        `RunProvenance(**p.to_dict())` raises `TypeError`. That is the contract,
        not an oversight: a run's identity comes from its *plan* and its
        environment, and is never rebuilt from the artifact it is trying to
        resume -- otherwise a corrupted or foreign checkpoint could define which
        experiment it belongs to instead of merely failing to match one. The
        only authoritative direction is `require_match`, which compares a
        recorded dict against a freshly constructed identity. To derive one
        provenance from another, use `dataclasses.replace`.
        """
        return {
            "run_seed": self.run_seed,
            "init_seed": self.init_seed,
            "corruption_seed": self.corruption_seed,
            "learning_rate": self.learning_rate,
            "r": self.r,
            **self.weights.to_dict(),
            "corpus_manifest_digest": self.corpus_manifest_digest,
            "repository_head": self.repository_head,
            "backbone_checkpoint": self.backbone_checkpoint,
            "backbone_revision": self.backbone_revision,
            "protocol_version": self.protocol_version,
            "precision": self.precision,
            "inventory": self.inventory.to_dict() if self.inventory is not None else None,
        }

    def require_match(self, other: dict[str, Any]) -> None:
        """Refuse to resume into a different experiment."""
        mine = self.to_dict()
        for key in (
            "run_seed", "init_seed", "corruption_seed", "learning_rate", "r",
            "corpus_manifest_digest", "backbone_checkpoint", "backbone_revision",
            "protocol_version", "precision",
            # `repository_head` was recorded but never compared until the Audit
            # 030 F3 hardening, so a checkpoint written by one commit could be
            # resumed by another whose trainer, objective or corruption code had
            # changed underneath it. Stage 6 spent three revisions establishing
            # that HEAD is identity, not decoration; training is no different.
            "repository_head",
            # D-S1A-008. Two runs that differ only in which syllable inventory
            # resolved eligibility are different experiments: the denominator of
            # every corruption rate changes. Compared as a whole so the message
            # shows both identities at once.
            "inventory",
        ):
            if other.get(key) != mine[key]:
                raise TrainerContractViolation(
                    f"checkpoint provenance mismatch on {key!r}: checkpoint has "
                    f"{other.get(key)!r}, this environment has {mine[key]!r}. Resuming "
                    "would silently continue a different experiment."
                )

        # The recorded weights are DERIVED from `r`, so an honestly written
        # artifact can never disagree with its own ratio -- but a corrupted,
        # truncated or hand-edited one can, and nothing anywhere reads these two
        # keys back. Without this they would be the only scientific quantity a
        # checkpoint carries that no gate ever checks. `r` is proven equal above,
        # so `mine` holds exactly what `other`'s own `r` must derive.
        for key in self.DERIVED_KEYS:
            if key not in other:
                raise TrainerContractViolation(
                    f"checkpoint provenance is missing the derived key {key!r}; it was "
                    "not written by this repository's serializer and cannot be trusted "
                    "to describe the objective it was trained under."
                )
            if other[key] != mine[key]:
                raise TrainerContractViolation(
                    f"checkpoint provenance is internally inconsistent: it records "
                    f"r={other['r']!r} and {key}={other[key]!r}, but r={other['r']!r} "
                    f"derives {key}={mine[key]!r} under lambdas_for_r. The artifact "
                    "misdescribes its own objective; it will not be resumed."
                )


# ---------------------------------------------------------------------------
# Model contract
# ---------------------------------------------------------------------------
def verify_model_contract(unmark_encoder: Any) -> dict[str, Any]:
    """Assert the frozen-encoder / trainable-adapter split. **Lazy torch.**"""
    import torch
    from torch import nn

    encoder = unmark_encoder.encoder
    trainable = [(n, p) for n, p in unmark_encoder.named_parameters() if p.requires_grad]
    encoder_names = {n for n, _ in encoder.named_parameters()}
    leaked = [n for n, _ in trainable if n.startswith("encoder.") or n in encoder_names]
    if leaked:
        raise TrainerContractViolation(
            f"{len(leaked)} encoder parameter(s) require grad, e.g. {leaked[:5]}. The "
            "encoder is fully frozen (proposal §5.1)."
        )
    if encoder.training:
        raise TrainerContractViolation("the frozen encoder must stay in eval mode")
    total = sum(p.numel() for _, p in trainable)
    hidden = int(getattr(encoder.config, "hidden_size", HIDDEN_SIZE))
    if hidden == HIDDEN_SIZE and total != ADAPTER_TRAINABLE_PARAMETERS:
        raise TrainerContractViolation(
            f"adapter has {total} trainable parameters; the locked architecture has "
            f"{ADAPTER_TRAINABLE_PARAMETERS} at d={HIDDEN_SIZE}. Capacity is not changed "
            "here -- a larger adapter is a later ablation."
        )
    return {
        "trainable_parameters": total,
        "trainable_tensors": len(trainable),
        "encoder_trainable_parameters": 0,
        "encoder_training_mode": False,
        "hidden_size": hidden,
        "precision": PRECISION,
    }


# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------
@dataclass
class MonitorWindow:
    """Aggregate evidence over a window, not per-batch verdicts.

    A single batch may legitimately contain no letter-degraded syllable -- with
    `pi_strip = 0.25` that is ordinary sampling, not a defect. Failing a batch
    for it would be a false alarm, so the channel-liveness evidence is
    accumulated and reported over the window instead.
    """

    batches: int = 0
    tone_channel_differs: int = 0
    letter_channel_differs: int = 0
    tone_embedding_grad_batches: int = 0
    letter_embedding_grad_batches: int = 0
    base_invariance_violations: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "batches": self.batches,
            "tone_channel_differs": self.tone_channel_differs,
            "letter_channel_differs": self.letter_channel_differs,
            "tone_embedding_grad_batches": self.tone_embedding_grad_batches,
            "letter_embedding_grad_batches": self.letter_embedding_grad_batches,
            "base_invariance_violations": self.base_invariance_violations,
            "letter_channel_ever_degraded": self.letter_channel_differs > 0,
            "tone_channel_ever_degraded": self.tone_channel_differs > 0,
        }


def gradient_report(unmark_encoder: Any) -> dict[str, Any]:
    """Per-group gradient norms, plus the two hard invariants. **Lazy torch.**"""
    import torch

    groups: dict[str, float] = {}
    for name, parameter in unmark_encoder.named_parameters():
        if parameter.grad is None:
            continue
        norm = float(parameter.grad.detach().norm())
        if not torch.isfinite(torch.tensor(norm)):
            raise TrainerContractViolation(
                f"non-finite gradient in {name!r}. Stage-1 stops rather than stepping "
                "an optimizer on a poisoned gradient."
            )
        groups[name] = norm
    encoder_names = {n for n, _ in unmark_encoder.encoder.named_parameters()}
    encoder_grads = {n: v for n, v in groups.items() if n in encoder_names or n.startswith("encoder.")}
    if encoder_grads:
        raise TrainerContractViolation(
            f"the frozen encoder received gradients: {sorted(encoder_grads)[:5]}"
        )
    return {"adapter_group_grad_norms": groups, "encoder_grad_tensors": 0}


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def checkpoint_payload(
    *,
    provenance: RunProvenance,
    adapter_state: Any,
    optimizer_state: Any,
    global_update: int,
    sampler_state: dict[str, Any],
    cap: int,
    budget_limited: bool,
    points: Sequence[Any] = (),
    execution: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Everything required to resume **exactly**. No raw corpus text.

    `points` is the validation history. `train_run` has always *read* it back
    (`resume.get("points", [])`) but this function never wrote it, so a resumed
    run would have silently lost every validation measurement taken before the
    interruption -- and selection consumes exactly those. Found and closed by
    the Audit 030 F3 hardening.
    """
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "provenance": provenance.to_dict(),
        "adapter_state": adapter_state,
        "optimizer_state": optimizer_state,
        "global_update": global_update,
        "sampler_state": sampler_state,
        "cap": cap,
        "budget_limited": budget_limited,
        "points": [p.to_dict() if hasattr(p, "to_dict") else dict(p) for p in points],
        "execution": dict(execution) if execution is not None else None,
    }


REQUIRED_CHECKPOINT_KEYS = (
    "schema_version",
    "provenance",
    "adapter_state",
    "optimizer_state",
    "global_update",
    "sampler_state",
    "cap",
    "points",
    "execution",
)


def require_optimizer_parameter_identity(optimizer: Any, adapter: Any) -> None:
    """Every optimizer parameter **is** a current adapter parameter, and vice versa.

    Object identity, never value equality (Audit 030 §AD.6(C)). This catches the
    classic resume failure where a restored adapter is used for the forward while
    the optimizer still holds the previous nominal run's `Parameter` objects — a
    value comparison cannot see that, because the values may legitimately match.
    """
    grouped = [p for group in optimizer.param_groups for p in group["params"]]
    grouped_ids = [id(p) for p in grouped]
    current = {id(p): name for name, p in adapter.named_parameters()}

    foreign = [i for i in grouped_ids if i not in current]
    if foreign:
        raise TrainerContractViolation(
            f"{len(foreign)} optimizer parameter(s) are not parameters of the current "
            "adapter: the optimizer is bound to stale, foreign or frozen-encoder "
            "tensors and would update something the forward pass never reads."
        )
    duplicates = len(grouped_ids) - len(set(grouped_ids))
    if duplicates:
        raise TrainerContractViolation(
            f"{duplicates} adapter parameter(s) appear more than once in the optimizer "
            "parameter groups; they would receive multiple updates per step."
        )
    missing = sorted(name for i, name in current.items() if i not in set(grouped_ids))
    if missing:
        raise TrainerContractViolation(
            f"adapter parameter(s) {missing} are absent from the optimizer; they would "
            "never be trained while still contributing to the loss."
        )


def require_optimizer_state_device(optimizer: Any, device: Any) -> None:
    """Every tensor in optimizer state sits on `device`. Asserted, not assumed.

    `Optimizer.load_state_dict` casts state to each parameter's device, so this is
    a **postcondition** of supported PyTorch behaviour rather than a migration
    this repository implements (Audit 030 §AC.6). Traversed recursively, because
    Adam's state nests and a stray CPU `exp_avg` beside a CUDA parameter is
    exactly the cross-device step this guards.
    """
    import torch

    def walk(value: Any, path: str) -> None:
        if torch.is_tensor(value):
            # Adam's `step` may legitimately be a CPU scalar; only real state moves.
            if value.dim() == 0 and path.endswith("step"):
                return
            if value.device != device:
                raise TrainerContractViolation(
                    f"optimizer state {path} is on {value.device}, not the training "
                    f"device {device}. A cross-device optimizer step would follow."
                )
        elif isinstance(value, dict):
            for key, item in value.items():
                walk(item, f"{path}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                walk(item, f"{path}[{index}]")

    for index, state in optimizer.state_dict().get("state", {}).items():
        walk(state, f"state[{index}]")


def verify_checkpoint(payload: dict[str, Any], provenance: RunProvenance) -> None:
    """Fail closed unless the checkpoint can reproduce this exact run."""
    missing = [k for k in REQUIRED_CHECKPOINT_KEYS if k not in payload]
    if missing:
        raise TrainerContractViolation(f"checkpoint is missing {missing}; cannot resume exactly")
    if payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise TrainerContractViolation(
            f"checkpoint schema {payload['schema_version']!r} != {CHECKPOINT_SCHEMA_VERSION!r}"
        )
    provenance.require_match(payload["provenance"])


LAST_CHECKPOINT_NAME = "training-checkpoint-last.pt"
"""The most recent checkpoint. What a resume continues from."""

BEST_CHECKPOINT_NAME = "training-checkpoint-best.pt"
"""The checkpoint at the best validation point **under the locked selection
rule** (`selection.select_checkpoint`: lowest score, then lower `d_clean`, then
earliest update). What the run's result refers to."""

TRAINING_CHECKPOINT_NAME = LAST_CHECKPOINT_NAME
"""Backwards-compatible alias for the resume target."""

CHECKPOINT_EVERY_UPDATES = EVAL_EVERY_UPDATES
"""Checkpoint cadence: **one per validation point**.

Not a new choice. D-S1B-004 already locks the eval cadence at 500 updates and
"best + last checkpoint persistence; optimizer and corruption `visit` state
persistence". This constant exists so the two cannot drift apart, and the
persistence below implements that locked contract rather than inventing one --
the F3 defect was that the machinery existed and was never invoked, not that a
cadence was undecided.
"""


def _publish(path: Path, payload: dict[str, Any]) -> Path:
    """Atomic publication. temp -> flush -> fsync -> replace -> dir fsync.

    The same failure-atomic discipline Stage 6 proved on real Drive: a crash
    part-way through a write leaves the **previous** checkpoint intact and
    valid, never a truncated file that resume would try to read.
    """
    import torch

    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)
    final = path
    temp = final.with_name(final.name + ".tmp")
    with open(temp, "wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, final)
    directory_fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    except OSError:  # pragma: no cover - not every filesystem allows it
        pass
    finally:
        os.close(directory_fd)
    return final


def save_training_checkpoint(
    directory: Path, payload: dict[str, Any], *, is_best: bool = False
) -> Path:
    """Persist **last**, and **best** when this is the best point so far.

    D-S1B-004 locks "best + last checkpoint persistence". `last` is what a
    resume continues from; `best` is the checkpoint the run's result refers to,
    chosen by the **already-locked** selection rule rather than by a new one --
    the caller passes `is_best` after consulting `select_checkpoint`, so there
    is exactly one definition of "best" in the repository.
    """
    directory = Path(directory)
    published = _publish(directory / LAST_CHECKPOINT_NAME, payload)
    if is_best:
        _publish(directory / BEST_CHECKPOINT_NAME, payload)
    return published


def load_training_checkpoint(directory: Path, *, best: bool = False) -> dict[str, Any] | None:
    """The `last` checkpoint (default) or the `best` one; `None` if absent."""
    import torch

    name = BEST_CHECKPOINT_NAME if best else LAST_CHECKPOINT_NAME
    path = Path(directory) / name
    if not path.is_file():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------
@dataclass
class RunResult:
    provenance: RunProvenance
    points: list[ValidationPoint] = field(default_factory=list)
    cap: int = INITIAL_MAX_UPDATES
    budget_limited: bool = False
    continued: bool = False

    @property
    def selected(self) -> ValidationPoint:
        return select_checkpoint(self.points)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provenance": self.provenance.to_dict(),
            "cap": self.cap,
            "continued_past_initial_budget": self.continued,
            "budget_limited": self.budget_limited,
            "evaluations": [p.to_dict() for p in self.points],
            "selected": self.selected.to_dict(),
            "optimizer": {
                "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
                "gradient_clipping": GRADIENT_CLIPPING,
                "batch_size": BATCH_SIZE,
                "eval_every_updates": EVAL_EVERY_UPDATES,
            },
            "raw_text_persisted": False,
        }


def resolve_budget(result: RunResult) -> RunResult:
    """Apply the precommitted budget rule to a finished trajectory.

    Pure bookkeeping over already-computed validation points, so the rule is
    testable without training anything.
    """
    decision = budget_decision(result.selected.update, result.cap)
    if decision.continue_run:
        result.cap = EXTENDED_MAX_UPDATES
        result.continued = True
        result.budget_limited = False
    else:
        result.budget_limited = decision.budget_limited
    return result


def train_run(
    *,
    objective: Any,
    provenance: RunProvenance,
    train_chunks: dict[str, str],
    tokenizer: Any,
    corruption_policy: Any,
    truncation: TruncationPolicy,
    evaluate_fn: Callable[[int], ValidationPoint],
    pad_token_id: int,
    classifier: Any = None,
    unk_token_id: int | None = None,
    cap: int = INITIAL_MAX_UPDATES,
    resume: dict[str, Any] | None = None,
    checkpoint_dir: Path | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
    execution: Any = None,
) -> RunResult:
    """One Stage-1 run, to `cap` updates. **Imports torch lazily.**

    Not executed by Audit 029: no real model or corpus is loaded there.

    The loop is deliberately small. Everything scientific -- the schedule, the
    selection rule, the budget rule, the corruption draws, the parameter groups
    -- is decided in the torch-free modules and only *applied* here.
    """
    import torch

    from unmark.stage1.data import (
        Stage1Example,
        batch_to_device,
        collate_stage1_batch,
        module_device,
        prepare_example,
    )
    from unmark.stage1.optim import build_optimizer

    if not train_chunks:
        raise TrainerContractViolation("no training chunks supplied")
    adapter = objective.unmark_encoder.adapter
    if not corruption_policy.is_locked_mixture:
        raise TrainerContractViolation(
            "scientific training requires the locked corruption mixture (D-S1B-003); a "
            "run-global scope is exactly the defect that left STRIP-ALL unsupported"
        )
    contract = verify_model_contract(objective.unmark_encoder)

    optimizer = build_optimizer(
        [(n, p) for n, p in objective.unmark_encoder.named_parameters() if p.requires_grad],
        provenance.learning_rate,
    )
    # Bound to THIS nominal run's adapter, and proven so by object identity
    # (D-S1B-017). Checked here for a fresh run and again after any restore.
    require_optimizer_parameter_identity(optimizer, adapter)
    sampler = DeterministicSampler(tuple(sorted(train_chunks)), seed=provenance.run_seed)
    global_update = 0
    result = RunResult(provenance=provenance, cap=cap)

    if resume is not None:
        verify_checkpoint(resume, provenance)
        if execution is not None and resume.get("execution") is not None:
            execution.require_compatible(resume["execution"])
        # STRICT, into the adapter that the forward pass actually uses. v1 loaded
        # the whole wrapper with strict=False, so a key mismatch restored nothing
        # and training silently continued from fresh weights (Audit 030 §AC.9).
        adapter.load_state_dict(resume["adapter_state"], strict=True)
        optimizer.load_state_dict(resume["optimizer_state"])
        require_optimizer_parameter_identity(optimizer, adapter)
        require_optimizer_state_device(optimizer, module_device(objective))
        sampler = DeterministicSampler.from_state(tuple(sorted(train_chunks)), resume["sampler_state"])
        global_update = int(resume["global_update"])
        result.points = [ValidationPoint(**p) for p in resume.get("points", [])]
        result.continued = cap > INITIAL_MAX_UPDATES

    # Update 0 BEFORE any optimizer step, so the initial clean-path distance and
    # the initial condition distances are measured rather than assumed.
    if not any(p.update == 0 for p in result.points):
        result.points.append(evaluate_fn(0))

    window = MonitorWindow()
    objective.train(True)
    while global_update < cap:
        pairs = sampler.next_batch(BATCH_SIZE)
        prepared = []
        for chunk_id, visit in pairs:
            item = prepare_example(
                Stage1Example(text=train_chunks[chunk_id], sample_id=chunk_id),
                tokenizer,
                corruption_policy=corruption_policy,
                truncation=truncation,
                visit=visit,
                classifier=classifier,
                unk_token_id=unk_token_id,
            )
            if item is None:
                raise TrainerContractViolation(
                    f"chunk {chunk_id!r} overflowed at training time; after correct "
                    "pre-chunking this cannot happen (on_overflow=FAIL is a guard)"
                )
            window.tone_channel_differs += int(item.channels_differ)
            window.letter_channel_differs += int(item.letter_channels_differ)
            prepared.append(item)

        # Same one boundary as `validation.evaluate`: the batch follows the
        # model, derived from the objective's own parameters. A no-op on CPU.
        batch = batch_to_device(
            collate_stage1_batch(prepared, pad_token_id), module_device(objective)
        )
        loss_result = objective(batch)

        optimizer.zero_grad(set_to_none=True)
        loss_result.loss.backward()
        grads = gradient_report(objective.unmark_encoder)
        window.batches += 1
        for name, norm in grads["adapter_group_grad_norms"].items():
            lowered = name.lower()
            if "tone" in lowered and norm > 0:
                window.tone_embedding_grad_batches += 1
            if "letter" in lowered and norm > 0:
                window.letter_embedding_grad_batches += 1
        optimizer.step()
        global_update += 1

        if global_update % EVAL_EVERY_UPDATES == 0 or global_update == cap:
            point = evaluate_fn(global_update)
            result.points.append(point)
            objective.train(True)
            # Checkpoint at the validation boundary (the cadence D-S1B-004 locks):
            # point where sampler, update count and validation history are all
            # mutually consistent, so a resume cannot land mid-pass.
            if checkpoint_dir is not None and (
                global_update % CHECKPOINT_EVERY_UPDATES == 0 or global_update == cap
            ):
                # "Best" is decided by the ALREADY-LOCKED selection rule, never
                # by a second comparison written here (D-S1B-004 persistence,
                # selection rule from `select_checkpoint`).
                from unmark.stage1.selection import select_checkpoint

                is_best = select_checkpoint(result.points).update == global_update
                save_training_checkpoint(
                    checkpoint_dir,
                    checkpoint_payload(
                        provenance=provenance,
                        adapter_state=adapter.state_dict(),
                        optimizer_state=optimizer.state_dict(),
                        global_update=global_update,
                        sampler_state=sampler.state_dict(),
                        cap=cap,
                        budget_limited=result.budget_limited,
                        points=result.points,
                        execution=execution.to_dict() if execution is not None else None,
                    ),
                    is_best=is_best,
                )
            if on_event is not None:
                on_event({
                    "global_update": global_update,
                    "visit": sampler.visit,
                    "position": sampler.position,
                    "loss": float(loss_result.loss.detach()),
                    "loss_align": float(loss_result.loss_align.detach()),
                    "loss_clean": float(loss_result.loss_clean.detach()),
                    "validation": point.to_dict(),
                    "monitor_window": window.to_dict(),
                    "gradients": grads,
                    "model_contract": contract,
                })
            window = MonitorWindow()

    return resolve_budget(result)
