"""Deterministic parallel CPU preparation for Stage-1 training (Audit 030 §AG).

§AF measured the real training path and found it **preparation-bound**: 79 % of
each step is `prepare_example`, and only 12.6 % is GPU work. This module removes
that bottleneck by running the *same* `prepare_example` across a persistent
worker pool -- and by nothing else. It is a **pure engineering change**: every
prepared example is byte-identical to what the serial path produces, which is
what makes it not a scientific decision.

**What the main process keeps, absolutely.** The sampler, `next_batch`, visit
advancement, batch membership, batch order, `global_update`, validation history,
checkpoints, the optimizer, the CUDA model, collation, H2D and
forward/backward/step. Workers receive `(sample_id, visit, text)` triples that
the main process has *already chosen*, and hand back prepared examples. They
never see a sampler, never choose a sample or a visit, never touch CUDA, and
never load model weights -- only the pinned tokenizer.

**Order is reconstructed, never inferred.** `Executor.map` yields results in
*input* order regardless of completion order, so worker scheduling cannot leak
into scientific order.

**No prefetch.** Exactly one sampler batch is consumed per synchronous prepared
batch. Checkpointing commits sampler state together with the completed update,
so reading ahead would create a resume-state problem that this change is not
scoped to solve.

**Spawn, never fork.** See `MULTIPROCESSING_START_METHOD`.
"""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Iterable, Sequence

from unmark.stage1.contracts import (
    CorruptionRatePolicy,
    OverflowBehaviour,
    Stage1ContractViolation,
    TruncationPolicy,
)

PREPARATION_WORKERS = 8
"""Persistent CPU preparation workers.

**Operational, not scientific.** Prepared output is byte-identical across worker
counts -- proven by the equivalence tests -- so this number changes wall-clock
and nothing else. It is therefore recorded as operational provenance and is
deliberately **not** part of `RunProvenance` and **not** resume-blocking: a run
interrupted with 8 workers may legitimately resume with 4 and remain the same
experiment. Chosen from the §AF benchmark on a 24-physical-core host.
"""

MULTIPROCESSING_START_METHOD = "spawn"
"""**Spawn, never fork.**

The §AF benchmark used `fork` because it was a CPU-only process in which CUDA had
never been initialised. Production Stage-1 is the opposite: by the time the
training loop runs, the parent holds a CUDA context, the placed model and the
optimizer. Forking a CUDA-initialised parent copies CUDA state that is not valid
in the child and is documented as unsupported; it can deadlock or corrupt rather
than fail cleanly. `spawn` starts each worker from a fresh interpreter, which is
also why workers must rebuild their tokenizer and inventory from identity rather
than inherit them.
"""

PREPARATION_BACKEND = "multiprocessing_spawn"
ORDER_PRESERVING = True
PREFETCH_ENABLED = False


class PreparationContractViolation(Stage1ContractViolation):
    """Parallel preparation could not honour its contract. Never falls back."""


_WORKER: dict[str, Any] = {}


def _initialise_worker(config: dict[str, Any]) -> None:
    """Build one worker's immutable preparation state. Once per process.

    Fails closed if the pinned tokenizer or the verified inventory cannot be
    established: a worker that silently prepared examples under a different
    tokenizer or a different syllable inventory would corrupt the experiment in a
    way no downstream check would catch.
    """
    from unmark.linguistics import load_inventory, make_classifier

    # A picklable factory, exactly as Stage-6's `parallel.py` does it. Production
    # passes `pinned_tokenizer`, which refuses any revision but the locked one;
    # tests inject a tiny stand-in so the pool itself is testable without
    # downloading PhoBERT. There is no way to reach production with a stub,
    # because `execute_stage` names the pinned factory directly.
    tokenizer = config["tokenizer_factory"](
        config["encoder_checkpoint"], config["encoder_revision"]
    )
    # The exact pinned inventory, re-verified in this process against the
    # committed manifest. `load_inventory` (not `try_load_inventory`) so a
    # missing or altered cache raises here rather than degrading eligibility.
    inventory = load_inventory()

    _WORKER.clear()
    _WORKER.update(
        tokenizer=tokenizer,
        classifier=make_classifier(inventory),
        corruption_policy=CorruptionRatePolicy(
            seed=config["corruption_seed"], pi_strip=config["pi_strip"]
        ),
        truncation=TruncationPolicy(
            max_length=config["max_length"],
            on_overflow=OverflowBehaviour[config["on_overflow"]],
        ),
        unk_token_id=config["unk_token_id"],
    )


def _prepare_one(task: tuple[str, int, str]):
    """Prepare exactly one example. The authoritative helper, unmodified.

    Scientific preparation logic is **not** reimplemented here: this calls
    `prepare_example`, the same function the serial path calls.
    """
    from unmark.stage1.data import Stage1Example, prepare_example

    sample_id, visit, text = task
    if not _WORKER:  # pragma: no cover - only reachable if the initialiser failed
        raise PreparationContractViolation("preparation worker was never initialised")
    return prepare_example(
        Stage1Example(text=text, sample_id=sample_id),
        _WORKER["tokenizer"],
        corruption_policy=_WORKER["corruption_policy"],
        truncation=_WORKER["truncation"],
        visit=visit,
        classifier=_WORKER["classifier"],
        unk_token_id=_WORKER["unk_token_id"],
    )


def pinned_tokenizer(checkpoint: str, revision: str):
    """The pinned slow tokenizer. Fails closed on any other revision.

    Module-level and picklable by reference, which is what `spawn` requires.
    """
    from unmark.stage1.protocol import ENCODER_REVISION

    # Checked BEFORE transformers is imported or any network identity is touched:
    # a worker must fail on a foreign revision, not on the way to fetching one.
    if revision != ENCODER_REVISION:
        raise PreparationContractViolation(
            f"preparation worker was given revision {revision!r}, not the locked "
            f"{ENCODER_REVISION!r}; a worker must never advance the backbone identity"
        )
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(checkpoint, revision=revision, use_fast=False)


def worker_config(
    *,
    encoder_checkpoint: str,
    encoder_revision: str,
    corruption_policy: CorruptionRatePolicy,
    truncation: TruncationPolicy,
    unk_token_id: int | None,
    tokenizer_factory: Any = pinned_tokenizer,
) -> dict[str, Any]:
    """The immutable, picklable identity a worker rebuilds itself from.

    Deliberately small. Under `spawn` every argument is pickled to each worker,
    so the 2.6M-entry corpus dictionary must never appear here -- only the text
    of the examples the sampler has already selected travels, one batch at a time.
    """
    return {
        "tokenizer_factory": tokenizer_factory,
        "encoder_checkpoint": encoder_checkpoint,
        "encoder_revision": encoder_revision,
        "corruption_seed": corruption_policy.seed,
        "pi_strip": corruption_policy.pi_strip,
        "max_length": truncation.max_length,
        "on_overflow": truncation.on_overflow.name,
        "unk_token_id": unk_token_id,
    }


class PreparationPool:
    """A persistent, order-preserving, deterministic preparation pool.

    Created once for a Stage-1 execution scope, not once per batch: under `spawn`
    each worker reloads the pinned tokenizer and re-verifies the inventory, which
    is far too expensive to repeat 20 000 times.
    """

    def __init__(self, config: dict[str, Any], workers: int = PREPARATION_WORKERS) -> None:
        if workers < 1:
            raise PreparationContractViolation(f"workers must be >= 1, got {workers}")
        self.workers = workers
        self.config = config
        self._pool: ProcessPoolExecutor | None = None

    def __enter__(self) -> PreparationPool:
        context = multiprocessing.get_context(MULTIPROCESSING_START_METHOD)
        self._pool = ProcessPoolExecutor(
            max_workers=self.workers,
            mp_context=context,
            initializer=_initialise_worker,
            initargs=(self.config,),
        )
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Shut down on normal completion, on exception and on fail-closed abort."""
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=True)

    def prepare(self, tasks: Sequence[tuple[str, int, str]]) -> list[Any]:
        """Prepare the given `(sample_id, visit, text)` triples, **in input order**.

        `Executor.map` yields results in the order the tasks were submitted,
        whatever order the workers finish in, so completion order cannot reach
        scientific order.

        A worker exception propagates and aborts: `map` re-raises on iteration, so
        a partial batch can never reach training. There is **no serial fallback** —
        silently degrading would hide a broken pool behind a slow run.
        """
        if self._pool is None:
            raise PreparationContractViolation(
                "preparation pool is not running; use it as a context manager"
            )
        try:
            prepared = list(self._pool.map(_prepare_one, tasks))
        except Exception as error:  # noqa: BLE001 - re-raised, never swallowed
            raise PreparationContractViolation(
                f"parallel preparation failed and will not fall back to serial: {error}"
            ) from error
        if len(prepared) != len(tasks):  # pragma: no cover - map is length-preserving
            raise PreparationContractViolation(
                f"pool returned {len(prepared)} results for {len(tasks)} tasks"
            )
        return prepared


def prepare_serially(
    tasks: Iterable[tuple[str, int, str]],
    tokenizer: Any,
    *,
    corruption_policy: CorruptionRatePolicy,
    truncation: TruncationPolicy,
    classifier: Any = None,
    unk_token_id: int | None = None,
) -> list[Any]:
    """The serial reference implementation. **Tests and diagnostics only.**

    Exists so equivalence can be asserted against something, and is deliberately
    *not* reachable as a fallback from the scientific training path.
    """
    from unmark.stage1.data import Stage1Example, prepare_example

    return [
        prepare_example(
            Stage1Example(text=text, sample_id=sample_id),
            tokenizer,
            corruption_policy=corruption_policy,
            truncation=truncation,
            visit=visit,
            classifier=classifier,
            unk_token_id=unk_token_id,
        )
        for sample_id, visit, text in tasks
    ]


def preparation_provenance(workers: int = PREPARATION_WORKERS) -> dict[str, Any]:
    """Operational provenance for the run report. **Not scientific identity.**"""
    return {
        "preparation_backend": PREPARATION_BACKEND,
        "preparation_workers": workers,
        "order_preserving": ORDER_PRESERVING,
        "prefetch": PREFETCH_ENABLED,
    }
