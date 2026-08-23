"""Ordered, document-level parallel chunk computation. **Operational only.**

Stage 6 is CPU tokenizer and Python work, and the real runtime exposes 48
cores while the committed runner uses one. This module lets the *compute* fan
out without letting anything else move:

* **workers own exactly one thing** -- `chunk_document` for a single document.
  They never serialise a payload, never touch the checkpoint, never write to
  the destination, and never see the document order;
* **the main process owns everything else** -- global order, JSON
  serialisation, membership accumulation, checkpoint state and durable
  persistence;
* **the collector re-imposes order.** Results may complete out of order; they
  are emitted strictly by original document index, and chunks within a document
  keep their `chunk_index` order because `chunk_document` produced them that
  way. Only a contiguous emitted prefix can ever become checkpoint state, which
  is the invariant the durability guarantee already rested on;
* **in-flight work is bounded**, so RAM cannot grow with corpus size.

**The worker count is not science.** `chunk_document` is a pure function of the
document, the two length functions and `max_length`; the caches inside the
length functions are memoisation and change no value. So the emitted chunks --
ids, ranges, text, both lengths, order -- are identical for any worker count,
and the tests assert exactly that for 1, 2, 4 and 8 workers.

A worker failure is **fatal and provenanced**: it is re-raised in the main
process naming the document, and because emission is strictly ordered the
checkpoint cannot have advanced past it.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Iterator, Sequence

from unmark.stage1.chunking import chunk_document
from unmark.stage1.contracts import Stage1ContractViolation
from unmark.stage1.protocol import MAX_LENGTH

DEFAULT_MAX_IN_FLIGHT_PER_WORKER = 4
"""How many documents may be outstanding per worker.

Bounds collector memory at `workers * this` documents' chunks. Operational: it
changes throughput and peak RSS, never output.
"""


# ---------------------------------------------------------------------------
# Worker side -- one tokenizer per process, built once
# ---------------------------------------------------------------------------
_WORKER: dict[str, Any] = {}


def _initialise_worker(factory: Callable[[], Any], max_length: int) -> None:
    """Build this worker's own tokenizer, length functions and classifier.

    Called once per process. Each worker therefore has **its own** tokenizer and
    its own memo tables -- no mutable tokenizer state is shared across
    processes, which is the only safe assumption for the pinned slow Python
    tokenizer. **Correctness never depends on a warm cache**: the caches are
    memoisation of pure functions, so a cold worker computes the same values,
    only slower.
    """
    from unmark.linguistics import make_classifier, try_load_inventory
    from unmark.stage1.lengths import build_length_functions

    tokenizer = factory()
    reference_length, base_length, transforms = build_length_functions(tokenizer)
    _WORKER.update(
        reference_length=reference_length,
        base_length=base_length,
        transforms=transforms,
        max_length=max_length,
        # Built here, from the same repository inventory the serial path uses,
        # rather than pickled across the boundary, so the worker and serial
        # paths are constructed identically.
        #
        # It does NOT affect chunk boundaries. `safe_cut_offsets` accepts a
        # classifier for signature compatibility and ignores it: where a cut may
        # land is an orthographic question (unit boundaries and maximal letter
        # runs), not a lexical one -- D-S1B-013's lexicon-free rule. Verified
        # structurally and empirically in
        # `tests/test_stage1_inventory_preflight.py`. This comment previously
        # claimed the opposite, which would have implied the prepared corpus
        # depends on the syllable inventory; it does not (Audit 030 §W.7).
        classifier=make_classifier(try_load_inventory()),
    )


def _chunk_one(payload: tuple[int, Any, str]) -> tuple[int, list[Any]]:
    """Chunk one document. Returns `(index, chunks)`; raises with provenance."""
    index, document, partition = payload
    try:
        chunks = chunk_document(
            document,
            partition,
            reference_length=_WORKER["reference_length"],
            base_length=_WORKER["base_length"],
            max_length=_WORKER["max_length"],
            classifier=_WORKER["classifier"],
        )
    except Stage1ContractViolation:
        raise
    except Exception as error:  # noqa: BLE001 - re-raised with provenance below
        raise Stage1ContractViolation(
            f"worker failed on document index {index} "
            f"({getattr(document, 'document_id', '?')!r}): {error!r}"
        ) from error
    return index, chunks


# ---------------------------------------------------------------------------
# Main-process side -- the ordered, bounded collector
# ---------------------------------------------------------------------------
def resolve_worker_count(requested: int | None) -> int:
    """`None` means one worker. Never guesses a number from the machine.

    A worker count that changed with the host would make the operational
    configuration of a run unreproducible from its command line, so the default
    is deliberately conservative and explicit.
    """
    if requested is None:
        return 1
    workers = int(requested)
    if workers < 1:
        raise Stage1ContractViolation(
            f"--prepare-workers must be at least 1, got {workers}"
        )
    return min(workers, max(1, (os.cpu_count() or 1)))


def ordered_document_chunks(
    documents: Sequence[Any],
    partition_of: dict[str, str],
    *,
    start_index: int,
    tokenizer_factory: Callable[[], Any],
    workers: int,
    max_length: int = MAX_LENGTH,
    max_in_flight: int | None = None,
    on_wait: Callable[[float], None] | None = None,
    serial_length_functions: tuple[Callable[[str], int], Callable[[str], int]] | None = None,
    classifier: Callable[[str], Any] | None = None,
) -> Iterator[tuple[int, Any, list[Any]]]:
    """Yield `(index, document, chunks)` in **strict original index order**.

    With `workers == 1` this is an ordinary serial loop in this process -- no
    pool is created, so the single-worker path pays nothing for the existence of
    the parallel one.
    """
    total = len(documents)
    if workers <= 1:
        if serial_length_functions is not None:
            # Reuse the caller's already-built functions so its counters keep
            # reporting the run that actually happened.
            reference_length, base_length = serial_length_functions
        else:
            from unmark.stage1.lengths import build_length_functions

            reference_length, base_length, _ = build_length_functions(tokenizer_factory())
        for index in range(start_index, total):
            document = documents[index]
            yield index, document, chunk_document(
                document,
                partition_of[document.document_id],
                reference_length=reference_length,
                base_length=base_length,
                max_length=max_length,
                classifier=classifier,
            )
        return

    import time

    in_flight = max_in_flight or workers * DEFAULT_MAX_IN_FLIGHT_PER_WORKER
    in_flight = max(1, in_flight)
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialise_worker,
        initargs=(tokenizer_factory, max_length),
    ) as pool:
        pending: dict[int, Any] = {}
        submit_index = start_index
        for emit_index in range(start_index, total):
            # Keep the pool fed, but never more than `in_flight` documents'
            # results can exist at once -- that is the memory bound.
            while submit_index < total and len(pending) < in_flight:
                document = documents[submit_index]
                pending[submit_index] = pool.submit(
                    _chunk_one,
                    (submit_index, document, partition_of[document.document_id]),
                )
                submit_index += 1
            future = pending.pop(emit_index)
            waited = time.monotonic()
            # `.result()` re-raises the worker's exception here, in the main
            # process, before this document is emitted -- so an ordered prefix
            # can never contain a document whose worker failed.
            index, chunks = future.result()
            if on_wait is not None:
                on_wait(time.monotonic() - waited)
            if index != emit_index:
                raise Stage1ContractViolation(
                    f"collector received document index {index} while emitting "
                    f"{emit_index}; ordered emission is not optional"
                )
            yield emit_index, documents[emit_index], chunks
