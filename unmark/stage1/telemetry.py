"""Structured Stage-1 telemetry. **Operational only — no scientific effect.**

Motivation (Audit 040): the first authorised `lr-pilot` attempt printed
``frozen backbone VERIFIED on cuda`` and then went quiet for minutes while it
read a 2.2 GB corpus, built the held-out condition batches and spawned eight
preparation workers. External monitoring could see CPU and GPU state but could
not tell *which* phase was running, which candidate, or how far along. That is
an observability defect, not a scientific one, and it is fixed here without
touching a single scientific decision.

**This module has no dependencies.** No wandb, no network, no psutil, no rich,
no tqdm — nothing beyond the standard library. The scientific process must never
be able to fail because a monitoring package is missing, so the monitoring layer
lives outside it (`scripts/stage1_wandb_monitor.py`) and *reads* what this
module writes.

Contract, enforced by tests:

* emitting consumes **zero** RNG and performs no forward, backward, optimizer
  step, sampling or corruption draw;
* emitting never mutates training state — values are read, serialised, dropped;
* emitting can never raise into the scientific path. A closed pipe or an
  unserialisable value degrades to silence, never to a crashed run;
* the default sink is `NullSink`, so a caller that does not opt in gets
  byte-identical behaviour to the pre-telemetry code.

Output is one line per event on stdout::

    UNMARK_TELEMETRY {"schema": "stage1-telemetry-v1", "event": "...", ...}

The prefix exists so a parser can pick these out of the runner's ordinary human
prose without the two ever being confused.
"""

from __future__ import annotations

import json
import os
import sys
import time
from contextlib import contextmanager
from typing import Any, Iterator, Mapping

SCHEMA = "stage1-telemetry-v1"
"""Versioned. A consumer that does not recognise the schema must refuse rather
than guess: the field set is a contract, not a convenience."""

PREFIX = "UNMARK_TELEMETRY "
"""Line marker. Chosen to be unmistakable in mixed console output."""

ENV_FLAG = "UNMARK_TELEMETRY"
"""``1`` enables emission. Absent or ``0`` yields a `NullSink`."""

PROGRESS_EVERY_UPDATES = 50
"""**OPERATIONAL cadence — deliberately NOT a protocol constant.**

It lives here, not in `protocol.py`, because changing it changes nothing
scientific: it does not touch the locked `EVAL_EVERY_UPDATES = 500` evaluation
cadence, the identical checkpoint cadence, or any budget. It only decides how
often a progress line is printed.

50 gives roughly one line every few seconds at Stage-1 speeds — frequent enough
that a stall is obvious, rare enough that the console stays readable.
"""

MAX_STRING = 200
"""Hard cap on any emitted string.

A leak guard, not a formatting nicety. Nothing in this module is ever *meant* to
carry corpus text, but a future caller could pass a chunk by mistake; truncating
at 200 characters means such a bug is visibly broken rather than silently
exfiltrating training data into a dashboard. Digests, paths and labels all fit
comfortably.
"""

_TRUNCATED = "...[truncated]"


def _safe(value: Any) -> Any:
    """Coerce to something JSON-serialisable, bounded and leak-resistant."""
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        # NaN/Inf are not valid JSON; emit them as strings so a stalled or
        # diverged run is still *visible* rather than crashing the emitter.
        if value != value or value in (float("inf"), float("-inf")):
            return repr(value)
        return value
    if isinstance(value, str):
        return value if len(value) <= MAX_STRING else value[:MAX_STRING] + _TRUNCATED
    if isinstance(value, Mapping):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    return _safe(str(value))


class TelemetrySink:
    """Base sink. The default implementation does nothing at all."""

    enabled = False

    def emit(self, event: str, **fields: Any) -> None:
        """Record one event. Must never raise."""

    def progress_every(self) -> int:
        """Update cadence for `train_progress`; 0 disables it."""
        return 0


class NullSink(TelemetrySink):
    """The default. Zero cost, zero output, zero behavioural difference."""


class JsonlSink(TelemetrySink):
    """Writes one JSON line per event to a stream (stdout by default)."""

    enabled = True

    def __init__(
        self,
        stream: Any = None,
        *,
        clock: Any = time.time,
        monotonic: Any = time.monotonic,
        progress_every_updates: int = PROGRESS_EVERY_UPDATES,
    ) -> None:
        self._stream = stream if stream is not None else sys.stdout
        self._clock = clock
        self._monotonic = monotonic
        self._progress_every = int(progress_every_updates)
        self._started = monotonic()
        self._sequence = 0

    def progress_every(self) -> int:
        return self._progress_every

    def emit(self, event: str, **fields: Any) -> None:
        # Fail silent, always. Telemetry is observability; a monitoring problem
        # must never become a scientific failure.
        try:
            self._sequence += 1
            payload = {
                "schema": SCHEMA,
                "event": str(event),
                "seq": self._sequence,
                "wall_clock": self._clock(),
                "elapsed_s": round(self._monotonic() - self._started, 6),
            }
            for key, value in fields.items():
                payload[str(key)] = _safe(value)
            line = PREFIX + json.dumps(payload, sort_keys=True, default=str)
            self._stream.write(line + "\n")
            self._stream.flush()
        except Exception:  # noqa: BLE001 - observability must not break training
            pass


def sink_from_environment(env: Mapping[str, str] | None = None, stream: Any = None) -> TelemetrySink:
    """`JsonlSink` when `UNMARK_TELEMETRY=1`, otherwise `NullSink`.

    Opt-in by default so that every existing caller — and every existing test —
    keeps exactly the behaviour it had before this module existed.
    """
    environment = os.environ if env is None else env
    if str(environment.get(ENV_FLAG, "")).strip() in ("1", "true", "TRUE", "yes", "on"):
        return JsonlSink(stream)
    return NullSink()


@contextmanager
def phase(sink: TelemetrySink, name: str, **fields: Any) -> Iterator[None]:
    """Bracket a real long-running operation with START / DONE events.

    Elapsed time is **operational metadata only**: it is not recorded in any
    artifact, not compared against anything, and not used by any decision.

    A phase that fails emits `stage_phase` with `state="FAILED"` and the
    exception type — then re-raises unchanged, so fail-closed behaviour is
    exactly what it was.
    """
    if not sink.enabled:
        yield
        return
    started = time.monotonic()
    sink.emit("stage_phase", phase=name, state="START", **fields)
    try:
        yield
    except BaseException as error:  # noqa: BLE001 - observed, then re-raised
        sink.emit(
            "stage_phase", phase=name, state="FAILED",
            elapsed_phase_s=round(time.monotonic() - started, 6),
            error_type=type(error).__name__, **fields,
        )
        raise
    sink.emit(
        "stage_phase", phase=name, state="DONE",
        elapsed_phase_s=round(time.monotonic() - started, 6), **fields,
    )
