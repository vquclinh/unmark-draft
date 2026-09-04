#!/usr/bin/env python3
"""EXTERNAL operational monitor for Stage-1. **Never imported by the science.**

Audit 040. This process launches `scripts/stage1_runner.py` as a subprocess
using the *accepted scientific Python executable*, reads its structured
`UNMARK_TELEMETRY` stdout stream, renders a live Colab console, and mirrors
scalars into Weights & Biases.

The isolation is the point:

    monitoring venv  (wandb, psutil)
      -> this script
         -> accepted scientific Python  (torch, transformers -- NO wandb)
            -> scripts/stage1_runner.py

The scientific process never imports wandb, never opens a socket, and cannot
fail because a dashboard is unavailable. If W&B breaks, training continues and
telemetry is still written to a local JSONL file for later sync. **Nothing here
can alter scientific behaviour**: this process only reads stdout.

Terminology, deliberately precise. Stage-1 is **update-based, not epoch-based**,
so this monitor never prints "epoch". It derives and clearly labels:

    sample_visits_total    = global_update * BATCH_SIZE
    corpus_pass_equivalent = sample_visits_total / train_chunks

These are **sample-visit / corpus-pass equivalents** over train *chunks*. They
are emphatically NOT "unique sentences seen": the sampler revisits chunks, so a
pass-equivalent of 1.0 means "as many chunk-visits as there are train chunks",
not "every chunk seen once". The scientific axis remains `global_update`.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]

TELEMETRY_PREFIX = "UNMARK_TELEMETRY "
SUPPORTED_SCHEMA = "stage1-telemetry-v1"

VERIFIED_TRAIN_CHUNKS = 2_621_624
"""The verified Stage-6 train chunk count. Used ONLY as a fallback denominator
for the pass-equivalent display when telemetry has not yet reported the real
count; `corpus_loaded` / `run_start` carry the live value and take precedence."""

HEARTBEAT_SECONDS = 15.0
STALL_WARN_SECONDS = 300.0


# ---------------------------------------------------------------------------
# Derived operational metrics -- pure, no dependencies, fully testable
# ---------------------------------------------------------------------------
def derive_pass_metrics(global_update: int, batch_size: int, train_chunks: int) -> dict[str, Any]:
    """Sample-visit and corpus-pass EQUIVALENTS. Derived, never exact-sentence.

    `pass_index` is 1-based for display: during the first pass it reads 1.
    """
    visits = int(global_update) * int(batch_size)
    if train_chunks <= 0:
        return {
            "sample_visits_total": visits,
            "corpus_pass_equivalent": None,
            "pass_index": None,
            "sample_visits_in_current_pass": None,
            "pass_fraction": None,
            "pass_percent": None,
        }
    equivalent = visits / train_chunks
    in_pass = visits % train_chunks
    return {
        "sample_visits_total": visits,
        "corpus_pass_equivalent": equivalent,
        "pass_index": int(equivalent) + 1,
        "sample_visits_in_current_pass": in_pass,
        "pass_fraction": in_pass / train_chunks,
        "pass_percent": 100.0 * in_pass / train_chunks,
    }


def parse_line(line: str) -> dict[str, Any] | None:
    """A telemetry event, or `None` for ordinary human output."""
    if not line.startswith(TELEMETRY_PREFIX):
        return None
    try:
        event = json.loads(line[len(TELEMETRY_PREFIX):])
    except json.JSONDecodeError:
        return None
    if not isinstance(event, dict) or event.get("schema") != SUPPORTED_SCHEMA:
        # Refuse rather than guess: an unrecognised schema is a contract change.
        return None
    return event


def candidate_key(event: Mapping[str, Any]) -> str | None:
    """Stable identity for one scientific candidate, from production values."""
    stage, label = event.get("stage"), event.get("label")
    if stage is None or label is None:
        return None
    return f"{stage}/{label}/seed-{event.get('seed')}"


def candidate_run_name(event: Mapping[str, Any]) -> str:
    """Deterministic, useful W&B run name derived from telemetry, not hard-coded."""
    stage = str(event.get("stage", "stage")).replace("_", "-")
    label = str(event.get("label", "run")).replace("=", "-").replace(" ", "")
    return f"{stage}-{label}-seed-{event.get('seed')}"


# ---------------------------------------------------------------------------
# Monitor state -- consumes events, exposes everything the console/W&B need
# ---------------------------------------------------------------------------
class MonitorState:
    """Pure event consumer. No wandb, no I/O, no network — so it is testable."""

    def __init__(self, train_chunks: int = VERIFIED_TRAIN_CHUNKS) -> None:
        self.train_chunks = train_chunks
        self.batch_size = 128
        self.phase: str | None = None
        self.phase_started: float | None = None
        self.candidate: dict[str, Any] | None = None
        self.candidate_key: str | None = None
        self.global_update = 0
        self.cap = 0
        self.last_progress: dict[str, Any] | None = None
        self.last_validation: dict[str, Any] | None = None
        self.latest_checkpoint_update: int | None = None
        self.checkpoints = 0
        self.selection: dict[str, Any] | None = None
        self.stage_complete: dict[str, Any] | None = None
        self.events = 0
        self.validations: list[dict[str, Any]] = []
        self.train_losses: list[tuple[int, float]] = []
        self.campaign: dict[str, Any] = {}
        """Authoritative campaign provenance, straight from the production
        `CampaignIdentity` plus the verified corpus pin. Never re-derived here."""

    # -- ingestion ---------------------------------------------------------
    def consume(self, event: Mapping[str, Any]) -> str:
        """Apply one event; returns its kind."""
        self.events += 1
        kind = str(event.get("event", ""))
        handler = getattr(self, f"_on_{kind}", None)
        if handler is not None:
            handler(event)
        return kind

    def _on_stage_phase(self, event: Mapping[str, Any]) -> None:
        if event.get("state") == "START":
            self.phase = str(event.get("phase"))
            self.phase_started = time.monotonic()
        elif event.get("state") in ("DONE", "FAILED"):
            self.phase = None
            self.phase_started = None

    def _on_campaign_identity(self, event: Mapping[str, Any]) -> None:
        self.campaign = {k: v for k, v in event.items()
                         if k in CAMPAIGN_PROVENANCE_KEYS}

    def _on_corpus_loaded(self, event: Mapping[str, Any]) -> None:
        chunks = event.get("train_chunks")
        if isinstance(chunks, int) and chunks > 0:
            self.train_chunks = chunks

    def _on_run_start(self, event: Mapping[str, Any]) -> None:
        self.candidate = dict(event)
        self.candidate_key = candidate_key(event)
        self.global_update = int(event.get("initial_global_update") or 0)
        self.cap = int(event.get("cap") or 0)
        if isinstance(event.get("batch_size"), int):
            self.batch_size = int(event["batch_size"])
        if isinstance(event.get("train_chunks"), int) and event["train_chunks"] > 0:
            self.train_chunks = int(event["train_chunks"])
        self.last_validation = None
        self.validations = []
        self.train_losses = []
        self.checkpoints = 0
        self.latest_checkpoint_update = None

    def _on_train_progress(self, event: Mapping[str, Any]) -> None:
        self.global_update = int(event.get("global_update") or 0)
        self.cap = int(event.get("cap") or self.cap)
        self.last_progress = dict(event)
        loss = event.get("loss")
        if isinstance(loss, (int, float)):
            self.train_losses.append((self.global_update, float(loss)))

    def _on_validation(self, event: Mapping[str, Any]) -> None:
        self.last_validation = dict(event)
        self.validations.append(dict(event))

    def _on_checkpoint(self, event: Mapping[str, Any]) -> None:
        self.checkpoints += 1
        update = event.get("update")
        if isinstance(update, int):
            self.latest_checkpoint_update = update

    def _on_selection(self, event: Mapping[str, Any]) -> None:
        self.selection = dict(event)

    def _on_stage_complete(self, event: Mapping[str, Any]) -> None:
        self.stage_complete = dict(event)

    # -- derived views -----------------------------------------------------
    def pass_metrics(self) -> dict[str, Any]:
        return derive_pass_metrics(self.global_update, self.batch_size, self.train_chunks)

    def run_fraction(self) -> float | None:
        return self.global_update / self.cap if self.cap else None

    def next_eval_update(self, every: int = 500) -> int | None:
        if not self.cap:
            return None
        nxt = ((self.global_update // every) + 1) * every
        return min(nxt, self.cap)

    def diagnostics(self) -> dict[str, Any]:
        """OBSERVATIONAL ONLY. Never feeds back into training.

        Deliberately neutral names. `select_checkpoint` minimises the score
        (`min(points, key=(score, d_clean, update))`), so *lower is better* —
        but this monitor still refuses to publish an "overfit" verdict, because
        no authoritative overfitting definition exists in the protocol. Trends
        are reported; the human reads the graph.
        """
        out: dict[str, Any] = {}
        if len(self.train_losses) >= 2:
            window = self.train_losses[-10:]
            out["train_trend"] = window[-1][1] - window[0][1]
        if len(self.validations) >= 2:
            prev, last = self.validations[-2], self.validations[-1]
            if isinstance(prev.get("score"), (int, float)) and isinstance(last.get("score"), (int, float)):
                # Negative = score fell = improving under the locked rule.
                out["validation_trend"] = float(last["score"]) - float(prev["score"])
        if "train_trend" in out and "validation_trend" in out:
            out["divergence_watch"] = out["validation_trend"] - out["train_trend"]
        return out


# ---------------------------------------------------------------------------
# System sampling -- optional, degrades to "unavailable"
# ---------------------------------------------------------------------------
def sample_gpu() -> dict[str, Any]:
    """One `nvidia-smi` sample. Returns `{}` when unavailable."""
    binary = shutil.which("nvidia-smi")
    if binary is None:
        return {}
    try:
        out = subprocess.run(
            [binary, "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        if out.returncode != 0 or not out.stdout.strip():
            return {}
        util, used, total, power, temp = [x.strip() for x in out.stdout.strip().splitlines()[0].split(",")]
        return {"gpu_util_percent": float(util), "vram_used_mib": float(used),
                "vram_total_mib": float(total), "gpu_power_w": float(power),
                "gpu_temp_c": float(temp)}
    except Exception:  # noqa: BLE001 - monitoring only
        return {}


def sample_process(pid: int) -> dict[str, Any]:
    """Process CPU/RSS/threads/IO. Returns `{}` without psutil."""
    try:
        import psutil  # noqa: PLC0415 - optional monitoring dependency
    except ImportError:
        return {}
    try:
        proc = psutil.Process(pid)
        with proc.oneshot():
            info: dict[str, Any] = {
                "proc_cpu_percent": proc.cpu_percent(interval=None),
                "proc_rss_mib": proc.memory_info().rss / (1024 * 1024),
                "proc_threads": proc.num_threads(),
            }
            try:
                io = proc.io_counters()
                info["proc_read_mib"] = io.read_bytes / (1024 * 1024)
                info["proc_write_mib"] = io.write_bytes / (1024 * 1024)
            except Exception:  # noqa: BLE001 - not available on every platform
                pass
        return info
    except Exception:  # noqa: BLE001 - monitoring only
        return {}


# ---------------------------------------------------------------------------
# W&B bridge -- lazily imported, entirely optional
# ---------------------------------------------------------------------------
class WandbBridge:
    """One W&B run per scientific candidate. Degrades to a no-op on any failure.

    Only scalars, config and safe identifiers are uploaded. **Never** raw corpus
    text, chunks, prepared data, official TEST, checkpoints, weights or
    tokenizer artifacts.
    """

    def __init__(self, project: str, group: str | None, state_dir: Path,
                 enabled: bool = True, mode: str | None = None) -> None:
        self.project, self.group = project, group
        self.state_dir = Path(state_dir)
        self.enabled = enabled
        self.mode = mode
        self._wandb: Any = None
        self._run: Any = None
        self._key: str | None = None
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._ids_path = self.state_dir / "wandb_run_ids.json"

    # -- persisted run identity -------------------------------------------
    def _ids(self) -> dict[str, str]:
        try:
            return json.loads(self._ids_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}

    def _remember(self, key: str, run_id: str) -> None:
        ids = self._ids()
        ids[key] = run_id
        try:
            self._ids_path.write_text(json.dumps(ids, indent=2, sort_keys=True) + "\n",
                                      encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    def _load(self) -> Any:
        if self._wandb is None and self.enabled:
            try:
                import wandb  # noqa: PLC0415 - monitoring-only dependency
                self._wandb = wandb
            except ImportError:
                print("[monitor] wandb not installed; console-only, telemetry still "
                      "written locally for later sync", flush=True)
                self.enabled = False
        return self._wandb

    def start_candidate(self, event: Mapping[str, Any], config: Mapping[str, Any]) -> str | None:
        """Create or RESUME the W&B run for this candidate. Returns its URL."""
        key = candidate_key(event)
        if key is None or not self.enabled:
            return None
        wandb = self._load()
        if wandb is None:
            return None
        self.finish()
        existing = self._ids().get(key)
        try:
            self._run = wandb.init(
                project=self.project, group=self.group, name=candidate_run_name(event),
                id=existing, resume="allow" if existing else None,
                config=dict(config), mode=self.mode,
                # No code, no artifacts, no source scraping.
                settings=wandb.Settings(save_code=False) if hasattr(wandb, "Settings") else None,
            )
            self._key = key
            self._remember(key, self._run.id)
            # global_update is the scientific X axis -- never W&B's internal
            # log-call counter.
            self._run.define_metric("progress/global_update")
            for prefix in ("progress/", "data/", "train/", "validation/",
                           "throughput/", "checkpoint/", "diagnostics/"):
                self._run.define_metric(prefix + "*", step_metric="progress/global_update")
            url = getattr(self._run, "url", None)
            if url:
                print(f"[monitor] W&B {'resumed' if existing else 'created'}: {url}", flush=True)
            return url
        except Exception as error:  # noqa: BLE001 - never break training
            print(f"[monitor] W&B unavailable ({type(error).__name__}); continuing "
                  "console-only. Scientific training is unaffected.", flush=True)
            self.enabled = False
            return None

    def log(self, metrics: Mapping[str, Any]) -> None:
        if self._run is None:
            return
        try:
            self._run.log({k: v for k, v in metrics.items() if v is not None})
        except Exception:  # noqa: BLE001
            pass

    def finish(self) -> None:
        if self._run is not None:
            try:
                self._run.finish()
            except Exception:  # noqa: BLE001
                pass
            self._run = None
            self._key = None


CAMPAIGN_PROVENANCE_KEYS = (
    # Exactly the production `CampaignIdentity` field set ...
    "repository_head",
    "protocol_version",
    "corpus_manifest_digest",
    "encoder_checkpoint",
    "encoder_revision",
    "precision",
    "inventory_source_name",
    "inventory_source_revision",
    "inventory_sha256",
    # ... plus the VERIFIED Stage-6 corpus pin, which is not part of campaign
    # identity but is safe, useful provenance for a dashboard.
    "corpus_dataset",
    "corpus_revision",
)
"""Campaign-level provenance. Every value originates in production identity --
this tuple selects, it never defines."""

CANDIDATE_CONFIG_KEYS = (
    "stage", "label", "lr", "r", "seed", "init_seed", "corruption_seed",
    "batch_size", "cap", "candidate_index", "candidate_count",
    "train_chunks", "resumed", "execution_mode",
)
"""Per-candidate scalars."""

SAFE_CONFIG_KEYS = tuple(dict.fromkeys(CANDIDATE_CONFIG_KEYS + CAMPAIGN_PROVENANCE_KEYS))
"""Exactly what may reach a dashboard. Provenance and scalars only — no text,
no corpus, no checkpoints, no weights, no code."""


def safe_config(event: Mapping[str, Any], extra: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Whitelist filter. Anything not named here cannot reach W&B."""
    config = {k: event[k] for k in SAFE_CONFIG_KEYS if k in event}
    for k, v in (extra or {}).items():
        if k in SAFE_CONFIG_KEYS:
            config[k] = v
    return config


# ---------------------------------------------------------------------------
# Console
# ---------------------------------------------------------------------------
def _fmt_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "--"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def render_progress(state: MonitorState, rate: float | None) -> str:
    """Concise training block. EXACT values and DERIVED values are labelled."""
    p = state.pass_metrics()
    cand = state.candidate or {}
    frac = state.run_fraction()
    eta = ((state.cap - state.global_update) / rate) if (rate and rate > 0) else None
    lines = [
        f"{str(cand.get('stage', '?')).upper()} | candidate "
        f"{cand.get('candidate_index', '?')}/{cand.get('candidate_count', '?')} | "
        f"{cand.get('label', '?')} | TRAIN",
        f"  update        {state.global_update} / {state.cap}"
        + (f" [{100 * frac:.2f}%]" if frac is not None else "")
        + "        (exact)",
        f"  sample-visits {p['sample_visits_total']}                    (derived)",
        f"  pass-equiv    {p['corpus_pass_equivalent']:.5f}" if p["corpus_pass_equivalent"] is not None
        else "  pass-equiv    --",
        f"  current pass  {p['sample_visits_in_current_pass']} / {state.train_chunks}"
        + (f" [{p['pass_percent']:.2f}%]" if p["pass_percent"] is not None else "")
        + "   (derived, chunk-visits)",
    ]
    prog = state.last_progress or {}
    for key in ("loss", "loss_align", "loss_clean"):
        if isinstance(prog.get(key), (int, float)):
            lines.append(f"  {key:<13} {prog[key]:.6f}                     (exact)")
    if rate:
        lines.append(f"  speed         {rate:.3f} updates/s                  (estimate)")
        lines.append(f"  throughput    {rate * state.batch_size:.1f} sample-visits/s        (estimate)")
    nxt = state.next_eval_update()
    if nxt is not None:
        lines.append(f"  next eval     update {nxt}                     (exact cadence)")
    lines.append(f"  ETA           {_fmt_seconds(eta)}                       (estimate)")
    return "\n".join(lines)


def render_validation(event: Mapping[str, Any]) -> str:
    d = event.get("distances") or {}
    lines = ["", "=" * 66,
             f"VALIDATION @ update {event.get('update')}   (exact, production ValidationPoint)",
             "-" * 66]
    for condition in ("FULL", "P50", "P100", "STRIP_ALL"):
        if condition in d:
            lines.append(f"  {condition:<10} {d[condition]:.6f}")
    lines.append(f"  {'d_clean':<10} {event.get('d_clean')}")
    lines.append(f"  {'score':<10} {event.get('score')}   (derived by production, worst-case)")
    lines.append("=" * 66)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable,
                        help="the ACCEPTED SCIENTIFIC python executable (torch/transformers). "
                             "Must NOT be the monitoring venv.")
    parser.add_argument("--project", default="unmark-stage1")
    parser.add_argument("--group", default=None,
                        help="W&B group; defaults to the stage name from telemetry")
    parser.add_argument("--state-dir", default=".unmark-monitor",
                        help="OPERATIONAL state only: W&B run ids and the telemetry "
                             "mirror. Place it on Drive so it survives a Colab runtime "
                             "deletion, e.g. /content/drive/MyDrive/UNMARK/"
                             "stage1-monitoring/<campaign-head>/. Nothing scientific "
                             "lives here: losing it costs dashboard continuity, never "
                             "a checkpoint.")
    parser.add_argument("--telemetry-log", default=None,
                        help="JSONL mirror of every telemetry event (default: state-dir)")
    parser.add_argument("--no-wandb", action="store_true", help="console only")
    parser.add_argument("--wandb-mode", default=None, choices=["online", "offline", "disabled"])
    parser.add_argument("runner_args", nargs=argparse.REMAINDER,
                        help="everything after -- is passed to scripts/stage1_runner.py")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runner_args = [a for a in args.runner_args if a != "--"]
    if not runner_args:
        print("REFUSED: no runner arguments. Example:\n"
              "  python scripts/stage1_wandb_monitor.py --python /path/to/sci/python -- "
              "lr-pilot --prepared-corpus ... --output-dir ...", file=sys.stderr)
        return 2

    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)
    mirror_path = Path(args.telemetry_log) if args.telemetry_log else state_dir / "telemetry.jsonl"

    environment = dict(os.environ)
    environment["UNMARK_TELEMETRY"] = "1"          # turn the emitter on
    environment["PYTHONUNBUFFERED"] = "1"

    command = [args.python, "-B", str(REPO_ROOT / "scripts" / "stage1_runner.py"), *runner_args]
    print(f"[monitor] launching scientific process: {' '.join(command)}", flush=True)

    child = subprocess.Popen(  # noqa: S603 - operator-supplied scientific command
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=environment, cwd=str(REPO_ROOT),
    )

    def forward_sigint(signum, frame):  # noqa: ARG001
        # Forward, never kill. Production fail-closed resume must be preserved.
        print("\n[monitor] SIGINT -> forwarding to the scientific process; "
              "checkpoints and output are left untouched.", flush=True)
        try:
            child.send_signal(signal.SIGINT)
        except Exception:  # noqa: BLE001
            pass

    signal.signal(signal.SIGINT, forward_sigint)

    state = MonitorState()
    bridge = WandbBridge(args.project, args.group, state_dir,
                         enabled=not args.no_wandb, mode=args.wandb_mode)
    last_output = time.monotonic()
    last_heartbeat = 0.0
    last_rate_sample: tuple[float, int] | None = None
    rate: float | None = None
    warned_stall = False

    with open(mirror_path, "a", encoding="utf-8") as mirror:
        assert child.stdout is not None
        for line in child.stdout:
            line = line.rstrip("\n")
            last_output = time.monotonic()
            warned_stall = False
            event = parse_line(line)
            if event is None:
                print(line, flush=True)          # ordinary runner prose
            else:
                mirror.write(json.dumps(event, sort_keys=True) + "\n")
                mirror.flush()
                kind = state.consume(event)
                _handle(kind, event, state, bridge, args)
                if kind == "train_progress":
                    now = time.monotonic()
                    if last_rate_sample is not None:
                        dt = now - last_rate_sample[0]
                        du = state.global_update - last_rate_sample[1]
                        if dt > 0 and du > 0:
                            rate = du / dt
                    last_rate_sample = (now, state.global_update)
                    print(render_progress(state, rate), flush=True)

            now = time.monotonic()
            if now - last_heartbeat >= HEARTBEAT_SECONDS and state.phase:
                last_heartbeat = now
                _heartbeat(state, child.pid, now - last_output)

    code = child.wait()
    bridge.finish()
    print(f"[monitor] scientific process exited with code {code}", flush=True)
    print(f"[monitor] telemetry mirrored to {mirror_path}", flush=True)
    return code


def _handle(kind: str, event: Mapping[str, Any], state: MonitorState,
            bridge: WandbBridge, args: Any) -> None:
    if kind == "stage_phase":
        if event.get("state") == "START":
            print(f"[phase] {event.get('phase')} START", flush=True)
        elif event.get("state") == "DONE":
            print(f"[phase] {event.get('phase')} DONE in "
                  f"{event.get('elapsed_phase_s')}s", flush=True)
        elif event.get("state") == "FAILED":
            print(f"[phase] {event.get('phase')} FAILED ({event.get('error_type')})", flush=True)
        return

    if kind == "run_start":
        group = args.group or str(event.get("stage", "")).replace("_", "-")
        bridge.group = group
        # Candidate scalars + authoritative campaign provenance, both filtered
        # through the same whitelist.
        bridge.start_candidate(event, safe_config(event, state.campaign))
        print(f"\n[run] {event.get('stage')} candidate "
              f"{event.get('candidate_index')}/{event.get('candidate_count')} "
              f"{event.get('label')} seed={event.get('seed')} cap={event.get('cap')}", flush=True)
        return

    if kind == "train_progress":
        p = state.pass_metrics()
        bridge.log({
            "progress/global_update": state.global_update,
            "progress/cap": state.cap,
            "progress/run_fraction": state.run_fraction(),
            "progress/run_percent": (state.run_fraction() or 0) * 100,
            "data/sample_visits_total": p["sample_visits_total"],
            "data/corpus_pass_equivalent": p["corpus_pass_equivalent"],
            "data/pass_index": p["pass_index"],
            "data/sample_visits_in_current_pass": p["sample_visits_in_current_pass"],
            "data/pass_fraction": p["pass_fraction"],
            "data/pass_percent": p["pass_percent"],
            "train/loss": event.get("loss"),
            "train/loss_align": event.get("loss_align"),
            "train/loss_clean": event.get("loss_clean"),
            **{f"diagnostics/{k}": v for k, v in state.diagnostics().items()},
        })
        return

    if kind == "validation":
        print(render_validation(event), flush=True)
        d = event.get("distances") or {}
        bridge.log({
            "progress/global_update": event.get("update"),
            **{f"validation/{c}": d.get(c) for c in ("FULL", "P50", "P100", "STRIP_ALL")},
            "validation/d_clean": event.get("d_clean"),
            "validation/score": event.get("score"),
            **{f"diagnostics/{k}": v for k, v in state.diagnostics().items()},
        })
        return

    if kind == "checkpoint":
        print(f"[checkpoint] saved at update {event.get('update')} "
              f"(cap {event.get('cap')}, best={event.get('is_best')}) "
              f"-> {event.get('checkpoint_name')}", flush=True)
        bridge.log({"progress/global_update": event.get("update"),
                    "checkpoint/latest_update": event.get("update")})
        return

    if kind == "run_end":
        print(f"[run] complete: update {event.get('global_update')} cap {event.get('cap')} "
              f"continued={event.get('continued_past_initial_budget')} "
              f"budget_limited={event.get('budget_limited')} "
              f"selected@{event.get('selected_update')} "
              f"score={event.get('selected_score')}", flush=True)
        bridge.finish()
        return

    if kind == "selection":
        print(f"\n[selection] {event.get('stage')} -> {event.get('selected')}", flush=True)
        return

    if kind == "stage_complete":
        print(f"[stage] complete -> {event.get('artifact_path')}", flush=True)
        return


def _heartbeat(state: MonitorState, pid: int, idle: float) -> None:
    gpu = sample_gpu()
    proc = sample_process(pid)
    elapsed = time.monotonic() - state.phase_started if state.phase_started else 0.0
    bits = [f"[{time.strftime('%H:%M:%S')}]", f"phase={state.phase}",
            f"elapsed={_fmt_seconds(elapsed)}", f"stdout_idle={idle:.0f}s"]
    if proc:
        bits.append(f"CPU={proc.get('proc_cpu_percent', '?')}%")
        bits.append(f"RSS={proc.get('proc_rss_mib', 0):.0f}MiB")
        bits.append(f"threads={proc.get('proc_threads', '?')}")
    if gpu:
        bits.append(f"GPU={gpu.get('gpu_util_percent', '?')}%")
        bits.append(f"VRAM={gpu.get('vram_used_mib', 0):.0f}/{gpu.get('vram_total_mib', 0):.0f}MiB")
        bits.append(f"power={gpu.get('gpu_power_w', '?')}W")
        bits.append(f"temp={gpu.get('gpu_temp_c', '?')}C")
    print(" ".join(bits), flush=True)

    quiet_cpu = proc.get("proc_cpu_percent", 100.0) < 1.0 if proc else False
    quiet_gpu = gpu.get("gpu_util_percent", 100.0) < 1.0 if gpu else False
    if idle > STALL_WARN_SECONDS and quiet_cpu and quiet_gpu:
        # OPERATIONAL WARNING ONLY. Never auto-kill scientific training.
        print(f"[monitor] WARNING: no telemetry for {idle:.0f}s with near-zero CPU and GPU. "
              "This may be a stall. NOT intervening — scientific training is never "
              "auto-killed by the monitor.", flush=True)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
