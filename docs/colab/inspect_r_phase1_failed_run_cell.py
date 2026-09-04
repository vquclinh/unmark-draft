# ==========================================================================================
# UNMARK - ONE CELL: INSPECT A FAILED OR INTERRUPTED R-PHASE1 RUN
#
# Default target is the first accelerated r-phase1 attempt:
#   /content/drive/MyDrive/UNMARK/UNMARK-BACKUP/stage1-training/a814a3081bdd/r-phase1
#
# This cell is read-only. It does not delete, resume, rewrite or train anything.
# ==========================================================================================

import datetime
import json
from collections import Counter, deque
from pathlib import Path


def banner(n, title):
    print("\n" + "=" * 110)
    print(f"{n} - {title}")
    print("=" * 110)


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as error:
        return {"__read_error__": f"{type(error).__name__}: {error}"}


def tail(path, *, lines=120):
    path = Path(path)
    print("\n" + "-" * 110)
    print("TAIL:", path)
    print("-" * 110)
    if not path.is_file():
        print("missing")
        return
    buf = deque(maxlen=lines)
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            buf.append(line.rstrip("\n"))
    for line in buf:
        print(line)


def mtime(path):
    path = Path(path)
    if not path.exists():
        return "-"
    return datetime.datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")


DRIVE = Path("/content/drive/MyDrive/UNMARK/UNMARK-BACKUP")
TAG = "a814a3081bdd"

OUTPUT = DRIVE / "stage1-training" / TAG / "r-phase1"
MONITOR_STATE = DRIVE / "stage1-monitoring" / TAG / "r-phase1"
TELEMETRY = MONITOR_STATE / "telemetry.jsonl"
RUN_LOG = MONITOR_STATE / "r_phase1_monitor.stdout.log"
WANDB_IDS = MONITOR_STATE / "wandb_run_ids.json"


banner("1 / 5", "TARGETS")
print("Drive root    :", DRIVE)
print("Tag           :", TAG)
print("Output        :", OUTPUT)
print("Monitor state :", MONITOR_STATE)
print("Telemetry     :", TELEMETRY)
print("Run log       :", RUN_LOG)


banner("2 / 5", "OUTPUT ARTIFACTS")
print("output exists :", OUTPUT.exists())
print("r_phase1.json :", (OUTPUT / "r_phase1.json").is_file(), mtime(OUTPUT / "r_phase1.json"))

for run_dir in sorted(OUTPUT.glob("run-r*")):
    last = run_dir / "_checkpoint" / "training-checkpoint-last.pt"
    best = run_dir / "_checkpoint" / "training-checkpoint-best.pt"
    run_json = run_dir.with_suffix(".json")
    print("\n", run_dir.name)
    print("  run json :", run_json.is_file(), mtime(run_json))
    print("  last ckpt:", last.is_file(), mtime(last), f"{last.stat().st_size / 1e6:.1f} MB" if last.is_file() else "")
    print("  best ckpt:", best.is_file(), mtime(best), f"{best.stat().st_size / 1e6:.1f} MB" if best.is_file() else "")


banner("3 / 5", "TELEMETRY EVENTS")
if TELEMETRY.is_file():
    events = []
    with TELEMETRY.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                events.append(json.loads(line))
            except Exception:
                pass
    print("parseable events:", len(events))
    print("by event        :", dict(Counter(e.get("event") for e in events)))
    by_candidate = Counter(
        f"{e.get('candidate_index')}:{e.get('label')}"
        for e in events
        if e.get("candidate_index") is not None
    )
    print("by candidate   :", dict(by_candidate))
    failed = [e for e in events if e.get("state") == "FAILED" or e.get("error_type")]
    if failed:
        print("\nfailed phase/event tail:")
        for e in failed[-20:]:
            print(json.dumps(e, ensure_ascii=False, sort_keys=True))
    print("\nlast 40 telemetry events:")
    for e in events[-40:]:
        print(
            " | ".join(
                str(x)
                for x in (
                    f"event={e.get('event')}",
                    f"phase={e.get('phase')}" if e.get("phase") else None,
                    f"state={e.get('state')}" if e.get("state") else None,
                    f"error={e.get('error_type')}" if e.get("error_type") else None,
                    f"label={e.get('label')}" if e.get("label") else None,
                    f"update={e.get('global_update', e.get('update'))}"
                    if e.get("global_update", e.get("update")) is not None else None,
                    f"cap={e.get('cap')}" if e.get("cap") is not None else None,
                )
                if x is not None
            )
        )
else:
    print("telemetry missing")


banner("4 / 5", "W&B IDS")
if WANDB_IDS.is_file():
    print(WANDB_IDS.read_text(encoding="utf-8", errors="replace"))
else:
    print("missing:", WANDB_IDS)


banner("5 / 5", "RAW LOG TAIL")
tail(RUN_LOG)
