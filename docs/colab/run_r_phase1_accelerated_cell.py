# ==========================================================================================
# UNMARK - ONE CELL: REISSUE LR HANDOFF, THEN RUN ACCELERATED R-PHASE1
#
# PURPOSE:
#   1. Do NOT rerun the 3 LR candidates.
#   2. Reissue lr_pilot.json under the current pushed code HEAD, preserving the
#      transparent author override to lr=0.0001.
#   3. Start r-phase1 with operational acceleration:
#        - UNMARK_STAGE1_PREPARATION_WORKERS=<auto bounded worker count>
#        - UNMARK_STAGE1_R_PHASE1_EXECUTION=fused
#
# SCIENTIFIC VALUES STAY LOCKED:
#   batch=128, max_updates=20000, validation cadence=500, fp32 deterministic,
#   r grid={0.25, 0.5, 1, 2, 4}, lr read from lr_pilot.json.
#
# CAMPAIGN ROOT:
#   /content/drive/MyDrive/UNMARK/UNMARK-BACKUP
# ==========================================================================================

import getpass
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


def fail(msg):
    raise RuntimeError(
        "\n"
        + "=" * 110
        + "\nACCELERATED R-PHASE1 REFUSED\n"
        + "=" * 110
        + "\n"
        + str(msg)
    )


def banner(n, title):
    print("\n" + "=" * 110)
    print(f"{n} - {title}")
    print("=" * 110)


def run(cmd, cwd=None, env=None):
    cmd = [str(x) for x in cmd]
    print("\n$", shlex.join(cmd), flush=True)
    completed = subprocess.run(cmd, cwd=cwd, env=env)
    if completed.returncode != 0:
        fail(f"Command failed rc={completed.returncode}:\n" + shlex.join(cmd))
    return completed


def capture(cmd, cwd=None, env=None):
    return subprocess.check_output(
        [str(x) for x in cmd],
        cwd=cwd,
        env=env,
        text=True,
    ).strip()


REPO_URL = "https://github.com/vquclinh/unmark-draft.git"
REPO = Path("/content/UNMARK")
SCI_PY = Path("/usr/bin/python3")
MON_VENV = Path("/content/unmark-monitoring-venv")
MON_PY = MON_VENV / "bin" / "python"

DRIVE = Path("/content/drive/MyDrive/UNMARK/UNMARK-BACKUP")
BACKBONE_REVISION = "01daacda68afe13d83023d16ec647239e344a1e6"

PREPARED = DRIVE / "stage1-prepared" / "aa49785eadcb"
COMPLETION = DRIVE / "stage1-checkpoints" / "aa49785eadcb"
PROJECT = "unmark-stage1"


# ==========================================================================================
# 1. MOUNT DRIVE
# ==========================================================================================

banner("1 / 8", "MOUNT DRIVE")

if not Path("/content/drive/MyDrive").is_dir():
    from google.colab import drive
    drive.mount("/content/drive")
else:
    print("Drive already mounted.")

if not DRIVE.is_dir():
    fail(f"UNMARK-BACKUP root is missing:\n{DRIVE}")

print("Campaign root:", DRIVE)


# ==========================================================================================
# 2. PROCESS SAFETY
# ==========================================================================================

banner("2 / 8", "PROCESS SAFETY")

active = subprocess.run(
    [
        "bash",
        "-lc",
        r"""
ps -eo pid,ppid,stat,etime,cmd |
grep -E '[s]tage1_runner\.py|[s]tage1_wandb_monitor\.py' || true
""",
    ],
    text=True,
    stdout=subprocess.PIPE,
).stdout.strip()

if active:
    print(active)
    fail("A Stage-1 process is already running.")

print("No Stage-1 process alive.")


# ==========================================================================================
# 3. RESTORE CURRENT REPO
# ==========================================================================================

banner("3 / 8", "RESTORE CURRENT REPO")

if not REPO.exists():
    run(["git", "clone", "--quiet", REPO_URL, REPO])

if not (REPO / ".git").is_dir():
    fail(f"Not a Git repository: {REPO}")

repo_wandb = REPO / "wandb"
if repo_wandb.exists():
    target = Path("/content/unmark-wandb-old-operational-before-r-phase1")
    if target.exists():
        shutil.rmtree(target)
    shutil.move(str(repo_wandb), str(target))
    print("Moved repo-local W&B debris to:", target)

run(["git", "fetch", "--quiet", "origin", "main"], cwd=REPO)
run(["git", "checkout", "--quiet", "--detach", "origin/main"], cwd=REPO)

status = capture(["git", "status", "--porcelain"], cwd=REPO)
if status:
    fail("Repository is dirty after checkout:\n" + status)

HEAD = capture(["git", "rev-parse", "HEAD"], cwd=REPO)
TAG = HEAD[:12]
print("Current execution HEAD:", HEAD)


# ==========================================================================================
# 4. REISSUE LR-PILOT ARTIFACT UNDER THIS HEAD
# ==========================================================================================

banner("4 / 8", "REISSUE LR-PILOT HANDOFF")

helper = REPO / "docs" / "colab" / "regenerate_lr_pilot_author_override_cell.py"
if not helper.is_file():
    fail(f"Missing helper cell in this checkout:\n{helper}")

# The helper verifies the old completed run-lr*.json files, preserves the
# author's lr=0.0001 override, rewrites lr_pilot.json to the current HEAD, and
# validates exactly what r-phase1 will read.
helper_globals = {"__name__": "__unmark_lr_reissue_helper__"}
exec(compile(helper.read_text(encoding="utf-8"), str(helper), "exec"), helper_globals)

LR_ARTIFACT = helper_globals["LR_PILOT_ARTIFACT"]
helper_head = helper_globals["current_head"]
if helper_head != HEAD:
    fail(
        "The LR helper reissued under a different HEAD.\n"
        f"helper: {helper_head}\n"
        f"runner: {HEAD}"
    )


# ==========================================================================================
# 5. INSTALL SCIENTIFIC + MONITORING ENVIRONMENTS
# ==========================================================================================

banner("5 / 8", "INSTALL RUNTIME")

EXP_REQ = REPO / "requirements" / "experiment.txt"
MON_REQ = REPO / "requirements" / "monitoring.txt"
MONITOR = REPO / "scripts" / "stage1_wandb_monitor.py"

for path in (EXP_REQ, MON_REQ, MONITOR, PREPARED / "chunks.jsonl", COMPLETION / "COMPLETE.json"):
    if not path.is_file():
        fail(f"Missing required file:\n{path}")

run([SCI_PY, "-m", "pip", "install", "-q", "-r", EXP_REQ])

if not MON_PY.is_file():
    run([SCI_PY, "-m", "pip", "install", "-q", "virtualenv"])
    run([SCI_PY, "-m", "virtualenv", MON_VENV])

run([MON_PY, "-m", "pip", "install", "-q", "-r", MON_REQ])

print("Scientific runtime ready.")
print("Monitoring runtime ready.")


# ==========================================================================================
# 6. ACCELERATION SETTINGS
# ==========================================================================================

banner("6 / 8", "ACCELERATION SETTINGS")

cpu_count = os.cpu_count() or 8
PREPARATION_WORKERS = min(24, max(8, cpu_count - 2))

OUTPUT = DRIVE / "stage1-training" / TAG / "r-phase1"
CACHE = DRIVE / "stage1-cache" / TAG
MONITOR_STATE = DRIVE / "stage1-monitoring" / TAG / "r-phase1"

if (OUTPUT / "r_phase1.json").is_file():
    fail(
        "r_phase1.json already exists; this stage appears complete.\n"
        f"Inspect instead of relaunching:\n{OUTPUT / 'r_phase1.json'}"
    )

CACHE.mkdir(parents=True, exist_ok=True)
MONITOR_STATE.mkdir(parents=True, exist_ok=True)

print("Preparation workers:", PREPARATION_WORKERS)
print("r-phase1 execution :", "fused")
print("Output             :", OUTPUT)
print("Cache              :", CACHE)
print("Monitor state      :", MONITOR_STATE)


# ==========================================================================================
# 7. W&B AUTH + OPERATIONAL DIRECTORIES
# ==========================================================================================

banner("7 / 8", "W&B CONTINUITY")

env = os.environ.copy()
env["PYTHONUNBUFFERED"] = "1"
env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
env["UNMARK_STAGE1_PREPARATION_WORKERS"] = str(PREPARATION_WORKERS)
env["UNMARK_STAGE1_R_PHASE1_EXECUTION"] = "fused"

WBROOT = Path("/content/unmark-wandb-runtime-r-phase1-accelerated")
for name in ("runs", "cache", "config", "artifacts"):
    (WBROOT / name).mkdir(parents=True, exist_ok=True)

env["WANDB_DIR"] = str(WBROOT / "runs")
env["WANDB_CACHE_DIR"] = str(WBROOT / "cache")
env["WANDB_CONFIG_DIR"] = str(WBROOT / "config")
env["WANDB_ARTIFACT_DIR"] = str(WBROOT / "artifacts")

wandb_key = env.get("WANDB_API_KEY")
if not wandb_key:
    try:
        from google.colab import userdata
        secret = userdata.get("WANDB_API_KEY")
        if secret:
            wandb_key = secret.strip()
    except Exception:
        pass

netrc = Path.home() / ".netrc"
has_netrc = (
    netrc.is_file()
    and "api.wandb.ai" in netrc.read_text(encoding="utf-8", errors="ignore")
)

if not wandb_key and not has_netrc:
    wandb_key = getpass.getpass("W&B API key (hidden): ").strip()
    if not wandb_key:
        fail("No W&B authentication available.")

if wandb_key:
    env["WANDB_API_KEY"] = wandb_key

env.pop("WANDB_ENTITY", None)

print("W&B operational root:", WBROOT)
print("W&B state ids       :", MONITOR_STATE / "wandb_run_ids.json")


# ==========================================================================================
# 8. RUN ACCELERATED R-PHASE1
# ==========================================================================================

banner("8 / 8", "RUN ACCELERATED R-PHASE1")

resume_args = ["--resume"] if OUTPUT.exists() else []

CMD = [
    str(MON_PY),
    "-u",
    str(MONITOR),
    "--python",
    str(SCI_PY),
    "--project",
    PROJECT,
    "--group",
    "r-phase1",
    "--state-dir",
    str(MONITOR_STATE),
    "--wandb-mode",
    "online",
    "--",
    "r-phase1",
    "--prepared-corpus",
    str(PREPARED),
    "--completion-dir",
    str(COMPLETION),
    "--output-dir",
    str(OUTPUT),
    *resume_args,
    "--cache-root",
    str(CACHE),
    "--revision",
    BACKBONE_REVISION,
    "--repository-head",
    HEAD,
    "--lr-artifact",
    str(LR_ARTIFACT),
]

for forbidden in (
    "--lr",
    "--r",
    "--batch-size",
    "--epochs",
    "--max-updates",
    "--max-length",
    "--pi-strip",
):
    assert forbidden not in CMD

print("REAL ACCELERATED R-PHASE1 COMMAND:")
print(shlex.join(CMD))

print(
    f"""
--------------------------------------------------------------------------------------------------------------
ACCELERATED R-PHASE1 STARTING

Drive root:
    {DRIVE}

Repository:
    {HEAD}

LR artifact:
    {LR_ARTIFACT}

Selected LR:
    0.0001, from transparent author override in lr_pilot.json

Operational acceleration:
    UNMARK_STAGE1_PREPARATION_WORKERS={PREPARATION_WORKERS}
    UNMARK_STAGE1_R_PHASE1_EXECUTION=fused

Resume mode:
    {'YES - existing output directory will be resumed' if resume_args else 'NO - fresh r-phase1 output directory'}

This does NOT rerun lr-pilot.
This does NOT change batch size, r grid, update budget, precision or validation cadence.
--------------------------------------------------------------------------------------------------------------
""",
    flush=True,
)

result = subprocess.run(CMD, cwd=REPO, env=env)

print("\n" + "=" * 110)
if result.returncode == 0:
    print("ACCELERATED R-PHASE1 COMPLETED")
    print("artifact:", OUTPUT / "r_phase1.json")
else:
    print(f"ACCELERATED R-PHASE1 EXITED rc={result.returncode}")
    print(
        """
DO NOT DELETE:
  stage1-training
  _checkpoint directories
  stage1-monitoring
  wandb_run_ids.json

Retry this same cell to resume. If fused resume refuses because candidates are
not aligned at the same checkpoint boundary, set:
  env["UNMARK_STAGE1_R_PHASE1_EXECUTION"] = "sequential"
inside this cell and retry with --resume.
"""
    )
print("=" * 110)
