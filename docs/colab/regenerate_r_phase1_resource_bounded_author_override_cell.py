# ==========================================================================================
# UNMARK - ONE CELL: REISSUE RESOURCE-BOUNDED R-PHASE1 HANDOFF TO R=1.0
#
# PURPOSE:
#   Do NOT rerun lr-pilot.
#   Do NOT rerun the five-r sweep.
#   Do NOT modify or delete per-r checkpoints.
#   Rebuild r_phase1.json from the stopped update-6500 checkpoint payloads.
#   Record transparently that r=1.0 is an author amendment after partial
#   r-phase1 validation-curve review, not a completed 20,000-update selection
#   and not a claim of global optimality.
#
# CAMPAIGN ROOT:
#   /content/drive/MyDrive/UNMARK/UNMARK-BACKUP
#
# SOURCE R-PHASE1 EVIDENCE:
#   /content/drive/MyDrive/UNMARK/UNMARK-BACKUP/stage1-training/3bb2944e6f71/r-phase1
# ==========================================================================================

import datetime
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


# ==========================================================================================
# CONFIG
# ==========================================================================================

REPO_URL = "https://github.com/vquclinh/unmark-draft.git"
REPO = Path("/content/UNMARK")
SCI_PY = Path("/usr/bin/python3")

# Use latest pushed main by default. Set UNMARK_REPO_REF to pin the exact
# implementation commit reported by the handoff.
REPO_REF = os.environ.get("UNMARK_REPO_REF", "origin/main")

DRIVE = Path("/content/drive/MyDrive/UNMARK/UNMARK-BACKUP")

SOURCE_R_PHASE1_HEAD = "3bb2944e6f71865d5a37fe403b78ea640f8a3f1d"
SOURCE_R_PHASE1_TAG = SOURCE_R_PHASE1_HEAD[:12]
R_PHASE1_DIR = DRIVE / "stage1-training" / SOURCE_R_PHASE1_TAG / "r-phase1"
R_PHASE1_ARTIFACT = R_PHASE1_DIR / "r_phase1.json"
R_PHASE1_MONITOR_DIR = DRIVE / "stage1-monitoring" / SOURCE_R_PHASE1_TAG / "r-phase1"
R_PHASE1_TELEMETRY = R_PHASE1_MONITOR_DIR / "telemetry.jsonl"

AUTHOR = "Linh Vo Quoc"
BACKBONE_REVISION = "01daacda68afe13d83023d16ec647239e344a1e6"

EXPECTED_RESOURCE_BOUNDED_SUMMARIES = {
    0.25: {
        "median_score": 0.1192311382,
        "mean_score": 0.1296373979,
        "median_d_clean": 0.0818558441,
        "score_range": 0.0903892273,
        "score_std": 0.0292188811,
        "score_at_cutoff": 0.1909501625,
    },
    0.5: {
        "median_score": 0.1161014018,
        "mean_score": 0.1148930311,
        "median_d_clean": 0.0723250312,
        "score_range": 0.0211923227,
        "score_std": 0.0082052169,
        "score_at_cutoff": 0.1065570383,
    },
    1.0: {
        "median_score": 0.1056275400,
        "mean_score": 0.1105911372,
        "median_d_clean": 0.0767193542,
        "score_range": 0.0383433552,
        "score_std": 0.0125758211,
        "score_at_cutoff": 0.1040437879,
    },
    2.0: {
        "median_score": 0.1174547335,
        "mean_score": 0.1150982624,
        "median_d_clean": 0.0668128509,
        "score_range": 0.0184510154,
        "score_std": 0.0069971263,
        "score_at_cutoff": 0.1203045395,
    },
    4.0: {
        "median_score": 0.1295327435,
        "mean_score": 0.1311204687,
        "median_d_clean": 0.0786890076,
        "score_range": 0.0299517233,
        "score_std": 0.0108133741,
        "score_at_cutoff": 0.1224059213,
    },
}

EXPECTED_RESOURCE_BOUNDED_ORDER = ["r=1", "r=0.5", "r=2", "r=0.25", "r=4"]


# ==========================================================================================
# HELPERS
# ==========================================================================================

def fail(msg):
    raise RuntimeError(
        "\n"
        + "=" * 110
        + "\nR-PHASE1 RESOURCE-BOUNDED AMENDMENT REFUSED\n"
        + "=" * 110
        + "\n"
        + str(msg)
    )


def banner(n, title):
    print("\n" + "=" * 110)
    print(f"{n} - {title}")
    print("=" * 110)


def capture(cmd, cwd=None, env=None):
    return subprocess.check_output(
        [str(x) for x in cmd],
        cwd=cwd,
        env=env,
        text=True,
    ).strip()


def run(cmd, cwd=None, env=None):
    cmd = [str(x) for x in cmd]
    print("\n$", shlex.join(cmd), flush=True)
    completed = subprocess.run(cmd, cwd=cwd, env=env)
    if completed.returncode != 0:
        fail(f"Command failed rc={completed.returncode}:\n" + shlex.join(cmd))
    return completed


def require_clean_repo_checkout(repo):
    """Refuse any dirty checkout without moving or deleting W&B state."""
    status = capture(["git", "status", "--porcelain"], cwd=repo)
    if not status:
        return

    fail(
        "Repository checkout is dirty. This helper does not move, delete or "
        "rewrite W&B state; use a fresh runtime or clean checkout.\n"
        + status
    )


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, payload):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    body = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    with tmp.open("wb") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    try:
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


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
    fail("A Stage-1 process is already running. Do not rewrite the handoff artifact now.")

print("No Stage-1 process alive.")


# ==========================================================================================
# 3. RESTORE REPO WITH AMENDMENT SUPPORT
# ==========================================================================================

banner("3 / 8", "RESTORE REPO WITH AMENDMENT SUPPORT")

if not REPO.exists():
    run(["git", "clone", "--quiet", REPO_URL, REPO])

if not (REPO / ".git").is_dir():
    fail(f"Not a Git repository: {REPO}")

require_clean_repo_checkout(REPO)

run(["git", "fetch", "--quiet", "origin", "main"], cwd=REPO)
run(["git", "checkout", "--quiet", "--detach", REPO_REF], cwd=REPO)

require_clean_repo_checkout(REPO)

status = capture(["git", "status", "--porcelain"], cwd=REPO)
if status:
    fail("Repository is dirty after checkout:\n" + status)

HEAD = capture(["git", "rev-parse", "HEAD"], cwd=REPO)
TAG = HEAD[:12]
print("Source r-phase1 run HEAD:", SOURCE_R_PHASE1_HEAD)
print("Current amendment-support HEAD:", HEAD)

os.chdir(REPO)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from unmark.stage1.checkpoint import repository_execution_modifications  # noqa: E402

mods = repository_execution_modifications(REPO)
if mods:
    fail(
        "Execution code is dirty; final-main would refuse this tree:\n"
        + "\n".join(mods)
    )

print("Repository execution tree is clean.")


# ==========================================================================================
# 4. REISSUE LR-PILOT UNDER THIS SAME HEAD
# ==========================================================================================

banner("4 / 9", "FETCH AND VERIFY PINNED INVENTORY")

run([SCI_PY, "-m", "pip", "install", "-q", "-r", REPO / "requirements" / "experiment.txt"])
run([SCI_PY, REPO / "scripts" / "fetch_vietnamese_syllable_inventory.py"], cwd=REPO)

from unmark.stage1.preflight import verify_scientific_inputs  # noqa: E402

scientific_inputs = verify_scientific_inputs(repo_root=REPO)
inventory = scientific_inputs.inventory
print("Pinned inventory verified:")
print("  source_name    :", inventory.source_name)
print("  source_revision:", inventory.source_revision)
print("  sha256         :", inventory.sha256)


# ==========================================================================================
# 5. REISSUE LR-PILOT UNDER THIS SAME HEAD
# ==========================================================================================

banner("5 / 9", "REISSUE LR-PILOT HANDOFF")

lr_helper = REPO / "docs" / "colab" / "regenerate_lr_pilot_author_override_cell.py"
if not lr_helper.is_file():
    fail(f"Missing LR helper cell:\n{lr_helper}")

helper_globals = {"__name__": "__unmark_lr_reissue_helper__"}
exec(compile(lr_helper.read_text(encoding="utf-8"), str(lr_helper), "exec"), helper_globals)

LR_PILOT_ARTIFACT = helper_globals["LR_PILOT_ARTIFACT"]
LR_PILOT_DIR = helper_globals["LR_PILOT_DIR"]
lr_helper_head = helper_globals["current_head"]
if lr_helper_head != HEAD:
    fail(
        "The LR helper reissued under a different HEAD.\n"
        f"helper: {lr_helper_head}\n"
        f"r helper: {HEAD}"
    )


# ==========================================================================================
# 6. IMPORT AMENDMENT SUPPORT
# ==========================================================================================

banner("6 / 9", "IMPORT AMENDMENT SUPPORT")

try:
    import torch  # noqa: F401
except Exception:
    run([SCI_PY, "-m", "pip", "install", "-q", "-r", REPO / "requirements" / "experiment.txt"])

from unmark.stage1.artifact import (  # noqa: E402
    CampaignIdentity,
    R_PHASE1_RESOURCE_BOUNDED_AUTHOR_OVERRIDE_KIND,
    validate_selection_artifact,
)
from unmark.stage1.r_phase1_amendment import (  # noqa: E402
    assert_expected_resource_bounded_summaries,
    build_resource_bounded_r_phase1_artifact,
    load_r_phase1_last_checkpoints,
    verify_r_phase1_telemetry_evidence,
)

print("Override kind available:", R_PHASE1_RESOURCE_BOUNDED_AUTHOR_OVERRIDE_KIND)


# ==========================================================================================
# 7. VERIFY INPUT HANDOFFS, STOPPED CHECKPOINTS, AND TELEMETRY
# ==========================================================================================

banner("7 / 9", "VERIFY SOURCE EVIDENCE")

if not R_PHASE1_DIR.is_dir():
    fail(f"r-phase1 source directory is missing:\n{R_PHASE1_DIR}")

lr_artifact = load_json(LR_PILOT_ARTIFACT)
identity = CampaignIdentity(**lr_artifact["identity"])
if identity.repository_head != HEAD:
    fail(
        "lr_pilot.json was not reissued under the current HEAD.\n"
        f"artifact: {identity.repository_head}\n"
        f"current : {HEAD}"
    )

lr_winner = validate_selection_artifact(
    lr_artifact,
    expected_stage="lr_pilot",
    identity=identity,
    what=str(LR_PILOT_ARTIFACT),
)
if lr_winner.learning_rate != 1e-4:
    fail(f"LR handoff winner is {lr_winner.learning_rate!r}, not 0.0001")

checkpoint_payloads, checkpoint_paths = load_r_phase1_last_checkpoints(R_PHASE1_DIR)
print("Loaded stopped r-phase1 checkpoints:")
for r, path in sorted(checkpoint_paths.items()):
    print(f"  r={r:g}: {path}")

telemetry_evidence = verify_r_phase1_telemetry_evidence(
    R_PHASE1_TELEMETRY,
    expected_source_repository_head=SOURCE_R_PHASE1_HEAD,
    expected_learning_rate=lr_winner.learning_rate,
)
print("Telemetry evidence verified:", telemetry_evidence["source_telemetry"])
for label, state in sorted(telemetry_evidence["required_events_by_label"].items()):
    print(
        f"  {label}: run_start={state['run_start']}, "
        f"train_progress_6500={state['train_progress_6500']}, "
        f"validation_6500={state['validation_6500']}, "
        f"checkpoint_6500={state['checkpoint_6500']}"
    )

control_source = LR_PILOT_DIR / "run-lr0.0001.json"
if not control_source.is_file():
    fail(f"Missing historical lr=0.0001,r=1 control run:\n{control_source}")
control_run = load_json(control_source)


# ==========================================================================================
# 8. REBUILD AND VERIFY R-PHASE1 HANDOFF
# ==========================================================================================

banner("8 / 9", "REBUILD RESOURCE-BOUNDED R-PHASE1 ARTIFACT")

previous_artifact = load_json(R_PHASE1_ARTIFACT) if R_PHASE1_ARTIFACT.is_file() else None
artifact = build_resource_bounded_r_phase1_artifact(
    checkpoint_payloads=checkpoint_payloads,
    checkpoint_paths=checkpoint_paths,
    identity=identity,
    source_r_phase1_repository_head=SOURCE_R_PHASE1_HEAD,
    reissued_under_repository_head=HEAD,
    fixed_learning_rate=lr_winner.learning_rate,
    control_run_payload=control_run,
    control_source=control_source,
    telemetry_evidence=telemetry_evidence,
    author=AUTHOR,
    created_at=str(datetime.date.today()),
    previous_artifact=previous_artifact,
    selected_r=1.0,
)

override = artifact["selection_override"]
evidence = override["evidence"]
assert_expected_resource_bounded_summaries(
    evidence["candidate_summaries"],
    EXPECTED_RESOURCE_BOUNDED_SUMMARIES,
)
if evidence["resource_bounded_order"] != EXPECTED_RESOURCE_BOUNDED_ORDER:
    fail(
        "Resource-bounded order mismatch.\n"
        f"computed: {evidence['resource_bounded_order']}\n"
        f"expected: {EXPECTED_RESOURCE_BOUNDED_ORDER}"
    )

print("Computed resource-bounded order:", evidence["resource_bounded_order"])
print("Selected r:", artifact["selected"]["r"])
print("Fixed LR:", artifact["selected"]["learning_rate"])
print("Observed cutoff:", override["observed_cutoff_update"])
print("Original planned cap:", override["original_planned_cap"])
print("Global optimum claimed:", override["global_optimum_claimed"])

backup = None
if R_PHASE1_ARTIFACT.is_file():
    backup = R_PHASE1_ARTIFACT.with_name(
        "r_phase1.before-resource-bounded-r-override."
        + datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
        + ".json"
    )
    shutil.copy2(R_PHASE1_ARTIFACT, backup)

write_json(R_PHASE1_ARTIFACT, artifact)

if backup is not None:
    print("Backup written:", backup)
print("Regenerated:", R_PHASE1_ARTIFACT)


# ==========================================================================================
# 9. VALIDATE EXACTLY WHAT FINAL-MAIN WILL CONSUME
# ==========================================================================================

banner("9 / 9", "VALIDATE FINAL-MAIN HANDOFFS")

reloaded_lr = load_json(LR_PILOT_ARTIFACT)
reloaded_r = load_json(R_PHASE1_ARTIFACT)
final_identity = CampaignIdentity(**reloaded_lr["identity"])
if reloaded_r["identity"] != reloaded_lr["identity"]:
    fail("LR and r handoff identities differ; final-main would refuse the campaign.")

final_lr = validate_selection_artifact(
    reloaded_lr,
    expected_stage="lr_pilot",
    identity=final_identity,
    what=str(LR_PILOT_ARTIFACT),
)
final_r = validate_selection_artifact(
    reloaded_r,
    expected_stage="r_phase1",
    identity=final_identity,
    what=str(R_PHASE1_ARTIFACT),
)

if final_lr.learning_rate != 1e-4:
    fail(f"Final-main LR validation returned {final_lr.learning_rate!r}")
if final_r.learning_rate != final_lr.learning_rate:
    fail("Final-main LR/r artifact cross-check would fail.")
if final_r.r != 1.0:
    fail(f"Final-main r validation returned {final_r.r!r}")

FINAL_MAIN_OUTPUT = DRIVE / "stage1-training" / TAG / "final-main"
FINAL_MAIN_CACHE = DRIVE / "stage1-cache" / TAG
FINAL_MAIN_MONITOR = DRIVE / "stage1-monitoring" / TAG / "final-main"

print(
    """
--------------------------------------------------------------------------------------------------------------
RESOURCE-BOUNDED R-PHASE1 HANDOFF IS READY

LR artifact:
    {lr_artifact}

r artifact:
    {r_artifact}

Validated final-main constants:
    lr = {lr}
    r  = {r}

Suggested final-main output:
    {final_output}

Suggested final-main cache:
    {final_cache}

Suggested final-main monitor state:
    {final_monitor}

This cell did not start final-main.
--------------------------------------------------------------------------------------------------------------
""".format(
        lr_artifact=LR_PILOT_ARTIFACT,
        r_artifact=R_PHASE1_ARTIFACT,
        lr=final_lr.learning_rate,
        r=final_r.r,
        final_output=FINAL_MAIN_OUTPUT,
        final_cache=FINAL_MAIN_CACHE,
        final_monitor=FINAL_MAIN_MONITOR,
    )
)
