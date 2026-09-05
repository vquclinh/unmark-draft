# ==========================================================================================
# UNMARK - ONE CELL: REISSUE LR-PILOT ARTIFACT WITH AUTHOR OVERRIDE TO LR=1E-4
#
# PURPOSE:
#   Do NOT rerun the 3 LR candidates.
#   Rebuild lr_pilot.json from the completed run-lr*.json files.
#   Record transparently that the author chose lr=0.0001 after W&B curve review,
#   superseding the old single-point min-score winner lr=0.0003.
#
# INPUT CAMPAIGN EVIDENCE:
#   /content/drive/MyDrive/UNMARK/UNMARK-BACKUP/stage1-training/bca24ade2082/lr-pilot
#
# IMPORTANT:
#   This cell must run after pulling a repo commit that contains
#   author_lr_override_after_validation_curve_review support in unmark/stage1/artifact.py.
#   It only regenerates the LR handoff artifact. It does not start r-phase1.
# ==========================================================================================

import datetime
import json
import os
import shlex
import shutil
import statistics
import subprocess
import sys
from pathlib import Path


# ==========================================================================================
# CONFIG
# ==========================================================================================

REPO_URL = "https://github.com/vquclinh/unmark-draft.git"
REPO = Path("/content/UNMARK")

# Use latest pushed main by default. Set UNMARK_REPO_REF to pin a specific
# commit that contains the artifact override support.
REPO_REF = os.environ.get("UNMARK_REPO_REF", "origin/main")

DRIVE = Path("/content/drive/MyDrive/UNMARK/UNMARK-BACKUP")

# This is the historical LR-pilot campaign that produced the completed runs.
OLD_LR_PILOT_HEAD = "bca24ade208265a5a46a54fb2d2d9bd77d8f6703"
OLD_LR_PILOT_TAG = OLD_LR_PILOT_HEAD[:12]

LR_PILOT_DIR = (
    DRIVE
    / "stage1-training"
    / OLD_LR_PILOT_TAG
    / "lr-pilot"
)

LR_PILOT_ARTIFACT = LR_PILOT_DIR / "lr_pilot.json"

REQUIRED_RUN_FILES = (
    "run-lr0.0001.json",
    "run-lr0.0003.json",
    "run-lr0.001.json",
)

AUTHOR = "Linh Vo Quoc"


# ==========================================================================================
# HELPERS
# ==========================================================================================

def fail(msg):
    raise RuntimeError(
        "\n"
        + "=" * 110
        + "\nLR-PILOT AUTHOR OVERRIDE REFUSED\n"
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
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def candidate_for(lr, run_payload):
    recorded_lr = float(run_payload["provenance"]["learning_rate"])
    if recorded_lr != float(lr):
        fail(
            f"{lr} file records learning_rate={recorded_lr}; "
            "the run file naming and provenance disagree."
        )
    return {
        "budget_limited": bool(run_payload["budget_limited"]),
        "label": f"lr={lr:g}",
        "learning_rate": float(lr),
        "r": float(run_payload["provenance"]["r"]),
        "selected": run_payload["selected"],
    }


def score_stats(run_payload, threshold):
    points = [
        p for p in run_payload["evaluations"]
        if int(p["update"]) >= int(threshold)
    ]
    if not points:
        fail(f"No evaluation points at or after update {threshold}.")

    scores = [float(p["score"]) for p in points]
    best = min(
        points,
        key=lambda p: (
            float(p["score"]),
            float(p["d_clean"]),
            int(p["update"]),
        ),
    )
    return {
        "best_score": float(best["score"]),
        "best_update": int(best["update"]),
        "median_score": float(statistics.median(scores)),
        "mean_score": float(statistics.fmean(scores)),
        "sample_stdev_score": (
            float(statistics.stdev(scores)) if len(scores) > 1 else None
        ),
        "points": len(points),
    }


# ==========================================================================================
# 1. MOUNT DRIVE
# ==========================================================================================

banner("1 / 6", "MOUNT DRIVE")

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

banner("2 / 6", "PROCESS SAFETY")

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
# 3. RESTORE REPO WITH OVERRIDE SUPPORT
# ==========================================================================================

banner("3 / 6", "RESTORE REPO WITH OVERRIDE SUPPORT")

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

current_head = capture(["git", "rev-parse", "HEAD"], cwd=REPO)
print("Historical LR-pilot run HEAD:", OLD_LR_PILOT_HEAD)
print("Current override-support HEAD:", current_head)

os.chdir(REPO)
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from unmark.stage1.artifact import (  # noqa: E402
    CampaignIdentity,
    LOCKED_LR_SELECTION_RULE,
    LR_PILOT_AUTHOR_OVERRIDE_KIND,
    validate_selection_artifact,
)
from unmark.stage1.checkpoint import repository_execution_modifications  # noqa: E402
from unmark.stage1.protocol import STAGE1_PROTOCOL_VERSION  # noqa: E402

mods = repository_execution_modifications(REPO)
if mods:
    fail(
        "Execution code is dirty; r-phase1 would refuse this tree:\n"
        + "\n".join(mods)
    )

print("Override kind available:", LR_PILOT_AUTHOR_OVERRIDE_KIND)
print("Repository execution tree is clean.")


# ==========================================================================================
# 4. VERIFY COMPLETED LR-PILOT EVIDENCE
# ==========================================================================================

banner("4 / 6", "VERIFY COMPLETED LR-PILOT EVIDENCE")

if not LR_PILOT_DIR.is_dir():
    fail(f"LR-pilot output directory is missing:\n{LR_PILOT_DIR}")

if not LR_PILOT_ARTIFACT.is_file():
    fail(
        "lr_pilot.json is missing. The previous cell output said LR-pilot completed, "
        f"so inspect this directory:\n{LR_PILOT_DIR}"
    )

for name in REQUIRED_RUN_FILES:
    path = LR_PILOT_DIR / name
    if not path.is_file():
        fail(f"Required completed LR run JSON is missing:\n{path}")

runs = {
    1e-4: load_json(LR_PILOT_DIR / "run-lr0.0001.json"),
    3e-4: load_json(LR_PILOT_DIR / "run-lr0.0003.json"),
    1e-3: load_json(LR_PILOT_DIR / "run-lr0.001.json"),
}

for lr, payload in runs.items():
    if int(payload.get("cap")) != 20000:
        fail(f"lr={lr:g} cap is not 20000: {payload.get('cap')!r}")
    if not payload.get("selected"):
        fail(f"lr={lr:g} records no selected checkpoint.")
    if payload["provenance"].get("repository_head") != OLD_LR_PILOT_HEAD:
        fail(
            f"lr={lr:g} was not produced by the historical LR-pilot HEAD.\n"
            f"Expected: {OLD_LR_PILOT_HEAD}\n"
            f"Actual:   {payload['provenance'].get('repository_head')}"
        )

print("Run JSON files are present and belong to the historical LR-pilot run.")


# ==========================================================================================
# 5. REGENERATE lr_pilot.json
# ==========================================================================================

banner("5 / 6", "REGENERATE LR-PILOT HANDOFF ARTIFACT")

artifact = load_json(LR_PILOT_ARTIFACT)

if not isinstance(artifact.get("identity"), dict):
    fail("Existing lr_pilot.json has no identity block.")

old_artifact_head = artifact.get("repository_head")
old_identity_head = artifact["identity"].get("repository_head")

candidates = [
    candidate_for(1e-4, runs[1e-4]),
    candidate_for(3e-4, runs[3e-4]),
    candidate_for(1e-3, runs[1e-3]),
]

locked = min(
    candidates,
    key=lambda c: (
        float(c["selected"]["score"]),
        float(c["selected"]["d_clean"]),
        float(c["learning_rate"]),
    ),
)
selected = next(c for c in candidates if float(c["learning_rate"]) == 1e-4)

if locked["learning_rate"] != 3e-4:
    fail(f"Expected old locked-rule winner to be lr=3e-4, got {locked!r}")

if selected["learning_rate"] != 1e-4:
    fail("Internal error: author-selected candidate is not lr=1e-4.")

run_0003_scores = {
    int(p["update"]): float(p["score"])
    for p in runs[3e-4]["evaluations"]
}

artifact["stage"] = "lr_pilot"
artifact["protocol_version"] = STAGE1_PROTOCOL_VERSION
artifact["repository_head"] = current_head
artifact["identity"]["repository_head"] = current_head
artifact["candidates"] = candidates
artifact["selected"] = selected

artifact["selection_override"] = {
    "kind": LR_PILOT_AUTHOR_OVERRIDE_KIND,
    "author": AUTHOR,
    "created_at": "2026-09-04",
    "selected_label": selected["label"],
    "selected_learning_rate": selected["learning_rate"],
    "superseded_locked_rule": LOCKED_LR_SELECTION_RULE,
    "superseded_locked_rule_winner": locked,
    "reason": (
        "After reviewing the W&B validation curves, the author selected lr=0.0001 "
        "instead of the old locked-rule winner lr=0.0003. The old rule used the "
        "single lowest validation/score point; lr=0.0003 won only at update 500 "
        "and then rose sharply, while lr=0.0001 was judged more stable across "
        "the later validation trajectory."
    ),
    "evidence": {
        "historical_lr_pilot_repository_head": OLD_LR_PILOT_HEAD,
        "previous_artifact_repository_head": old_artifact_head,
        "previous_artifact_identity_repository_head": old_identity_head,
        "reissued_under_repository_head": current_head,
        "source_files": [
            str(LR_PILOT_DIR / name) for name in REQUIRED_RUN_FILES
        ],
        "source_run_repository_heads": {
            f"lr={lr:g}": runs[lr]["provenance"].get("repository_head")
            for lr in (1e-4, 3e-4, 1e-3)
        },
        "old_rule_selected": {
            "label": locked["label"],
            "learning_rate": locked["learning_rate"],
            "score": locked["selected"]["score"],
            "update": locked["selected"]["update"],
        },
        "author_selected": {
            "label": selected["label"],
            "learning_rate": selected["learning_rate"],
            "score": selected["selected"]["score"],
            "update": selected["selected"]["update"],
        },
        "lr0.0003_early_scores": {
            str(update): run_0003_scores[update]
            for update in (500, 1000, 1500, 2000)
        },
        "robust_score_summary": {
            "threshold_update_2000": {
                f"lr={lr:g}": score_stats(runs[lr], 2000)
                for lr in (1e-4, 3e-4, 1e-3)
            },
            "threshold_update_5000": {
                f"lr={lr:g}": score_stats(runs[lr], 5000)
                for lr in (1e-4, 3e-4, 1e-3)
            },
        },
        "interpretation": (
            "Transparent post-hoc author override based on validation-curve stability; "
            "not a claim that the original min-score selector chose lr=0.0001."
        ),
    },
}

backup = LR_PILOT_ARTIFACT.with_name(
    "lr_pilot.before-author-override."
    + datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
    + ".json"
)
shutil.copy2(LR_PILOT_ARTIFACT, backup)

write_json(LR_PILOT_ARTIFACT, artifact)

print("Backup written:", backup)
print("Regenerated:", LR_PILOT_ARTIFACT)


# ==========================================================================================
# 6. VALIDATE EXACTLY WHAT r-phase1 WILL READ
# ==========================================================================================

banner("6 / 6", "VALIDATE REISSUED ARTIFACT")

reloaded = load_json(LR_PILOT_ARTIFACT)
identity = CampaignIdentity(**reloaded["identity"])
winner = validate_selection_artifact(
    reloaded,
    expected_stage="lr_pilot",
    identity=identity,
    what=str(LR_PILOT_ARTIFACT),
)

if winner.learning_rate != 1e-4:
    fail(f"Validator did not return lr=1e-4; got {winner.learning_rate!r}")

print("Validated winner:")
print("  label :", winner.label)
print("  lr    :", winner.learning_rate)
print("  score :", winner.selected.score)
print("  update:", winner.selected.update)

print("\nOld locked-rule winner:")
print("  label :", locked["label"])
print("  lr    :", locked["learning_rate"])
print("  score :", locked["selected"]["score"])
print("  update:", locked["selected"]["update"])

print(
    """
--------------------------------------------------------------------------------------------------------------
LR-PILOT HANDOFF ARTIFACT IS READY

Use this as --lr-artifact for r-phase1:
    {artifact_path}

Suggested new-code run tag:
    {new_tag}

Suggested r-phase1 output directory:
    {drive}/stage1-training/{new_tag}/r-phase1

Suggested r-phase1 cache root:
    {drive}/stage1-cache/{new_tag}

Suggested r-phase1 monitor state:
    {drive}/stage1-monitoring/{new_tag}/r-phase1

This cell did not start r-phase1.
--------------------------------------------------------------------------------------------------------------
""".format(
        artifact_path=LR_PILOT_ARTIFACT,
        new_tag=current_head[:12],
        drive=DRIVE,
    )
)
