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
import re
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

# ------------------------------------------------------------------------------------------
# THE HUMAN MUST PASTE THE REVIEWED COMMIT SHA HERE BEFORE RUNNING THIS CELL.
#
# This regenerates a scientific handoff artifact that final-main consumes, so the
# code that produces it has to be immutable and named. A branch name is not: a
# push between two runs of this cell would silently change what produced the
# artifact, and the artifact records that head as provenance. Anything other than
# a full 40-hex commit is refused below, BEFORE any artifact is touched.
# ------------------------------------------------------------------------------------------

IMPLEMENTATION_COMMIT = "REPLACE_WITH_FULL_40_HEX_COMMIT_SHA"

# Optional escape hatch for automation; it has NO default and is held to the
# identical 40-hex rule, so it cannot reintroduce a moving ref.
IMPLEMENTATION_COMMIT = os.environ.get(
    "UNMARK_IMPLEMENTATION_COMMIT", IMPLEMENTATION_COMMIT
)

DRIVE = Path("/content/drive/MyDrive/UNMARK/UNMARK-BACKUP")

SOURCE_R_PHASE1_HEAD = "3bb2944e6f71865d5a37fe403b78ea640f8a3f1d"
SOURCE_R_PHASE1_TAG = SOURCE_R_PHASE1_HEAD[:12]
R_PHASE1_DIR = DRIVE / "stage1-training" / SOURCE_R_PHASE1_TAG / "r-phase1"
R_PHASE1_ARTIFACT = R_PHASE1_DIR / "r_phase1.json"
R_PHASE1_MONITOR_DIR = DRIVE / "stage1-monitoring" / SOURCE_R_PHASE1_TAG / "r-phase1"
R_PHASE1_TELEMETRY = R_PHASE1_MONITOR_DIR / "telemetry.jsonl"

AUTHOR = "Linh Vo Quoc"
BACKBONE_MODEL = "vinai/phobert-base"
BACKBONE_REVISION = "01daacda68afe13d83023d16ec647239e344a1e6"
PROTOCOL_VERSION = "stage1-protocol-v1"

# ------------------------------------------------------------------------------------------
# INDEPENDENT SCIENTIFIC IDENTITY
#
# These are the expected values of the runtime the artifacts must describe. They
# are checked against the real prepared corpus and the real pinned inventory, and
# the campaign identity is built FROM that verified evidence. The LR artifact is
# then required to agree with it. The direction matters: an identity read out of
# lr_pilot.json and then used to validate lr_pilot.json proves nothing.
# ------------------------------------------------------------------------------------------

CORPUS_KEY = "aa49785eadcb"
PREPARED_CORPUS_DIR = DRIVE / "stage1-prepared" / CORPUS_KEY
CORPUS_COMPLETION_DIR = DRIVE / "stage1-checkpoints" / CORPUS_KEY
LOCAL_PREPARED_DIR = Path("/content/stage1-prepared") / CORPUS_KEY

EXPECTED_CHUNKS_BYTES = 2198412593
EXPECTED_CHUNKS_SHA256 = (
    "5e4c5e0c77e7677e188501723651e0923d072a31a9048a7d04042ff7b290cad6"
)
EXPECTED_MANIFEST_BYTES = 2878
EXPECTED_MANIFEST_SHA256 = (
    "6f33c2aa51b63a4dc68e238594acbec581b2a1f6b0f7be42e002dfb10a02ef62"
)
EXPECTED_MEMBERSHIP_DIGEST = (
    "250859a57d745675c5dba2c7a35df08ccc123988bece873b0c9b29c6e78413d6"
)

EXPECTED_INVENTORY_REVISION = "135a4d9716e49a981624474156d6f247b9b46f6a"
EXPECTED_INVENTORY_SHA256 = (
    "78eeb840d50455b14bd564da5aed7318d96468b8deaad5986b77bf5c538315d2"
)
EXPECTED_INVENTORY_SIZE = 116290
EXPECTED_INVENTORY_PATH = (
    REPO / ".resources-cache" / "vietnamese-syllables" / "all-vietnamese-syllables.txt"
)

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


def require_immutable_commit(value):
    """Return `value` only if it is a full 40-hex commit SHA.

    Rejects every moving target -- ``origin/main``, ``main``, ``master``,
    ``HEAD``, tags, arbitrary refs, short SHAs, the unreplaced placeholder and
    the empty string -- because each of them can resolve to different code on
    two different days while the artifact claims a single provenance.
    """
    text = "" if value is None else str(value).strip()
    if not text or text == "REPLACE_WITH_FULL_40_HEX_COMMIT_SHA":
        fail(
            "No implementation commit was supplied.\n\n"
            "Edit this cell and replace the placeholder with the full 40-hex SHA of the\n"
            "reviewed commit (or export UNMARK_IMPLEMENTATION_COMMIT):\n\n"
            '    IMPLEMENTATION_COMMIT = "REPLACE_WITH_FULL_40_HEX_COMMIT_SHA"\n\n'
            "Nothing has been read or written yet."
        )
    if not re.fullmatch(r"[0-9a-fA-F]{40}", text):
        fail(
            f"IMPLEMENTATION_COMMIT must match ^[0-9a-fA-F]{{40}}$, got {text!r}.\n\n"
            "Branch names, tags, HEAD and short SHAs are refused: this cell writes a\n"
            "scientific handoff artifact that records the producing commit as provenance,\n"
            "so that commit has to be immutable.\n\n"
            "Nothing has been read or written yet."
        )
    return text.lower()


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
# 1. REQUIRE AN IMMUTABLE IMPLEMENTATION COMMIT
#
# First, before Drive is mounted and before anything is read or written.
# ==========================================================================================

banner("1 / 11", "REQUIRE IMMUTABLE IMPLEMENTATION COMMIT")

IMPLEMENTATION_COMMIT = require_immutable_commit(IMPLEMENTATION_COMMIT)
print("Requested implementation commit:", IMPLEMENTATION_COMMIT)


# ==========================================================================================
# 2. MOUNT DRIVE
# ==========================================================================================

banner("2 / 11", "MOUNT DRIVE")

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

banner("3 / 11", "PROCESS SAFETY")

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
# 4. RESTORE REPO WITH AMENDMENT SUPPORT
# ==========================================================================================

banner("4 / 11", "RESTORE REPO WITH AMENDMENT SUPPORT")

if not REPO.exists():
    run(["git", "clone", "--quiet", REPO_URL, REPO])

if not (REPO / ".git").is_dir():
    fail(f"Not a Git repository: {REPO}")

require_clean_repo_checkout(REPO)

# Fetch the object by SHA, not by branch, so no branch name participates.
run(["git", "fetch", "--quiet", "origin", IMPLEMENTATION_COMMIT], cwd=REPO)
run(["git", "checkout", "--quiet", "--detach", IMPLEMENTATION_COMMIT], cwd=REPO)

require_clean_repo_checkout(REPO)

status = capture(["git", "status", "--porcelain"], cwd=REPO)
if status:
    fail("Repository is dirty after checkout:\n" + status)

HEAD = capture(["git", "rev-parse", "HEAD"], cwd=REPO)
if HEAD != IMPLEMENTATION_COMMIT:
    fail(
        "Checkout did not land on the requested immutable commit.\n"
        f"requested: {IMPLEMENTATION_COMMIT}\n"
        f"HEAD     : {HEAD}"
    )
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
# 5. FETCH AND VERIFY PINNED INVENTORY
# ==========================================================================================

banner("5 / 11", "FETCH AND VERIFY PINNED INVENTORY")

run([SCI_PY, "-m", "pip", "install", "-q", "-r", REPO / "requirements" / "experiment.txt"])
run([SCI_PY, REPO / "scripts" / "fetch_vietnamese_syllable_inventory.py"], cwd=REPO)

from unmark.stage1.preflight import verify_scientific_inputs  # noqa: E402

scientific_inputs = verify_scientific_inputs(repo_root=REPO)
inventory = scientific_inputs.inventory
print("Pinned inventory verified:")
print("  source_name    :", inventory.source_name)
print("  source_revision:", inventory.source_revision)
print("  sha256         :", inventory.sha256)
print("  size_bytes     :", inventory.size_bytes)

# `verify_scientific_inputs` proves the file matches ITS pinned constant. These
# check that constant is the one Audit 047 expects, so a repository that repinned
# the inventory cannot quietly reissue this handoff.
if inventory.source_revision != EXPECTED_INVENTORY_REVISION:
    fail(
        "Pinned inventory revision mismatch.\n"
        f"expected: {EXPECTED_INVENTORY_REVISION}\n"
        f"runtime : {inventory.source_revision}"
    )
if inventory.sha256 != EXPECTED_INVENTORY_SHA256:
    fail(
        "Pinned inventory sha256 mismatch.\n"
        f"expected: {EXPECTED_INVENTORY_SHA256}\n"
        f"runtime : {inventory.sha256}"
    )
if inventory.size_bytes is not None and inventory.size_bytes != EXPECTED_INVENTORY_SIZE:
    fail(
        "Pinned inventory size mismatch.\n"
        f"expected: {EXPECTED_INVENTORY_SIZE}\n"
        f"runtime : {inventory.size_bytes}"
    )
if not EXPECTED_INVENTORY_PATH.is_file():
    fail(f"Reconstructed inventory is not at the expected path:\n{EXPECTED_INVENTORY_PATH}")
print("Inventory matches the Audit 047 pins.")


# ==========================================================================================
# 6. ESTABLISH SCIENTIFIC IDENTITY INDEPENDENTLY OF THE ARTIFACTS
#
# Built from the real prepared corpus and the real inventory, never from
# lr_pilot.json. The artifacts are then required to agree with it.
# ==========================================================================================

banner("6 / 11", "INDEPENDENT SCIENTIFIC IDENTITY")

from unmark.stage1.checkpoint import (  # noqa: E402
    concatenate_shards,
    verify_prepared_corpus,
)
from unmark.stage1.artifact import CampaignIdentity  # noqa: E402

if not CORPUS_COMPLETION_DIR.is_dir():
    fail(f"Prepared-corpus completion directory is missing:\n{CORPUS_COMPLETION_DIR}")

# `verify_prepared_corpus` binds artifacts by RELATIVE name, so the payload may be
# served from local SSD while COMPLETE.json stays on Drive. Colab deletion wipes
# /content, so rebuild the payload from the immutable shards when it is absent.
# This writes only to /content and never to Drive.
corpus_dir = PREPARED_CORPUS_DIR
if not (PREPARED_CORPUS_DIR / "chunks.jsonl").is_file():
    print("chunks.jsonl absent from Drive; rebuilding from immutable shards to local SSD.")
    state = load_json(CORPUS_COMPLETION_DIR / "state.json")
    shard_paths = [
        CORPUS_COMPLETION_DIR / "shards" / shard["name"] for shard in state["shards"]
    ]
    missing_shards = [str(p) for p in shard_paths if not p.is_file()]
    if missing_shards:
        fail(
            f"{len(missing_shards)} prepared-corpus shard(s) missing, e.g.\n  "
            + "\n  ".join(missing_shards[:5])
        )
    LOCAL_PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PREPARED_CORPUS_DIR / "manifest.json", LOCAL_PREPARED_DIR / "manifest.json")
    rebuilt_bytes, rebuilt_sha = concatenate_shards(
        shard_paths, LOCAL_PREPARED_DIR / "chunks.jsonl"
    )
    print(f"Rebuilt chunks.jsonl: {rebuilt_bytes:,} bytes, sha256={rebuilt_sha}")
    corpus_dir = LOCAL_PREPARED_DIR

verified_corpus = verify_prepared_corpus(corpus_dir, CORPUS_COMPLETION_DIR)

corpus_artifacts = dict(verified_corpus.artifacts)
for name, (expected_bytes, expected_sha) in {
    "chunks.jsonl": (EXPECTED_CHUNKS_BYTES, EXPECTED_CHUNKS_SHA256),
    "manifest.json": (EXPECTED_MANIFEST_BYTES, EXPECTED_MANIFEST_SHA256),
}.items():
    if name not in corpus_artifacts:
        fail(f"Verified corpus does not bind {name!r}.")
    got_bytes, got_sha = corpus_artifacts[name]
    if got_bytes != expected_bytes or got_sha != expected_sha:
        fail(
            f"Prepared corpus {name} does not match the Audit 047 pins.\n"
            f"expected: {expected_bytes} bytes, sha256={expected_sha}\n"
            f"runtime : {got_bytes} bytes, sha256={got_sha}"
        )

if verified_corpus.chunk_membership_digest != EXPECTED_MEMBERSHIP_DIGEST:
    fail(
        "Prepared corpus membership digest mismatch.\n"
        f"expected: {EXPECTED_MEMBERSHIP_DIGEST}\n"
        f"runtime : {verified_corpus.chunk_membership_digest}"
    )

print("Prepared corpus verified independently:")
print("  key              :", CORPUS_KEY)
print("  payload          :", corpus_dir / "chunks.jsonl")
print("  membership digest:", verified_corpus.chunk_membership_digest)

# The campaign identity every artifact must describe, derived from evidence.
RUNTIME_IDENTITY = CampaignIdentity.from_inputs(
    repository_head=HEAD,
    corpus_manifest_digest=verified_corpus.chunk_membership_digest,
    encoder_revision=BACKBONE_REVISION,
    inventory=inventory,
)
if RUNTIME_IDENTITY.encoder_checkpoint != BACKBONE_MODEL:
    fail(
        "Backbone model identity mismatch.\n"
        f"expected: {BACKBONE_MODEL}\n"
        f"runtime : {RUNTIME_IDENTITY.encoder_checkpoint}"
    )
if RUNTIME_IDENTITY.encoder_revision != BACKBONE_REVISION:
    fail(
        "Backbone revision mismatch.\n"
        f"expected: {BACKBONE_REVISION}\n"
        f"runtime : {RUNTIME_IDENTITY.encoder_revision}"
    )
if RUNTIME_IDENTITY.protocol_version != PROTOCOL_VERSION:
    fail(
        "Protocol identity mismatch.\n"
        f"expected: {PROTOCOL_VERSION}\n"
        f"runtime : {RUNTIME_IDENTITY.protocol_version}"
    )

print("Independently established campaign identity:")
for key, value in RUNTIME_IDENTITY.to_dict().items():
    print(f"  {key}: {value}")


# ==========================================================================================
# 7. REISSUE LR-PILOT UNDER THIS SAME HEAD
# ==========================================================================================

banner("7 / 11", "REISSUE LR-PILOT HANDOFF")

lr_helper = REPO / "docs" / "colab" / "regenerate_lr_pilot_author_override_cell.py"
if not lr_helper.is_file():
    fail(f"Missing LR helper cell:\n{lr_helper}")

# Hand the LR helper the commit this cell already pinned and verified, so it
# cannot check the repository back out onto a moving ref mid-run.
helper_globals = {
    "__name__": "__unmark_lr_reissue_helper__",
    "INJECTED_IMPLEMENTATION_COMMIT": IMPLEMENTATION_COMMIT,
}
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
# 8. IMPORT AMENDMENT SUPPORT
# ==========================================================================================

banner("8 / 11", "IMPORT AMENDMENT SUPPORT")

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
# 9. VERIFY INPUT HANDOFFS, STOPPED CHECKPOINTS, AND TELEMETRY
# ==========================================================================================

banner("9 / 11", "VERIFY SOURCE EVIDENCE")

if not R_PHASE1_DIR.is_dir():
    fail(f"r-phase1 source directory is missing:\n{R_PHASE1_DIR}")

lr_artifact = load_json(LR_PILOT_ARTIFACT)

# The identity comes from step 6 -- real corpus, real inventory, verified HEAD --
# and the artifact is required to agree with it. Reading the identity out of
# lr_pilot.json and then validating lr_pilot.json against it would only prove the
# file is self-consistent.
identity = RUNTIME_IDENTITY
recorded_identity = lr_artifact.get("identity")
if recorded_identity != identity.to_dict():
    fail(
        "lr_pilot.json does not describe the independently established runtime.\n"
        f"independent: {json.dumps(identity.to_dict(), indent=2, sort_keys=True)}\n"
        f"artifact   : {json.dumps(recorded_identity, indent=2, sort_keys=True)}"
    )
print("lr_pilot.json identity agrees with the independently established identity.")

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
# 10. REBUILD AND VERIFY R-PHASE1 HANDOFF
# ==========================================================================================

banner("10 / 11", "REBUILD RESOURCE-BOUNDED R-PHASE1 ARTIFACT")

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
# 11. VALIDATE EXACTLY WHAT FINAL-MAIN WILL CONSUME
# ==========================================================================================

banner("11 / 11", "VALIDATE FINAL-MAIN HANDOFFS")

reloaded_lr = load_json(LR_PILOT_ARTIFACT)
reloaded_r = load_json(R_PHASE1_ARTIFACT)

# Still the independently established identity, not one read back out of the
# files being validated.
final_identity = RUNTIME_IDENTITY
if reloaded_lr["identity"] != final_identity.to_dict():
    fail("Reloaded lr_pilot.json no longer matches the independently established identity.")
if reloaded_r["identity"] != final_identity.to_dict():
    fail("Reloaded r_phase1.json does not match the independently established identity.")
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
