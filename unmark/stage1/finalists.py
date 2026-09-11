"""Stage-1 finalist freeze: exactly two frozen adapters, adjudication still OPEN.

Audit 048. Stage-1 training, LR selection, `r` selection and **candidate
generation** are all CLOSED. What this module represents is the one question
still open afterwards:

    which of exactly TWO already-trained frozen checkpoints becomes the final
    downstream adapter?

**Why a module and not a paragraph.** The provisional `canonical-v1` package was
built from `final_main` seed 36930 at update 3500 -- the winner of the Stage-1
local-stability rule written before ranking. Before any downstream experiment was
run, the authors reopened *only* the final adapter adjudication, out of a
training-maturity concern about promoting update 3500 of 20000. Reopening a
closed decision is exactly the moment a candidate set silently grows, so the set
is pinned here in code and the pins are tested. A third checkpoint that "looks
good" at 17000 cannot be added later without changing this file under review.

**What is NOT reopened.** Stage-1 training, the LR, the `r`, the corpus, the
protocol and the checkpoint *search* are closed. `canonical-v1` is preserved as
history; it is superseded as the FINAL downstream selection, not deleted.

**Evidence direction.** A finalist's `checkpoint_sha256` is authoritative
evidence about a file that lives outside this repository (`*.pt` is
git-ignored). It is therefore either a full 64-hex digest obtained from the real
artifact, or the explicit :data:`SHA256_PENDING` sentinel. A truncated or
prefix-expanded digest is refused: a hash that is *nearly* right is not evidence,
and silently accepting 16 hex characters would let a different file pass.
:func:`freeze_is_complete` stays false while any digest is pending, so the
freeze cannot be reported as finished on partial evidence.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from unmark.stage1.checkpoint import sha256_file
from unmark.stage1.contracts import Stage1ContractViolation
from unmark.stage1.protocol import (
    ADAPTER_TRAINABLE_PARAMETERS,
    CORRUPTION_SEED,
    ENCODER_CHECKPOINT,
    ENCODER_REVISION,
    PRECISION,
    STAGE1_PROTOCOL_VERSION,
    adapter_init_seed,
)
from unmark.stage1.trainer import (
    CHECKPOINT_SCHEMA_VERSION,
    RunProvenance,
    TrainerContractViolation,
    verify_checkpoint,
)


class FinalistFreezeViolation(Stage1ContractViolation):
    """Raised when the finalist freeze is edited, extended or short-circuited."""


FINALIST_FREEZE_SCHEMA_VERSION = "stage1-finalist-freeze-v1"

SHA256_PENDING = "PENDING_AUTHORITATIVE_EVIDENCE"
"""Explicit marker for a digest that has not yet been taken from the real file.

Deliberately not `null` and not a prefix. `null` reads as "no such field" and a
prefix reads as "we know the answer", and neither is true. This value is a
sentence: the evidence exists somewhere durable, it has not been bound here yet,
and nothing downstream may proceed as though it had.
"""

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FULL_SHA = re.compile(r"^[0-9a-f]{40}$")

ADAPTER_TENSOR_COUNT = 8
"""`tone_embedding`, `letter_embedding`, `fusion.{weight,bias}`,
`gate.{weight,bias}`, `layer_norm.{weight,bias}`."""

ADAPTER_STATE_KEYS: tuple[str, ...] = (
    "fusion.bias",
    "fusion.weight",
    "gate.bias",
    "gate.weight",
    "layer_norm.bias",
    "layer_norm.weight",
    "letter_embedding.weight",
    "tone_embedding.weight",
)
"""Sorted `OrthographyInputAdapter.state_dict()` keys. Checked as a SET so a
renamed or dropped tensor fails closed rather than being counted as "8 things"."""

STAGE1_TRAINING_STATUS = "CLOSED"
STAGE1_CANDIDATE_GENERATION_STATUS = "CLOSED"
ADJUDICATION_OPEN = "OPEN"
FINALIST_COUNT = 2

ADJUDICATION_ALLOWED_EVIDENCE = "downstream_dev_only_under_a_separately_frozen_protocol"
"""The only admissible basis for choosing between A and B.

Recorded as a string in the artifact so the *absence* of the future Stage-2
protocol is visible rather than implied. TEST is sealed; a downstream TEST score
may never adjudicate A vs B, and no downstream result may reopen Stage-1.
"""

FINALIST_ROLES: tuple[str, ...] = (
    "stage1_stability_rule_winner",
    "mature_historical_lr_pilot_alternative",
)

CORPUS_MANIFEST_DIGEST = (
    "250859a57d745675c5dba2c7a35df08ccc123988bece873b0c9b29c6e78413d6"
)
"""The verified chunk-membership digest of the one prepared corpus.

`RunProvenance.require_match` treats `corpus_manifest_digest` as scientific
identity, so a finalist trained on different data cannot be waved through on the
strength of matching seeds. Cross-checked against
`docs/spec/stage1-final-freeze.json` (`data.chunk_membership_digest`) by the
tests, so the two committed specs cannot drift apart.
"""

INVENTORY_SHA256 = (
    "78eeb840d50455b14bd564da5aed7318d96468b8deaad5986b77bf5c538315d2"
)
INVENTORY_SOURCE_REVISION = "135a4d9716e49a981624474156d6f247b9b46f6a"
INVENTORY_SIZE_BYTES = 116290
"""The pinned Vietnamese syllable inventory (D-B3A-001 / D-S1A-008).

Two runs that differ only in which inventory resolved eligibility are different
experiments: the denominator of every corruption rate changes. `sha256` pins the
raw bytes, `source_revision` pins the upstream object, and `size_bytes` is part
of the locked shape, so these three are checked explicitly against the
checkpoint's own recorded inventory in addition to the whole-block comparison
`require_match` performs.
"""

VERIFIED_PROVENANCE_FIELDS: tuple[str, ...] = (
    "run_seed",
    "init_seed",
    "corruption_seed",
    "learning_rate",
    "r",
    "corpus_manifest_digest",
    "backbone_checkpoint",
    "backbone_revision",
    "protocol_version",
    "precision",
    "repository_head",
    "inventory",
    "objective",
    "fusion",
    "lambda_align",
    "lambda_clean",
)
"""Exactly what `RunProvenance.require_match` compares, in its own order.

`objective` joined the contract when the first post-hoc V2 candidate (V2-GC,
`grid-consistency-v1`) was implemented: which loss a run minimised is scientific
identity, so the finalist gate verifies it like every other field. `fusion`
joined with C1 (V2-SCF, `scale-calibrated-fusion-v1`), which trains the
HISTORICAL objective under a different adapter architecture -- so the objective
alone no longer identifies a checkpoint, and without this field a C1 adapter
would verify as UNMARK-A and load into the wrong mixture rule.

Both finalists are historical, and a historical provenance that predates either
field is read as the historical objective and the historical fusion, so neither A
nor B had to be re-verified or re-hashed -- the gate got strictly stronger, not
different.

Recorded so a test can assert this verifier is **not weaker** than the contract a
resume must satisfy. If `require_match` ever gains a field and this tuple does
not, the test fails rather than the finalist gate silently falling behind.
"""

BINDING_FIELDS: tuple[str, ...] = (
    "unmark/stage1/finalists.py :: FINALIST_B.checkpoint_sha256",
    "docs/spec/stage1-adapter-finalists.json :: adjudication.finalists[key=B].checkpoint_sha256",
    "docs/spec/stage1-adapter-finalists.json :: evidence.pending_finalist_digests",
    "docs/spec/stage1-adapter-finalists.json :: evidence.freeze_complete",
)
"""Every place finalist B's digest is represented.

Binding it is a four-field edit, not one. Leaving any of them behind produces an
artifact that is internally inconsistent -- a bound digest still declared
pending, or a freeze declaring itself complete while a sentinel remains -- and
`validate_freeze_payload` refuses all of those combinations. Enumerated here so
the operator has the checklist in the code rather than only in the audit.
"""

FROZEN_LEARNING_RATE = 1e-4
FROZEN_R = 1.0
"""Closed by D-S1B-020 (author LR override) and D-S1B-022 (resource-bounded r).

Repeated here as a *pin*, not as a new decision: both finalists must sit at the
adopted hyperparameters, so a checkpoint from some other LR or `r` cannot enter
the finalist set through this file.
"""


@dataclass(frozen=True)
class FinalistIdentity:
    """One frozen finalist. Every field is evidence about an external file."""

    key: str
    role: str
    source_stage: str
    run_seed: int
    update: int
    learning_rate: float
    r: float
    validation_score: float
    source_repository_head: str
    checkpoint_sha256: str
    robust_score: float | None = None
    """Stage-1 local-stability score. Present only where it was computed: the
    ranking was run over checkpoints that physically exist, and only finalist A
    was ranked as the winner of it. `None` is honest absence, never zero."""

    @property
    def sha256_bound(self) -> bool:
        return self.checkpoint_sha256 != SHA256_PENDING

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "key": self.key,
            "role": self.role,
            "source_stage": self.source_stage,
            "run_seed": self.run_seed,
            "update": self.update,
            "learning_rate": self.learning_rate,
            "r": self.r,
            "validation_score": self.validation_score,
            "source_repository_head": self.source_repository_head,
            "checkpoint_sha256": self.checkpoint_sha256,
        }
        if self.robust_score is not None:
            payload["robust_score"] = self.robust_score
        return payload


FINALIST_A = FinalistIdentity(
    key="A",
    role="stage1_stability_rule_winner",
    source_stage="final_main",
    run_seed=36930,
    update=3500,
    learning_rate=FROZEN_LEARNING_RATE,
    r=FROZEN_R,
    validation_score=0.0845640671895974,
    robust_score=0.10167897852382013,
    source_repository_head="7773c77b1df92a6e685dac13c49765ce974f84d8",
    checkpoint_sha256=(
        "6773fbb59c7381ba8ddaa944302124a124f5b8a5cb0a5dbb1a5063f3db4a2a91"
    ),
)
"""The provisional `canonical-v1` source. Preserved as history, superseded as
the FINAL selection."""

FINALIST_B = FinalistIdentity(
    key="B",
    role="mature_historical_lr_pilot_alternative",
    source_stage="lr_pilot",
    run_seed=21230,
    update=14500,
    learning_rate=FROZEN_LEARNING_RATE,
    r=FROZEN_R,
    validation_score=0.09000698585438581,
    robust_score=0.106576028028,
    source_repository_head="bca24ade208265a5a46a54fb2d2d9bd77d8f6703",
    checkpoint_sha256=(
        "9405bd76c04939641170cb71507ce8eb669eb2987016b86b495a403ceafcb9d2"
    ),
)
"""Update 14500 of 20000 in the historical LR pilot.

`checkpoint_sha256` was `SHA256_PENDING` until the authoritative Colab
verification under implementation commit
`054c6d8fa4c912f10a2bb4c21e272e5064e8f355` hashed the real checkpoint and
verified its full identity (Audit 048 s10.2). It is bound here from that
evidence, never from the previously observed 16-character prefix.
"""

FINALISTS: tuple[FinalistIdentity, ...] = (FINALIST_A, FINALIST_B)


def freeze_is_complete(finalists: Sequence[FinalistIdentity] = FINALISTS) -> bool:
    """True only when every finalist's digest is bound to real evidence."""
    return all(f.sha256_bound for f in finalists)


def pending_finalists(
    finalists: Sequence[FinalistIdentity] = FINALISTS,
) -> tuple[str, ...]:
    """Keys whose digest is still unbound. Empty means the freeze is complete."""
    return tuple(f.key for f in finalists if not f.sha256_bound)


def require_sha256(value: Any, what: str, *, allow_pending: bool = True) -> str:
    """A full lowercase 64-hex digest, or the explicit pending sentinel."""
    if not isinstance(value, str):
        raise FinalistFreezeViolation(f"{what} must be a string, got {type(value).__name__}")
    if value == SHA256_PENDING:
        if not allow_pending:
            raise FinalistFreezeViolation(
                f"{what} is still {SHA256_PENDING}; this operation requires the "
                "authoritative digest to have been bound first"
            )
        return value
    if not _SHA256.match(value):
        raise FinalistFreezeViolation(
            f"{what} must be 64 lowercase hex characters or {SHA256_PENDING!r}, got "
            f"{value!r} ({len(value)} characters). A truncated or prefix-expanded "
            "digest is not evidence."
        )
    return value


def require_repository_head(value: Any, what: str) -> str:
    if not isinstance(value, str) or not _FULL_SHA.match(value):
        raise FinalistFreezeViolation(
            f"{what} must be a full 40-character lowercase commit sha, got {value!r}"
        )
    return value


def validate_finalists(finalists: Sequence[FinalistIdentity]) -> None:
    """Fail closed unless this is exactly the frozen two-candidate set.

    The invariant that matters is not "some finalists are present" but that the
    set has not drifted: not grown, not shrunk, not duplicated, not re-pointed at
    a different update of the same run.
    """
    if len(finalists) != FINALIST_COUNT:
        raise FinalistFreezeViolation(
            f"the Stage-1 finalist set is exactly {FINALIST_COUNT} candidates, got "
            f"{len(finalists)}. Candidate generation is CLOSED: a further checkpoint "
            "cannot be added because it looks attractive, and a finalist cannot be "
            "dropped without a reviewed amendment."
        )
    keys = [f.key for f in finalists]
    if sorted(keys) != ["A", "B"]:
        raise FinalistFreezeViolation(f"finalist keys must be exactly A and B, got {keys}")
    roles = [f.role for f in finalists]
    if sorted(roles) != sorted(FINALIST_ROLES):
        raise FinalistFreezeViolation(
            f"finalist roles must be exactly {sorted(FINALIST_ROLES)}, got {sorted(roles)}"
        )
    coordinates = [(f.source_stage, f.run_seed, f.update) for f in finalists]
    if len(set(coordinates)) != len(coordinates):
        raise FinalistFreezeViolation(
            f"two finalists name the same (stage, seed, update): {coordinates}"
        )
    digests = [f.checkpoint_sha256 for f in finalists if f.sha256_bound]
    if len(set(digests)) != len(digests):
        raise FinalistFreezeViolation(
            "two finalists carry the same checkpoint sha256; they would be one file"
        )
    for finalist in finalists:
        expected = {"A": FINALIST_A, "B": FINALIST_B}[finalist.key]
        for field in (
            "role", "source_stage", "run_seed", "update",
            "learning_rate", "r", "validation_score", "source_repository_head",
        ):
            got, want = getattr(finalist, field), getattr(expected, field)
            if got != want:
                raise FinalistFreezeViolation(
                    f"finalist {finalist.key} {field} is {got!r}, not the frozen "
                    f"{want!r}. The finalist set is pinned by Audit 048."
                )
        require_sha256(finalist.checkpoint_sha256, f"finalist {finalist.key} sha256")
        # The digest is the one pinned field allowed to change, and when it does
        # it must change EVERYWHERE. Binding it in the artifact while the code
        # still says PENDING (or the reverse) leaves the freeze describing two
        # different states of the same evidence.
        if finalist.checkpoint_sha256 != expected.checkpoint_sha256:
            raise FinalistFreezeViolation(
                f"finalist {finalist.key} checkpoint_sha256 is "
                f"{finalist.checkpoint_sha256!r}, but unmark/stage1/finalists.py holds "
                f"{expected.checkpoint_sha256!r}. Binding a digest is a "
                f"{len(BINDING_FIELDS)}-field edit: {list(BINDING_FIELDS)}"
            )
        require_repository_head(
            finalist.source_repository_head, f"finalist {finalist.key} repository head"
        )
        if finalist.learning_rate != FROZEN_LEARNING_RATE:
            raise FinalistFreezeViolation(
                f"finalist {finalist.key} LR {finalist.learning_rate!r} is not the "
                f"closed {FROZEN_LEARNING_RATE!r}"
            )
        if finalist.r != FROZEN_R:
            raise FinalistFreezeViolation(
                f"finalist {finalist.key} r {finalist.r!r} is not the closed {FROZEN_R!r}"
            )
        if finalist.update <= 0 or finalist.update % 500 != 0:
            raise FinalistFreezeViolation(
                f"finalist {finalist.key} update {finalist.update} is not a validation "
                "cadence point"
            )


def finalist_for(key: str, finalists: Sequence[FinalistIdentity] = FINALISTS) -> FinalistIdentity:
    for finalist in finalists:
        if finalist.key == key:
            return finalist
    raise FinalistFreezeViolation(
        f"unknown finalist {key!r}; the frozen set is {[f.key for f in finalists]}"
    )


# ---------------------------------------------------------------------------
# The committed artifact
# ---------------------------------------------------------------------------
FREEZE_PATH = Path(__file__).resolve().parents[2] / "docs/spec/stage1-adapter-finalists.json"


def load_freeze(path: Path | str = FREEZE_PATH) -> dict[str, Any]:
    """Read and fully validate the committed finalist freeze artifact."""
    path = Path(path)
    if not path.is_file():
        raise FinalistFreezeViolation(f"finalist freeze artifact is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise FinalistFreezeViolation(f"{path} is malformed: {error}") from error
    validate_freeze_payload(payload, what=str(path))
    return payload


def _value(payload: Mapping[str, Any], section: str, field: str, what: str) -> Any:
    block = payload.get(section)
    if not isinstance(block, Mapping):
        raise FinalistFreezeViolation(f"{what} section {section!r} is missing")
    entry = block.get(field)
    if not isinstance(entry, Mapping) or "value" not in entry:
        raise FinalistFreezeViolation(
            f"{what} {section}.{field} must be an object carrying a 'value'"
        )
    if "classification" not in entry:
        raise FinalistFreezeViolation(f"{what} {section}.{field} carries no classification")
    return entry["value"]


def validate_freeze_payload(payload: Any, *, what: str = "finalist freeze") -> None:
    """Fail closed unless the artifact states exactly the Audit 048 boundary."""
    if not isinstance(payload, Mapping):
        raise FinalistFreezeViolation(f"{what} is not a JSON object")
    if payload.get("schema_version") != FINALIST_FREEZE_SCHEMA_VERSION:
        raise FinalistFreezeViolation(
            f"{what} schema {payload.get('schema_version')!r} != "
            f"{FINALIST_FREEZE_SCHEMA_VERSION!r}"
        )

    if _value(payload, "stage1", "training", what) != STAGE1_TRAINING_STATUS:
        raise FinalistFreezeViolation(f"{what} does not record Stage-1 training as CLOSED")
    if _value(payload, "stage1", "candidate_generation", what) != STAGE1_CANDIDATE_GENERATION_STATUS:
        raise FinalistFreezeViolation(f"{what} does not record candidate generation as CLOSED")
    for closed in ("learning_rate_selection", "r_selection"):
        if _value(payload, "stage1", closed, what) != "CLOSED":
            raise FinalistFreezeViolation(f"{what} does not record {closed} as CLOSED")

    if _value(payload, "model", "backbone", what) != ENCODER_CHECKPOINT:
        raise FinalistFreezeViolation(f"{what} backbone is not {ENCODER_CHECKPOINT!r}")
    if _value(payload, "model", "revision", what) != ENCODER_REVISION:
        raise FinalistFreezeViolation(f"{what} backbone revision is not the locked one")
    if _value(payload, "model", "protocol_version", what) != STAGE1_PROTOCOL_VERSION:
        raise FinalistFreezeViolation(f"{what} protocol version is not the locked one")
    if _value(payload, "model", "adapter_tensor_count", what) != ADAPTER_TENSOR_COUNT:
        raise FinalistFreezeViolation(f"{what} adapter tensor count is not {ADAPTER_TENSOR_COUNT}")
    if _value(payload, "model", "adapter_trainable_parameters", what) != ADAPTER_TRAINABLE_PARAMETERS:
        raise FinalistFreezeViolation(
            f"{what} adapter parameter count is not {ADAPTER_TRAINABLE_PARAMETERS}"
        )
    if _value(payload, "model", "adapter_dtype", what) != PRECISION:
        raise FinalistFreezeViolation(f"{what} adapter dtype is not {PRECISION!r}")
    if _value(payload, "model", "checkpoint_schema_version", what) != CHECKPOINT_SCHEMA_VERSION:
        raise FinalistFreezeViolation(f"{what} checkpoint schema is not the locked one")

    raw = _value(payload, "adjudication", "finalists", what)
    if not isinstance(raw, list):
        raise FinalistFreezeViolation(f"{what} finalists must be a list")
    finalists = [_finalist_from_dict(entry, what=what) for entry in raw]
    validate_finalists(finalists)

    if _value(payload, "adjudication", "status", what) != ADJUDICATION_OPEN:
        raise FinalistFreezeViolation(f"{what} adjudication status must be {ADJUDICATION_OPEN!r}")
    if _value(payload, "adjudication", "final_adapter_selected", what) is not False:
        raise FinalistFreezeViolation(
            f"{what} records a selected final adapter. Audit 048 freezes the candidate "
            "set only; the winner is chosen later under a separately frozen DEV-only "
            "protocol that does not exist yet."
        )
    winner = _value(payload, "adjudication", "winner", what)
    if winner is not None:
        raise FinalistFreezeViolation(
            f"{what} records winner {winner!r} with no adjudication protocol or evidence "
            "contract in force. A winner cannot be recorded before the Stage-2 "
            "adjudication protocol is frozen and reviewed."
        )
    if _value(payload, "adjudication", "candidate_universe_extensible", what) is not False:
        raise FinalistFreezeViolation(f"{what} must record the candidate universe as closed")
    if _value(payload, "adjudication", "allowed_evidence", what) != ADJUDICATION_ALLOWED_EVIDENCE:
        raise FinalistFreezeViolation(f"{what} allowed adjudication evidence is not the frozen one")
    if _value(payload, "adjudication", "downstream_results_seen", what) is not False:
        raise FinalistFreezeViolation(
            f"{what} records that downstream results were seen. The whole basis of this "
            "amendment is that it was made BEFORE any downstream result existed."
        )

    if _value(payload, "test_sealing", "official_uit_vsfc_test", what) != "SEALED":
        raise FinalistFreezeViolation(f"{what} does not record the official TEST as SEALED")
    for forbidden in ("official_test_used", "test_used_for_finalist_selection"):
        if _value(payload, "test_sealing", forbidden, what) is not False:
            raise FinalistFreezeViolation(f"{what} {forbidden} must be false")

    if _value(payload, "provisional_canonical_v1", "preserved", what) is not True:
        raise FinalistFreezeViolation(
            f"{what} must record that provisional canonical-v1 is preserved, not deleted"
        )
    if _value(payload, "provisional_canonical_v1", "is_final_adapter", what) is not False:
        raise FinalistFreezeViolation(
            f"{what} lets provisional canonical-v1 masquerade as the final adjudicated "
            "adapter; it is the superseded provisional winner"
        )

    if _value(payload, "corpus", "chunk_membership_digest", what) != CORPUS_MANIFEST_DIGEST:
        raise FinalistFreezeViolation(f"{what} corpus membership digest is not the pinned one")
    for field, want in (
        ("sha256", INVENTORY_SHA256),
        ("source_revision", INVENTORY_SOURCE_REVISION),
        ("size_bytes", INVENTORY_SIZE_BYTES),
    ):
        if _value(payload, "inventory", field, what) != want:
            raise FinalistFreezeViolation(f"{what} inventory {field} is not the pinned one")

    declared = _value(payload, "evidence", "provenance_fields_verified", what)
    if not isinstance(declared, list) or tuple(declared) != VERIFIED_PROVENANCE_FIELDS:
        raise FinalistFreezeViolation(
            f"{what} provenance_fields_verified must be exactly "
            f"{list(VERIFIED_PROVENANCE_FIELDS)}; the artifact may not advertise a "
            "weaker or stronger check than the verifier performs"
        )
    workflow = _value(payload, "evidence", "binding_workflow", what)
    if not isinstance(workflow, list) or tuple(workflow) != BINDING_FIELDS:
        raise FinalistFreezeViolation(
            f"{what} binding_workflow must enumerate exactly {list(BINDING_FIELDS)}"
        )

    complete = _value(payload, "evidence", "freeze_complete", what)
    pending = _value(payload, "evidence", "pending_finalist_digests", what)
    if not isinstance(pending, list):
        raise FinalistFreezeViolation(f"{what} pending_finalist_digests must be a list")
    observed = sorted(pending_finalists(finalists))
    if sorted(str(p) for p in pending) != observed:
        raise FinalistFreezeViolation(
            f"{what} declares pending digests {sorted(pending)} but the finalist records "
            f"show {observed}"
        )
    if complete is not (not observed):
        raise FinalistFreezeViolation(
            f"{what} freeze_complete={complete!r} disagrees with the pending digests "
            f"{observed}. The freeze is complete only when every digest is bound."
        )


def finalists_from_freeze(
    payload: Mapping[str, Any], *, what: str = "finalist freeze"
) -> list[FinalistIdentity]:
    """The finalist records carried by a already-validated freeze payload."""
    raw = _value(payload, "adjudication", "finalists", what)
    if not isinstance(raw, list):
        raise FinalistFreezeViolation(f"{what} finalists must be a list")
    return [_finalist_from_dict(entry, what=what) for entry in raw]


def _finalist_from_dict(entry: Any, *, what: str) -> FinalistIdentity:
    if not isinstance(entry, Mapping):
        raise FinalistFreezeViolation(f"{what} finalist entry is not a JSON object")
    required = (
        "key", "role", "source_stage", "run_seed", "update",
        "learning_rate", "r", "validation_score",
        "source_repository_head", "checkpoint_sha256",
    )
    missing = [field for field in required if field not in entry]
    if missing:
        raise FinalistFreezeViolation(f"{what} finalist entry is missing {missing}")
    unknown = sorted(set(entry) - set(required) - {"robust_score"})
    if unknown:
        raise FinalistFreezeViolation(
            f"{what} finalist entry carries unknown field(s) {unknown}; the schema is closed"
        )
    for integral in ("run_seed", "update"):
        value = entry[integral]
        if isinstance(value, bool) or not isinstance(value, int):
            raise FinalistFreezeViolation(f"{what} finalist {integral} must be an integer")
    return FinalistIdentity(
        key=str(entry["key"]),
        role=str(entry["role"]),
        source_stage=str(entry["source_stage"]),
        run_seed=int(entry["run_seed"]),
        update=int(entry["update"]),
        learning_rate=float(entry["learning_rate"]),
        r=float(entry["r"]),
        validation_score=float(entry["validation_score"]),
        source_repository_head=str(entry["source_repository_head"]),
        checkpoint_sha256=str(entry["checkpoint_sha256"]),
        robust_score=(
            float(entry["robust_score"]) if entry.get("robust_score") is not None else None
        ),
    )


# ---------------------------------------------------------------------------
# Checkpoint verification -- READ ONLY
# ---------------------------------------------------------------------------
def resolve_inventory() -> Any:
    """The pinned syllable inventory identity, from the repository's own preflight.

    Fails closed rather than degrading: an inventory that cannot be resolved means
    the `inventory` half of the provenance comparison would be skipped, and
    D-S1A-008 makes that half load-bearing.
    """
    from unmark.stage1.preflight import (  # noqa: PLC0415 - optional at import time
        ScientificInputsUnavailable,
        verify_scientific_inputs,
    )

    try:
        inventory = verify_scientific_inputs().inventory
    except ScientificInputsUnavailable as error:
        raise FinalistFreezeViolation(
            "the pinned Vietnamese syllable inventory is not available, so a "
            "finalist's inventory identity cannot be checked. Provision it with "
            "scripts/fetch_vietnamese_syllable_inventory.py and re-run.\n"
            f"{error}"
        ) from error
    if inventory.sha256 != INVENTORY_SHA256:
        raise FinalistFreezeViolation(
            f"resolved inventory sha256 {inventory.sha256!r} is not the Audit 048 pin "
            f"{INVENTORY_SHA256!r}"
        )
    if inventory.source_revision != INVENTORY_SOURCE_REVISION:
        raise FinalistFreezeViolation(
            f"resolved inventory revision {inventory.source_revision!r} is not the "
            f"Audit 048 pin {INVENTORY_SOURCE_REVISION!r}"
        )
    return inventory


def expected_run_provenance(finalist: FinalistIdentity, *, inventory: Any) -> RunProvenance:
    """The identity this finalist's checkpoint must match, built from the plan.

    Every value is derived or bound, never read from the artifact being checked:
    `init_seed` from `adapter_init_seed(run_seed)`, `corruption_seed` and the
    backbone/protocol/precision defaults from `protocol`, the corpus digest and
    inventory from the Audit 048 pins. That direction is the whole point --
    `RunProvenance.to_dict` is deliberately not constructor-round-trippable so a
    foreign checkpoint cannot define which experiment it belongs to.
    """
    return RunProvenance(
        run_seed=finalist.run_seed,
        init_seed=adapter_init_seed(finalist.run_seed),
        corruption_seed=CORRUPTION_SEED,
        learning_rate=finalist.learning_rate,
        r=finalist.r,
        corpus_manifest_digest=CORPUS_MANIFEST_DIGEST,
        repository_head=finalist.source_repository_head,
        inventory=inventory,
    )


def verify_finalist_checkpoint(
    path: Path | str,
    finalist: FinalistIdentity,
    *,
    require_bound_digest: bool = True,
    inventory: Any = None,
) -> dict[str, Any]:
    """Fail closed unless the file at `path` IS this finalist. Never writes.

    Reuses the repository's own `sha256_file` and the locked checkpoint schema
    rather than introducing a second, subtly different checkpoint parser.

    Args:
        path: operator-supplied checkpoint location. Never inferred, never a
            hard-coded Drive path.
        finalist: the expected identity.
        require_bound_digest: when true (the default) a finalist whose digest is
            still :data:`SHA256_PENDING` is refused, because there is nothing to
            verify against. Set false only by the binding workflow, which is
            establishing the digest for the first time and checks every other
            field before trusting it.
        inventory: the pinned inventory identity. Resolved from the repository's
            own preflight when omitted; injectable so tests need no cache.

    Returns:
        An evidence record: identifiers, scalars and hashes only. No tensors.
        It carries the checkpoint's `execution` fingerprint verbatim so two
        finalists' runtime conditions can be compared later -- that fingerprint
        is operational, not campaign identity, so it is reported rather than
        enforced here.
    """
    path = Path(path)
    if not path.is_file():
        raise FinalistFreezeViolation(f"checkpoint is missing: {path}")
    if inventory is None:
        inventory = resolve_inventory()

    digest = sha256_file(path)
    if finalist.sha256_bound:
        if digest != finalist.checkpoint_sha256:
            raise FinalistFreezeViolation(
                f"sha256 mismatch for finalist {finalist.key}: file is {digest}, frozen "
                f"identity is {finalist.checkpoint_sha256}"
            )
    elif require_bound_digest:
        raise FinalistFreezeViolation(
            f"finalist {finalist.key} has no bound sha256 ({SHA256_PENDING}); its "
            "authoritative digest must be supplied and reviewed before a checkpoint "
            "can be verified against it"
        )

    import torch  # lazy: the freeze contract itself is checkable without torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise FinalistFreezeViolation(
            f"{path} did not deserialise to a mapping; got {type(payload).__name__}"
        )

    # THE authoritative gate. `verify_checkpoint` enforces the full
    # `REQUIRED_CHECKPOINT_KEYS` set, the checkpoint schema version, and
    # `RunProvenance.require_match` -- which compares all twelve scientific
    # identity fields (run_seed, init_seed, corruption_seed, learning_rate, r,
    # corpus_manifest_digest, backbone_checkpoint, backbone_revision,
    # protocol_version, precision, repository_head, inventory) plus the two
    # derived objective weights. Reused rather than re-implemented so this
    # verifier can never drift weaker than the contract a resume must satisfy.
    expected = expected_run_provenance(finalist, inventory=inventory)
    try:
        verify_checkpoint(dict(payload), expected)
    except TrainerContractViolation as error:
        raise FinalistFreezeViolation(f"{path} is not finalist {finalist.key}: {error}") from error

    if payload.get("global_update") != finalist.update:
        raise FinalistFreezeViolation(
            f"{path} global_update {payload.get('global_update')!r} != {finalist.update}"
        )

    provenance = payload["provenance"]
    if not isinstance(provenance, Mapping):
        raise FinalistFreezeViolation(f"{path} carries no provenance mapping")

    # The three load-bearing inventory pins, checked by value as well as through
    # the whole-block comparison above, so a message names the offending field.
    recorded_inventory = provenance.get("inventory")
    if not isinstance(recorded_inventory, Mapping):
        raise FinalistFreezeViolation(
            f"{path} provenance records no inventory identity; D-S1A-008 requires a "
            "scientific run to name the inventory it resolved eligibility with"
        )
    for field, want in (
        ("sha256", INVENTORY_SHA256),
        ("source_revision", INVENTORY_SOURCE_REVISION),
        ("size_bytes", INVENTORY_SIZE_BYTES),
    ):
        got = recorded_inventory.get(field)
        if got != want:
            raise FinalistFreezeViolation(
                f"{path} provenance.inventory.{field} is {got!r}, expected {want!r}"
            )

    adapter_state = payload.get("adapter_state")
    if not isinstance(adapter_state, Mapping) or not adapter_state:
        raise FinalistFreezeViolation(f"{path} carries no adapter_state")
    keys = tuple(sorted(str(k) for k in adapter_state))
    if keys != ADAPTER_STATE_KEYS:
        raise FinalistFreezeViolation(
            f"{path} adapter_state keys {list(keys)} != the locked adapter contract "
            f"{list(ADAPTER_STATE_KEYS)}"
        )

    parameters = 0
    for name, tensor in adapter_state.items():
        if not isinstance(tensor, torch.Tensor):
            raise FinalistFreezeViolation(
                f"{path} adapter_state[{name!r}] is {type(tensor).__name__}, not a tensor"
            )
        if tensor.dtype is not torch.float32:
            raise FinalistFreezeViolation(
                f"{path} adapter_state[{name!r}] dtype is {tensor.dtype}, not float32 "
                f"({PRECISION})"
            )
        if not bool(torch.isfinite(tensor).all()):
            raise FinalistFreezeViolation(
                f"{path} adapter_state[{name!r}] contains NaN or Inf"
            )
        parameters += int(tensor.numel())
    if parameters != ADAPTER_TRAINABLE_PARAMETERS:
        raise FinalistFreezeViolation(
            f"{path} adapter has {parameters} parameters, not the locked "
            f"{ADAPTER_TRAINABLE_PARAMETERS}"
        )

    execution = payload.get("execution")
    return {
        "kind": "stage1_finalist_checkpoint_evidence",
        "finalist_key": finalist.key,
        "role": finalist.role,
        "source_stage": finalist.source_stage,
        "run_seed": finalist.run_seed,
        "init_seed": expected.init_seed,
        "corruption_seed": expected.corruption_seed,
        "update": finalist.update,
        "cap": payload.get("cap"),
        "learning_rate": finalist.learning_rate,
        "r": finalist.r,
        "lambda_align": expected.weights.lambda_align,
        "lambda_clean": expected.weights.lambda_clean,
        "checkpoint_sha256": digest,
        "checkpoint_bytes": path.stat().st_size,
        "source_repository_head": finalist.source_repository_head,
        "corpus_manifest_digest": CORPUS_MANIFEST_DIGEST,
        "inventory": dict(recorded_inventory),
        "backbone_checkpoint": ENCODER_CHECKPOINT,
        "backbone_revision": ENCODER_REVISION,
        "protocol_version": STAGE1_PROTOCOL_VERSION,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "adapter_tensor_count": len(adapter_state),
        "adapter_trainable_parameters": parameters,
        "adapter_dtype": PRECISION,
        "all_finite": True,
        "digest_was_bound_before_verification": finalist.sha256_bound,
        # Operational, NOT campaign identity: `require_match` does not compare it,
        # and D-S1B-015 treats it as resume-blocking rather than run-defining.
        # Emitted so the two finalists' runtime conditions can be compared once
        # both checkpoints are in reach (Audit 048 s7.3).
        "execution_fingerprint": dict(execution) if isinstance(execution, Mapping) else execution,
        "provenance_fields_verified": list(VERIFIED_PROVENANCE_FIELDS),
    }
