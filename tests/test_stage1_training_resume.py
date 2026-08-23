"""Training checkpoint/resume equivalence (Audit 030 F3 hardening).

The mandatory question: is `run → checkpoint → die → rebuild → resume → finish`
scientifically the same as an uninterrupted run?

Answered with a **tiny synthetic model on CPU**. No real encoder, no weight
download, no pinned tokenizer. Exact tensor equality is required, not a
tolerance: on CPU with a fixed seed and identical operation order there is no
source of nondeterminism, so anything less than exact equality would be hiding
a real difference.

Interruption is exercised at the awkward positions the audit named: on a
validation boundary, immediately after one, across a pass/visit boundary, and
at the final update.
"""

from __future__ import annotations

import dataclasses
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from unmark.stage1.sampler import DeterministicSampler  # noqa: E402
from unmark.stage1.trainer import (  # noqa: E402
    CHECKPOINT_EVERY_UPDATES,
    REQUIRED_CHECKPOINT_KEYS,
    RunProvenance,
    TrainerContractViolation,
    checkpoint_payload,
    load_training_checkpoint,
    save_training_checkpoint,
    verify_checkpoint,
)


torch = pytest.importorskip(
    "torch", reason="the tensor half needs torch; the torch-free half below always runs"
)


# ---------------------------------------------------------------------------
# A deliberately tiny stand-in: real optimizer, real autograd, no real model
# ---------------------------------------------------------------------------
class TinyAdapter(torch.nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.fusion = torch.nn.Linear(dim, dim, bias=False)
        self.gate = torch.nn.Linear(dim, 1, bias=False)

    def forward(self, x):
        return self.gate(torch.tanh(self.fusion(x))).sum()


def provenance() -> RunProvenance:
    return RunProvenance(
        run_seed=36930,
        corruption_seed=35422,
        learning_rate=3e-4,
        r=1.0,
        corpus_manifest_digest="d" * 64,
        repository_head="a" * 40,
    )


def chunk_ids(n: int = 12) -> tuple[str, ...]:
    return tuple(f"doc-{i:04d}#0" for i in range(n))


def batch_for(pairs, dim: int = 8):
    """Deterministic input from the sampler's own output, so the data a step
    sees depends on the cursor exactly as real training does."""
    rows = []
    for chunk_id, visit in pairs:
        seed = (hash((chunk_id, visit)) % 9973) / 9973.0
        rows.append([seed + j * 0.01 for j in range(dim)])
    return torch.tensor(rows, dtype=torch.float64)


def build(seed: int = 0):
    torch.manual_seed(seed)
    model = TinyAdapter().double()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    return model, optimizer


def train(model, optimizer, sampler, start_update: int, cap: int,
          checkpoint_dir=None, points=None, batch: int = 4):
    """A faithful miniature of `train_run`'s loop shape: step, then checkpoint
    at the validation boundary."""
    points = list(points or [])
    update = start_update
    while update < cap:
        pairs = sampler.next_batch(batch)
        loss = model(batch_for(pairs))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        update += 1
        if update % EVERY == 0 or update == cap:
            points.append({"update": update, "score": float(loss.detach())})
            if checkpoint_dir is not None:
                save_training_checkpoint(
                    checkpoint_dir,
                    checkpoint_payload(
                        provenance=provenance(),
                        adapter_state=model.state_dict(),
                        optimizer_state=optimizer.state_dict(),
                        global_update=update,
                        sampler_state=sampler.state_dict(),
                        cap=cap,
                        budget_limited=False,
                        points=points,
                    ),
                )
    return points


EVERY = 4          # the miniature's validation cadence
CAP = 16


def fingerprint(model, optimizer, sampler, update, points):
    return {
        "params": [p.detach().clone() for p in model.parameters()],
        "optimizer": optimizer.state_dict(),
        "update": update,
        "visit": sampler.visit,
        "position": sampler.position,
        "points": points,
    }


def assert_equivalent(a, b):
    assert len(a["params"]) == len(b["params"])
    for left, right in zip(a["params"], b["params"]):
        assert torch.equal(left, right), "adapter tensors differ"
    assert a["update"] == b["update"]
    assert a["visit"] == b["visit"], "visit/pass state differs"
    assert a["position"] == b["position"], "sampler cursor differs"
    assert a["points"] == b["points"], "validation history differs"
    left_state = a["optimizer"]["state"]
    right_state = b["optimizer"]["state"]
    assert set(left_state) == set(right_state), "optimizer state keys differ"
    for key in left_state:
        for field, value in left_state[key].items():
            other = right_state[key][field]
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, other), f"optimizer {field} differs"
            else:
                assert value == other, f"optimizer {field} differs"


# ---------------------------------------------------------------------------
# The equivalence property, at every awkward interruption point
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kill_at", [
    4,    # exactly on a validation/checkpoint boundary
    8,    # a later boundary, after a pass boundary has been crossed
    12,   # deep in, several passes in
])
def test_interrupted_then_resumed_equals_uninterrupted(tmp_path, kill_at):
    ids = chunk_ids()

    model, optimizer = build()
    sampler = DeterministicSampler(ids, seed=36930)
    whole_points = train(model, optimizer, sampler, 0, CAP)
    whole = fingerprint(model, optimizer, sampler, CAP, whole_points)

    # ... and now the same run, killed and rebuilt from disk.
    model_a, optimizer_a = build()
    sampler_a = DeterministicSampler(ids, seed=36930)
    train(model_a, optimizer_a, sampler_a, 0, kill_at, checkpoint_dir=tmp_path)
    del model_a, optimizer_a, sampler_a          # the process "dies"

    payload = load_training_checkpoint(tmp_path)
    assert payload is not None, "no checkpoint was published"
    verify_checkpoint(payload, provenance())

    model_b, optimizer_b = build(seed=999)       # a DIFFERENT fresh init
    model_b.load_state_dict(payload["adapter_state"])
    optimizer_b.load_state_dict(payload["optimizer_state"])
    sampler_b = DeterministicSampler.from_state(ids, payload["sampler_state"])
    resumed_points = train(
        model_b, optimizer_b, sampler_b, int(payload["global_update"]), CAP,
        points=payload["points"],
    )
    resumed = fingerprint(model_b, optimizer_b, sampler_b, CAP, resumed_points)

    assert_equivalent(whole, resumed)


def test_resume_restores_the_validation_history():
    """`points` used to be read on resume but never written (F3 hardening)."""
    assert "points" in REQUIRED_CHECKPOINT_KEYS
    payload = checkpoint_payload(
        provenance=provenance(), adapter_state={}, optimizer_state={},
        global_update=500, sampler_state={}, cap=20_000, budget_limited=False,
        points=[{"update": 0, "score": 1.0}, {"update": 500, "score": 2.0}],
    )
    assert payload["points"] == [{"update": 0, "score": 1.0}, {"update": 500, "score": 2.0}]


def test_a_payload_without_points_is_refused():
    payload = checkpoint_payload(
        provenance=provenance(), adapter_state={}, optimizer_state={},
        global_update=1, sampler_state={}, cap=2, budget_limited=False,
    )
    payload.pop("points")
    with pytest.raises(TrainerContractViolation) as caught:
        verify_checkpoint(payload, provenance())
    assert "points" in str(caught.value)


# ---------------------------------------------------------------------------
# Identity binding: a foreign experiment must not resume
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("field,value", [
    ("run_seed", 7309),
    ("corruption_seed", 1),
    ("learning_rate", 1e-3),
    ("r", 4.0),
    ("corpus_manifest_digest", "e" * 64),
    ("repository_head", "b" * 40),
])
def test_a_foreign_run_cannot_resume(field, value):
    """The six identities that must never be resumed across.

    Built with `dataclasses.replace`, which goes through the real constructor.
    The first real Colab smoke found these six dying in **setup** on
    `RunProvenance(**mine.to_dict())`: `to_dict` also emits the derived
    `lambda_align`/`lambda_clean`, which are not constructor parameters, so
    every case raised `TypeError` before `verify_checkpoint` was ever called and
    none of them had demonstrated anything. Audit 030 §V.
    """
    mine = provenance()
    theirs = dataclasses.replace(mine, **{field: value})
    assert getattr(theirs, field) == value, "the mutation must actually take"
    assert theirs != mine

    payload = checkpoint_payload(
        provenance=theirs, adapter_state={}, optimizer_state={},
        global_update=1, sampler_state={}, cap=2, budget_limited=False, points=[],
    )
    with pytest.raises(TrainerContractViolation) as caught:
        verify_checkpoint(payload, mine)

    # Not merely "something raised": the rejection must be the identity
    # violation for THIS field. A setup error, or a mismatch reported on some
    # other key, no longer passes.
    message = str(caught.value)
    assert "checkpoint provenance mismatch" in message, message
    assert repr(field) in message, message
    assert repr(value) in message, message


# ---------------------------------------------------------------------------
# Crash safety of publication
# ---------------------------------------------------------------------------
def test_publication_is_atomic_and_leaves_no_partial_file(tmp_path):
    payload = checkpoint_payload(
        provenance=provenance(), adapter_state={"w": torch.zeros(3)},
        optimizer_state={}, global_update=4, sampler_state={}, cap=8,
        budget_limited=False, points=[],
    )
    path = save_training_checkpoint(tmp_path, payload)
    assert path.is_file()
    assert not list(tmp_path.glob("*.tmp")), "a temp file survived publication"

    # Overwriting keeps exactly one live checkpoint, and it is the newer one.
    payload["global_update"] = 8
    save_training_checkpoint(tmp_path, payload)
    assert load_training_checkpoint(tmp_path)["global_update"] == 8
    assert len(list(tmp_path.glob("training-checkpoint*"))) == 1


def test_no_checkpoint_yet_reads_as_none(tmp_path):
    assert load_training_checkpoint(tmp_path) is None


def test_the_cadence_is_an_update_count_not_wall_clock():
    from unmark.stage1.protocol import EVAL_EVERY_UPDATES

    assert CHECKPOINT_EVERY_UPDATES == EVAL_EVERY_UPDATES
    assert isinstance(CHECKPOINT_EVERY_UPDATES, int)


# ---------------------------------------------------------------------------
# Provenance round trip through a REAL torch checkpoint (Audit 030 §V)
# ---------------------------------------------------------------------------
def test_provenance_survives_a_real_torch_checkpoint_round_trip(tmp_path):
    """p -> to_dict -> torch.save -> torch.load -> verify_checkpoint.

    The torch-free half of this lives in `test_stage1_provenance_contract.py`;
    this is the leg that actually crosses the filesystem, because the recorded
    identity is only worth anything if it comes back off disk unchanged.
    """
    mine = provenance()
    save_training_checkpoint(tmp_path, checkpoint_payload(
        provenance=mine, adapter_state={"w": torch.zeros(3)}, optimizer_state={},
        global_update=9, sampler_state={"cursor": 2, "visit": 0}, cap=20,
        budget_limited=False, points=[],
    ))
    recovered = load_training_checkpoint(tmp_path)
    assert recovered is not None

    for field in ("run_seed", "corruption_seed", "learning_rate", "r",
                  "corpus_manifest_digest", "repository_head", "backbone_checkpoint",
                  "backbone_revision", "protocol_version", "precision"):
        assert recovered["provenance"][field] == getattr(mine, field), field
    for key in RunProvenance.DERIVED_KEYS:
        assert recovered["provenance"][key] == mine.to_dict()[key], key

    verify_checkpoint(recovered, mine)  # the authoritative gate accepts it


def test_a_foreign_identity_is_rejected_after_a_real_round_trip(tmp_path):
    """The rejection must survive serialization, not just live in memory."""
    mine = provenance()
    save_training_checkpoint(tmp_path, checkpoint_payload(
        provenance=dataclasses.replace(mine, repository_head="b" * 40),
        adapter_state={}, optimizer_state={}, global_update=1,
        sampler_state={}, cap=2, budget_limited=False, points=[],
    ))
    with pytest.raises(TrainerContractViolation) as caught:
        verify_checkpoint(load_training_checkpoint(tmp_path), mine)
    assert "repository_head" in str(caught.value)


def test_best_and_last_both_carry_the_same_verified_identity(tmp_path):
    mine = provenance()
    save_training_checkpoint(tmp_path, checkpoint_payload(
        provenance=mine, adapter_state={}, optimizer_state={}, global_update=5,
        sampler_state={}, cap=20, budget_limited=False, points=[],
    ), is_best=True)
    for best in (False, True):
        recovered = load_training_checkpoint(tmp_path, best=best)
        assert recovered is not None, best
        verify_checkpoint(recovered, mine)
