"""The real `train_run` resume seam. **Needs torch; Colab/CUDA runs it.**

The torch-free half is `test_stage1_resume_state_machine.py`, kept in a separate
FILE because a module-level `pytest.importorskip` would otherwise skip it too.

Everything here is production machinery: the real `OrthographyInputAdapter`, the
real AdamW from `build_optimizer`, the real `save_training_checkpoint` /
`load_training_checkpoint`, and the real `train_run` restore path -- including
the exact line that used to read `ValidationPoint(**p)`.

No optimizer step is executed, and that is enforced rather than promised: every
case restores at `global_update == cap`, `AdamW.step` is replaced with a
function that fails the test if called, and the objective's forward raises.
"""

from __future__ import annotations

import pathlib
import sys
import types

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from unmark.stage1.contracts import CorruptionRatePolicy  # noqa: E402
from unmark.stage1.protocol import (  # noqa: E402
    CORRUPTION_SEED,
    EXTENDED_MAX_UPDATES,
    INITIAL_MAX_UPDATES,
)
from unmark.stage1.selection import SelectionViolation  # noqa: E402
from unmark.stage1.trainer import (  # noqa: E402
    TrainerContractViolation,
    checkpoint_payload,
    resume_cap,
)
from unmark.stage1.sampler import DeterministicSampler  # noqa: E402

from test_stage1_resume_state_machine import point, provenance  # noqa: E402

# ---------------------------------------------------------------------------
# The real seam: save -> load -> train_run reconstruction
# ---------------------------------------------------------------------------
torch = pytest.importorskip(
    "torch", reason="the real train_run half needs torch; the torch-free half above always runs"
)

from unmark.modeling.adapter import UnmarkEncoder  # noqa: E402
from unmark.stage1.initialisation import fresh_adapter  # noqa: E402
from unmark.stage1.objective import Stage1Objective  # noqa: E402
from unmark.stage1.trainer import (  # noqa: E402
    load_training_checkpoint,
    save_training_checkpoint,
    train_run,
)

CHUNKS = {"doc-0000#0": "xin chao", "doc-0001#0": "chao ban", "doc-0002#0": "toi khoe"}
"""The one chunk set every sampler state in this file is built over. The
sampler binds a `corpus_digest`, so a fabricated state would be refused --
which is the point: these are real checkpoints, not plausible-looking dicts."""

TINY_HIDDEN = 8
"""`d` for the whole file.

Deliberately **not** 768, and this is load-bearing rather than cosmetic:
`verify_model_contract` reads `int(getattr(encoder.config, "hidden_size", 768))`
and enforces the locked 3 551 232 trainable parameters **only when that value
equals 768**. A backbone reporting `hidden_size = 8` therefore exercises every
other clause of the contract -- no leaked encoder parameter, encoder in eval --
while letting a tiny adapter stand in for the real one.

The known-good `RobertaLike` in `tests/test_stage1.py` omits `hidden_size`
entirely, which is correct *there* because that test builds a d=768-independent
wrapper and never calls `verify_model_contract`. Copying it verbatim here would
fall back to 768, activate the parameter-count gate and fail a d=8 adapter."""


class _RobertaLikeConfig:
    """Exactly the config fields production reads. Nothing else.

    Each attribute is here because a named production function consumes it:

    * `model_type`     -- `detect_model_family`, matched against the profile;
    * `_name_or_path`  -- `detect_checkpoint`, the checkpoint *identity*;
    * `pad_token_id`   -- `detect_padding_index`, its documented last resort;
    * `hidden_size`    -- `verify_model_contract` (see `TINY_HIDDEN`).
    """

    model_type = "roberta"
    _name_or_path = "vinai/phobert-base"
    pad_token_id = 1
    hidden_size = TINY_HIDDEN


class _FrozenBackbone(torch.nn.Module):
    """A minimal faithful stand-in for the frozen PhoBERT encoder.

    Audit 034 (034-MAJ1) found the previous stub could not be constructed at
    all. `UnmarkEncoder.__init__` deliberately fails fast:

        self.position_profile = resolve_position_profile(encoder)
        self.padding_index    = detect_padding_index(encoder)

    `resolve_position_profile` matches the WHOLE profile -- checkpoint, model
    type and model class -- against `PHOBERT_BASE_POSITION_PROFILE`
    (`vinai/phobert-base` / `roberta` / `RobertaModel`), because D-B4B-002
    measured one checkpoint and a shared `model_type` is not evidence. The old
    stub offered `(None, None, "_FrozenStub")` and raised
    `UnsupportedPositionSemantics`; it also exposed no padding index at all.

    This class satisfies that contract the way a real encoder does rather than
    by weakening it -- nothing is monkeypatched and no test-only branch exists
    in production. Real (frozen) parameters are carried so that
    `freeze_encoder` has something to freeze and `verify_model_contract`'s
    encoder-leak check is non-vacuous.
    """

    def __init__(self) -> None:
        super().__init__()
        self.word_embeddings = torch.nn.Embedding(64, TINY_HIDDEN, padding_idx=1)
        self.projection = torch.nn.Linear(TINY_HIDDEN, TINY_HIDDEN)
        # `detect_padding_index` prefers the embedding module's own
        # `padding_idx`, then the word-embedding table's, then
        # `config.pad_token_id`. All three agree here, as they do on the real
        # model, so the test cannot pass by accident on the fallback alone.
        self.embeddings = types.SimpleNamespace(
            padding_idx=1, word_embeddings=self.word_embeddings
        )
        self.config = _RobertaLikeConfig()
        self.eval()

    def get_input_embeddings(self):
        return self.word_embeddings

    def forward(self, *args, **kwargs):  # pragma: no cover - never reached
        raise AssertionError(
            "the frozen backbone ran a forward pass: this file restores at "
            "global_update == cap and must perform no update"
        )


# `resolve_position_profile` compares `type(encoder).__name__`, so the class
# must *identify* as RobertaModel -- the same idiom as tests/test_stage1.py.
_FrozenBackbone.__name__ = "RobertaModel"
_FrozenBackbone.__qualname__ = "RobertaModel"


class _NoForwardObjective(Stage1Objective):
    """The REAL `Stage1Objective`, with only its forward poisoned.

    Subclassed rather than replaced so construction still goes through the
    production constructor (which requires a real `ObjectiveWeights`), and so
    `train_run` drives a genuine objective. The forward is unreachable in this
    file by construction -- every case restores at `global_update == cap` -- and
    raising here turns "no update happened" from a claim into an assertion.
    """

    def forward(self, *args, **kwargs):  # pragma: no cover - never reached
        raise AssertionError(
            "the loop body ran: this test restores at global_update == cap and "
            "must perform no update"
        )


def build_objective(seed: int = 51800):
    """A real adapter + real `UnmarkEncoder` + real objective, at d=8."""
    adapter = fresh_adapter(TINY_HIDDEN, seed)
    encoder = _FrozenBackbone()
    wrapper = UnmarkEncoder(encoder=encoder, adapter=adapter)
    return _NoForwardObjective(wrapper, provenance().weights)


def real_checkpoint(tmp_path, *, global_update, cap, points, prov=None, seed=51800):
    """Write a checkpoint the way production writes one, then read it back."""
    objective = build_objective(seed)
    adapter = objective.unmark_encoder.adapter
    from unmark.stage1.optim import build_optimizer

    optimizer = build_optimizer(
        [(n, p) for n, p in objective.unmark_encoder.named_parameters() if p.requires_grad],
        3e-4,
    )
    payload = checkpoint_payload(
        provenance=prov or provenance(),
        adapter_state=adapter.state_dict(),
        optimizer_state=optimizer.state_dict(),
        global_update=global_update,
        sampler_state=DeterministicSampler(
            tuple(sorted(CHUNKS)), seed=(prov or provenance()).run_seed
        ).state_dict(),
        cap=cap,
        budget_limited=False,
        points=points,
    )
    save_training_checkpoint(tmp_path, payload)
    return load_training_checkpoint(tmp_path)


def resume_via_train_run(tmp_path, carried, *, cap, prov=None, chunks=None):
    """Drive the REAL `train_run` resume path. No update may occur."""
    objective = build_objective()
    calls = {"evaluate": 0}

    def evaluate_fn(update):  # pragma: no cover - only if update 0 were missing
        calls["evaluate"] += 1
        return point(update)

    original_step = torch.optim.AdamW.step

    def poisoned_step(self, *args, **kwargs):  # pragma: no cover - must not run
        raise AssertionError("optimizer.step() ran during a no-update resume test")

    torch.optim.AdamW.step = poisoned_step
    try:
        result = train_run(
            objective=objective,
            provenance=prov or provenance(),
            train_chunks=chunks or CHUNKS,
            tokenizer=None,
            corruption_policy=CorruptionRatePolicy(seed=CORRUPTION_SEED),
            truncation=None,
            evaluate_fn=evaluate_fn,
            pad_token_id=1,
            cap=cap,
            resume=carried,
            checkpoint_dir=tmp_path,
            execution=None,
        )
    finally:
        torch.optim.AdamW.step = original_step
    return result, calls


def test_case_a_initial_leg_checkpoint_resumes(tmp_path):
    """A. A legitimate initial-leg checkpoint restores through the real reader."""
    points = [point(0, 0.4), point(500, 0.3)]
    carried = real_checkpoint(
        tmp_path, global_update=INITIAL_MAX_UPDATES, cap=INITIAL_MAX_UPDATES,
        points=points,
    )
    assert resume_cap(carried) == INITIAL_MAX_UPDATES
    result, calls = resume_via_train_run(
        tmp_path, carried, cap=INITIAL_MAX_UPDATES
    )
    assert result.points == points, "the validation history did not survive the round trip"
    assert calls["evaluate"] == 0, "update 0 was restored, not re-measured"


def test_case_b_the_boundary_checkpoint_promotes_to_the_continuation(tmp_path):
    """B. At exactly 20k the budget rule may promote the SAME run to 40k."""
    points = [point(0, 0.9), point(INITIAL_MAX_UPDATES, 0.1)]
    carried = real_checkpoint(
        tmp_path, global_update=INITIAL_MAX_UPDATES, cap=INITIAL_MAX_UPDATES,
        points=points,
    )
    result, _ = resume_via_train_run(tmp_path, carried, cap=INITIAL_MAX_UPDATES)
    # The best checkpoint IS the cap, so the locked rule continues the run.
    assert result.cap == EXTENDED_MAX_UPDATES
    assert result.continued is True
    assert result.budget_limited is False


def test_case_c_a_40k_continuation_checkpoint_reconstructs_cap_40000(tmp_path):
    """C. The blocker. `global_update=20500, cap=40000` must resume as 40k.

    Before the repair `execute_stage` passed `cap=INITIAL_MAX_UPDATES` here, so
    `while global_update < cap` was already false and the run recorded
    `cap=20000, continued=False` -- 40k of work labelled a complete 20k run,
    with no error raised anywhere.
    """
    points = [point(0, 0.1), point(INITIAL_MAX_UPDATES, 0.4), point(20_500, 0.5)]
    carried = real_checkpoint(
        tmp_path, global_update=20_500, cap=EXTENDED_MAX_UPDATES, points=points,
    )
    assert resume_cap(carried) == EXTENDED_MAX_UPDATES

    result, _ = resume_via_train_run(
        tmp_path, carried, cap=resume_cap(carried)
    )
    assert result.cap == EXTENDED_MAX_UPDATES, "the continuation leg was mislabelled"
    assert result.continued is True
    assert result.points == points


def test_case_c_the_pre_repair_cap_is_now_refused(tmp_path):
    """The silent mislabel is impossible: the wrong cap fails closed."""
    carried = real_checkpoint(
        tmp_path, global_update=20_500, cap=EXTENDED_MAX_UPDATES,
        points=[point(0, 0.1), point(20_500, 0.5)],
    )
    with pytest.raises(TrainerContractViolation, match="smaller cap"):
        resume_via_train_run(tmp_path, carried, cap=INITIAL_MAX_UPDATES)


def test_case_d_a_later_continuation_preserves_continued_state(tmp_path):
    """D. Deep in the 40k leg, `continued` and the budget survive the restore."""
    points = [point(0, 0.1), point(39_500, 0.6), point(EXTENDED_MAX_UPDATES, 0.7)]
    carried = real_checkpoint(
        tmp_path, global_update=EXTENDED_MAX_UPDATES, cap=EXTENDED_MAX_UPDATES,
        points=points,
    )
    result, _ = resume_via_train_run(
        tmp_path, carried, cap=resume_cap(carried)
    )
    assert result.cap == EXTENDED_MAX_UPDATES
    assert result.continued is True
    assert len(result.points) == 3


def test_case_d_budget_limited_when_the_best_is_the_extended_cap(tmp_path):
    """The other end of the locked rule: stop and mark BUDGET_LIMITED."""
    points = [point(0, 0.9), point(EXTENDED_MAX_UPDATES, 0.1)]
    carried = real_checkpoint(
        tmp_path, global_update=EXTENDED_MAX_UPDATES, cap=EXTENDED_MAX_UPDATES,
        points=points,
    )
    result, _ = resume_via_train_run(
        tmp_path, carried, cap=resume_cap(carried)
    )
    assert result.budget_limited is True
    assert result.cap == EXTENDED_MAX_UPDATES


def test_case_e_global_update_beyond_the_cap_is_refused_by_train_run(tmp_path):
    """E. Through the real reader, not just the helper."""
    carried = real_checkpoint(
        tmp_path, global_update=INITIAL_MAX_UPDATES, cap=INITIAL_MAX_UPDATES,
        points=[point(0)],
    )
    carried["global_update"] = 25_000
    with pytest.raises(TrainerContractViolation, match="cannot have progressed"):
        resume_via_train_run(tmp_path, carried, cap=INITIAL_MAX_UPDATES)


def test_case_f_an_invalid_persisted_cap_is_refused_by_train_run(tmp_path):
    """F. Through the real reader."""
    carried = real_checkpoint(
        tmp_path, global_update=500, cap=INITIAL_MAX_UPDATES, points=[point(0)],
    )
    carried["cap"] = 30_000
    with pytest.raises(TrainerContractViolation, match="locked budgets"):
        resume_via_train_run(tmp_path, carried, cap=INITIAL_MAX_UPDATES)


def test_case_g_a_malformed_point_is_refused_by_train_run(tmp_path):
    """G. The malformed payload reaches the production reader and is refused."""
    carried = real_checkpoint(
        tmp_path, global_update=500, cap=INITIAL_MAX_UPDATES,
        points=[point(0), point(500)],
    )
    carried["points"][1]["score"] = 99.0
    with pytest.raises(SelectionViolation, match="recompute"):
        resume_via_train_run(tmp_path, carried, cap=INITIAL_MAX_UPDATES)


def test_the_adapter_and_optimizer_state_still_restore(tmp_path):
    """The repairs must not have cost the existing restore guarantees."""
    carried = real_checkpoint(
        tmp_path, global_update=INITIAL_MAX_UPDATES, cap=INITIAL_MAX_UPDATES,
        points=[point(0)],
    )
    assert carried["adapter_state"], "adapter state must be persisted"
    assert carried["optimizer_state"]["param_groups"], "optimizer state must be persisted"
    result, _ = resume_via_train_run(tmp_path, carried, cap=INITIAL_MAX_UPDATES)
    assert result.provenance.run_seed == 36930


def test_a_foreign_run_still_cannot_resume(tmp_path):
    """Provenance matching is unchanged by the cap repair."""
    carried = real_checkpoint(
        tmp_path, global_update=500, cap=INITIAL_MAX_UPDATES, points=[point(0)],
        prov=provenance(run_seed=7309),
    )
    with pytest.raises(TrainerContractViolation, match="provenance mismatch"):
        resume_via_train_run(tmp_path, carried, cap=INITIAL_MAX_UPDATES)


# ---------------------------------------------------------------------------
# Multi-run orchestration: completed / crashed-in-40k / not started
# ---------------------------------------------------------------------------
def test_three_candidates_resume_independently(tmp_path):
    """A completed, B crashed during the 40k leg, C never started.

    Each candidate owns its own checkpoint namespace, so B's continuation cap
    cannot reach A or C, and C must still start fresh at 20k.
    """
    runs = {
        "A": (tmp_path / "run-A" / "_checkpoint", INITIAL_MAX_UPDATES, INITIAL_MAX_UPDATES),
        "B": (tmp_path / "run-B" / "_checkpoint", 20_500, EXTENDED_MAX_UPDATES),
    }
    for label, (directory, global_update, cap) in runs.items():
        directory.mkdir(parents=True)
        real_checkpoint(
            directory, global_update=global_update, cap=cap,
            points=[point(0, 0.2), point(INITIAL_MAX_UPDATES, 0.3)]
            if global_update >= INITIAL_MAX_UPDATES else [point(0, 0.2)],
        )

    carried_a = load_training_checkpoint(runs["A"][0])
    carried_b = load_training_checkpoint(runs["B"][0])
    carried_c = load_training_checkpoint(tmp_path / "run-C" / "_checkpoint")

    assert resume_cap(carried_a) == INITIAL_MAX_UPDATES, "A is on the initial leg"
    assert resume_cap(carried_b) == EXTENDED_MAX_UPDATES, "B is on the continuation leg"
    assert carried_c is None, "C has no checkpoint and must start fresh"

    # The cap `execute_stage` derives for each: from the checkpoint, or the
    # locked initial budget when there is none.
    caps = {
        "A": resume_cap(carried_a),
        "B": resume_cap(carried_b),
        "C": INITIAL_MAX_UPDATES,
    }
    assert caps == {"A": 20_000, "B": 40_000, "C": 20_000}

    # B's restored history must not leak into A's.
    result_a, _ = resume_via_train_run(runs["A"][0], carried_a, cap=caps["A"])
    result_b, _ = resume_via_train_run(runs["B"][0], carried_b, cap=caps["B"])
    assert result_a.cap == INITIAL_MAX_UPDATES
    assert result_b.cap == EXTENDED_MAX_UPDATES
    assert result_b.continued is True
    assert result_a.points is not result_b.points
