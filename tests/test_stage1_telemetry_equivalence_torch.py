"""Telemetry changes NOTHING scientific. **Needs torch; Colab/CUDA runs it.**

Audit 040. The torch-free half is `test_stage1_telemetry.py`, kept in a separate
FILE because a module-level `pytest.importorskip` would skip it too.

This is the mandatory equivalence proof. The **real** `train_run` is driven
twice over the smallest genuine training path available -- real
`OrthographyInputAdapter`, real `Stage1Objective`, real AdamW from
`build_optimizer`, real `DeterministicSampler`, real
`PreparedStage1Example`s built by the real `prepare_example` -- once with
telemetry OFF and once with it ON. Everything scientific must come out
bit-identical:

    final adapter parameters
    optimizer state
    sampler state
    global_update
    ValidationPoint history
    checkpoint payload
    production RunResult

and the counted number of forward, backward, optimizer, sampling, preparation
and evaluation calls must be equal.

Preparation is supplied by a deterministic stub pool that returns a **fixed,
precomputed** batch. That is not mocking away anything under test: it removes
the tokenizer/corruption worker pool, which telemetry does not touch, so that
the comparison isolates exactly the thing being proven. Every seam telemetry
*does* touch -- the loop, the optimizer, the sampler, validation, checkpointing
-- is real on both sides.
"""

from __future__ import annotations

import copy
import io
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

torch = pytest.importorskip(
    "torch", reason="the scientific-equivalence half needs torch; "
                    "test_stage1_telemetry.py always runs"
)

import types  # noqa: E402

from unmark.linguistics import make_classifier, try_load_inventory  # noqa: E402
from unmark.modeling.adapter import UnmarkEncoder  # noqa: E402
from unmark.stage1.contracts import CorruptionRatePolicy, TruncationPolicy  # noqa: E402
from unmark.stage1.data import Stage1Example, prepare_example  # noqa: E402
from unmark.stage1.initialisation import fresh_adapter  # noqa: E402
from unmark.stage1.objective import Stage1Objective  # noqa: E402
from unmark.stage1.optim import build_optimizer  # noqa: E402
from unmark.stage1.protocol import (  # noqa: E402
    BATCH_SIZE,
    CORRUPTION_SEED,
    EVAL_EVERY_UPDATES,
    INITIAL_MAX_UPDATES,
    VALIDATION_CONDITIONS,
)
from unmark.stage1.selection import ValidationPoint  # noqa: E402
from unmark.stage1.sampler import DeterministicSampler  # noqa: E402
from unmark.stage1.telemetry import PREFIX, PROGRESS_EVERY_UPDATES, JsonlSink  # noqa: E402
from unmark.stage1.trainer import (  # noqa: E402
    RunProvenance,
    checkpoint_payload,
    load_training_checkpoint,
    save_training_checkpoint,
    train_run,
)

from test_stage1 import StubTokenizer  # noqa: E402

TINY_HIDDEN = 8
"""Not 768, so `verify_model_contract` skips the locked parameter-count clause
while every other clause still applies. Same rationale as the resume tests."""

CAP = INITIAL_MAX_UPDATES
"""The REAL locked initial budget, 20 000.

The first CUDA run of this file failed here, in the telemetry-OFF execution and
before any ON-vs-OFF comparison. The fixture used to run a *fresh* run with
`CAP = 4`; the four iterations executed, and then `train_run` correctly reached
`resolve_budget` -> `budget_decision(selected_update=4, cap=4)`, which raised

    SelectionViolation: cap 4 is not one of the locked budgets (20000, 40000)

That is production behaving exactly as designed: the budget rule is
precommitted, and an arbitrary small cap is not a legal Stage-1 budget. The
defect was the fixture, not the rule -- so the rule is untouched and the fixture
now runs under a legal cap instead."""

START_UPDATE = CAP - 50
"""19 950. The fixture RESUMES here rather than starting fresh, so the loop runs
exactly 50 real iterations under the real 20 000 cap. That is a legal scientific
state -- a crash resume on the initial leg -- and it costs 50 optimizer steps
rather than 20 000."""

UPDATES = CAP - START_UPDATE
"""50 real training iterations per execution."""

CHECKPOINT_NAME = "training-checkpoint-last.pt"

CHUNKS = {f"doc-{i:04d}#0": f"xin chao ban {i}" for i in range(6)}


def provenance() -> RunProvenance:
    return RunProvenance(
        run_seed=36930, init_seed=51800, corruption_seed=CORRUPTION_SEED,
        learning_rate=3e-4, r=1.0, corpus_manifest_digest="d" * 64,
        repository_head="a" * 40,
    )


class _RobertaLikeConfig:
    model_type = "roberta"
    _name_or_path = "vinai/phobert-base"
    pad_token_id = 1
    hidden_size = TINY_HIDDEN


class _TinyBackbone(torch.nn.Module):
    """A frozen encoder that actually runs a forward, at d=8."""

    def __init__(self) -> None:
        super().__init__()
        self.word_embeddings = torch.nn.Embedding(4096, TINY_HIDDEN, padding_idx=1)
        self.projection = torch.nn.Linear(TINY_HIDDEN, TINY_HIDDEN)
        self.embeddings = types.SimpleNamespace(
            padding_idx=1, word_embeddings=self.word_embeddings
        )
        self.config = _RobertaLikeConfig()
        self.eval()

    def get_input_embeddings(self):
        return self.word_embeddings

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None,
                position_ids=None, **_):
        hidden = self.word_embeddings(input_ids) if inputs_embeds is None else inputs_embeds
        return self.projection(hidden)


_TinyBackbone.__name__ = "RobertaModel"
_TinyBackbone.__qualname__ = "RobertaModel"


def build_objective():
    """Deterministic: the adapter comes from the locked init-seed derivation."""
    adapter = fresh_adapter(TINY_HIDDEN, provenance().init_seed)
    wrapper = UnmarkEncoder(encoder=_TinyBackbone(), adapter=adapter)
    return Stage1Objective(wrapper, provenance().weights)


def fixed_batch():
    """One precomputed batch of REAL `PreparedStage1Example`s.

    Built once, by the real `prepare_example`, and reused verbatim by both runs
    so preparation contributes nothing to the comparison.
    """
    classifier = make_classifier(try_load_inventory())
    tokenizer = StubTokenizer()
    policy = CorruptionRatePolicy(seed=CORRUPTION_SEED)
    prepared = []
    for index in range(BATCH_SIZE):
        prepared.append(prepare_example(
            Stage1Example(f"xin chao ban {index}", f"doc-{index:04d}#0"),
            tokenizer, corruption_policy=policy, classifier=classifier,
            truncation=TruncationPolicy.unbounded(), visit=0,
        ))
    return prepared


class _FixedPool:
    """A deterministic stand-in for `PreparationPool`. Counts its calls."""

    def __init__(self, prepared) -> None:
        self._prepared = prepared
        self.calls = 0

    def prepare(self, tasks):
        self.calls += 1
        return list(self._prepared[:len(tasks)])


def clone_tensors(state):
    """Deep, tensor-aware snapshot. Shallow dict equality is not enough here."""
    out = {}
    for key, value in state.items():
        out[key] = value.detach().clone() if torch.is_tensor(value) else copy.deepcopy(value)
    return out


def same_tensor_state(a, b) -> bool:
    """Bit-identical comparison of two tensor state dicts."""
    if set(a) != set(b):
        return False
    for key, value in a.items():
        other = b[key]
        if torch.is_tensor(value) != torch.is_tensor(other):
            return False
        if torch.is_tensor(value):
            if value.shape != other.shape or not torch.equal(value, other):
                return False
        elif value != other:
            return False
    return True


def assert_same_start(first, second, what: str) -> None:
    """Every scientific input to `train_run` must match, bit for bit."""
    a, b = first["initial"], second["initial"]
    assert same_tensor_state(a["adapter"], b["adapter"]), f"{what}: adapter start differs"
    assert same_tensor_state(a["encoder"], b["encoder"]), (
        f"{what}: FROZEN ENCODER start differs -- the encoder is restored from no "
        "checkpoint, so it must be constructed under a controlled seed"
    )
    assert same_tensor_state(a["carried_adapter"], b["carried_adapter"]), (
        f"{what}: resume payload adapter differs")
    assert a["carried_sampler"] == b["carried_sampler"], f"{what}: sampler state differs"
    assert a["carried_points"] == b["carried_points"], f"{what}: point history differs"
    assert a["carried_global_update"] == b["carried_global_update"], f"{what}: global_update"
    assert a["carried_cap"] == b["carried_cap"], f"{what}: cap differs"
    assert a["carried_provenance"] == b["carried_provenance"], f"{what}: provenance differs"
    assert torch.equal(a["rng"], b["rng"]), f"{what}: torch RNG state differs at entry"


def assert_same_outcome(first, second, what: str) -> None:
    """Every scientific output must match, bit for bit."""
    assert same_tensor_state(first["adapter_state"], second["adapter_state"]), (
        f"{what}: final adapter tensors differ")
    assert same_tensor_state(first["checkpoint"]["adapter_state"],
                             second["checkpoint"]["adapter_state"]), (
        f"{what}: written checkpoint adapter tensors differ")
    assert first["points"] == second["points"], f"{what}: validation history differs"
    assert first["cap"] == second["cap"], f"{what}: cap differs"
    assert first["continued"] == second["continued"], f"{what}: continued differs"
    assert first["budget_limited"] == second["budget_limited"], f"{what}: budget_limited"
    assert first["result"].to_dict() == second["result"].to_dict(), f"{what}: RunResult differs"
    assert first["checkpoint"]["sampler_state"] == second["checkpoint"]["sampler_state"], (
        f"{what}: sampler state differs")
    assert torch.equal(first["rng_after"], second["rng_after"]), f"{what}: RNG state differs"
    for counter in ("forward", "step", "next_batch", "evaluate"):
        assert getattr(first["counters"], counter) == getattr(second["counters"], counter), (
            f"{what}: {counter} count differs")
    assert first["pool_calls"] == second["pool_calls"], f"{what}: preparation calls differ"


class _Counters:
    def __init__(self) -> None:
        self.forward = 0
        self.step = 0
        self.next_batch = 0
        self.evaluate = 0


def point_for(update: int) -> ValidationPoint:
    """Deterministic validation, so the comparison isolates telemetry.

    Update 0 is deliberately the BEST point (lowest worst-case distance). Under
    the locked rule `select_checkpoint` minimises `(score, d_clean, update)`, so
    the selected checkpoint is update 0, and `budget_decision(0, 20000)` returns
    "selected checkpoint is inside the budget" -- no continuation is triggered.
    The fixture therefore exercises the real budget rule under a real cap
    without dragging a 40 000-update leg into a unit test.
    """
    worst = 0.10 + (update / CAP) * 0.40
    return ValidationPoint(
        update=update,
        distances={c: worst + i / 100.0 for i, c in enumerate(VALIDATION_CONDITIONS)},
        d_clean=worst / 2.0,
    )


def resume_history():
    """The canonical `ValidationPoint` history a run at 19 950 would hold.

    Every 500-update boundary up to the last one before `START_UPDATE`, exactly
    as production would have written it, plus the mandatory update 0.
    """
    updates = [0] + [u for u in range(EVAL_EVERY_UPDATES, START_UPDATE + 1,
                                      EVAL_EVERY_UPDATES)]
    return [point_for(u) for u in updates]


def build_resume_payload(tmp_path):
    """A REAL checkpoint at `global_update=19950, cap=20000`, via production.

    Written by `checkpoint_payload` + `save_training_checkpoint` and read back by
    `load_training_checkpoint`, so the payload both runs resume from is the real
    schema produced by the real writer -- not a hand-built dict.

    The persisted optimizer state is that of a freshly built AdamW, i.e. no
    accumulated moments. That is a structurally valid payload and keeps the
    fixture minimal; the *populated* optimizer state is still compared, because
    the checkpoint written at update 20 000 by each execution carries real
    moments and is compared ON vs OFF.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    # Seeded for the same reason `run_once` is: `_TinyBackbone` draws from the
    # global RNG, so an unseeded build would make the payload depend on whatever
    # ran before it.
    torch.manual_seed(0)
    objective = build_objective()
    adapter = objective.unmark_encoder.adapter
    optimizer = build_optimizer(
        [(n, p) for n, p in objective.unmark_encoder.named_parameters() if p.requires_grad],
        provenance().learning_rate,
    )
    sampler = DeterministicSampler(tuple(sorted(CHUNKS)), seed=provenance().run_seed)
    payload = checkpoint_payload(
        provenance=provenance(),
        adapter_state=adapter.state_dict(),
        optimizer_state=optimizer.state_dict(),
        global_update=START_UPDATE,
        sampler_state=sampler.state_dict(),
        cap=CAP,
        budget_limited=False,
        points=resume_history(),
    )
    save_training_checkpoint(tmp_path, payload)
    return load_training_checkpoint(tmp_path)


def run_once(tmp_path, *, sink, prepared, carried, seed=0):
    """One REAL `train_run` RESUME, instrumented with counters.

    `carried` is the real checkpoint at 19 950/20 000. Nothing on the restore
    path is bypassed: `verify_checkpoint`, `require_resumable_leg`,
    `adapter.load_state_dict(strict=True)`, `optimizer.load_state_dict`,
    `require_optimizer_parameter_identity`, `require_optimizer_state_device`,
    `DeterministicSampler.from_state`, `ValidationPoint.from_dict` and
    `resolve_budget` all execute for real.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    # A private deep copy per execution, so OFF and ON provably begin from
    # scientifically identical state even if a restore path were ever to touch
    # the payload it was handed.
    carried = copy.deepcopy(carried)

    # ------------------------------------------------------------------
    # SEED FIRST. This line used to sit ~28 lines further down, AFTER
    # `build_objective()`, and that was the entire CUDA failure (Audit 042).
    #
    # `_TinyBackbone.__init__` builds `nn.Embedding` and `nn.Linear`, whose
    # `reset_parameters()` draws from the GLOBAL torch RNG. The adapter is
    # immune twice over -- `fresh_adapter` forks the RNG and restores it, and
    # `adapter.load_state_dict(..., strict=True)` overwrites it from the resume
    # payload -- but the frozen ENCODER is restored from nothing, because v2
    # checkpoints are adapter-only by design.
    #
    # So with the old ordering the second execution built its encoder from the
    # RNG state the FIRST execution's 50 updates had left behind: different
    # frozen weights, different representations, different loss, different
    # gradients, and a diverged `tone_embedding.weight`. It would have failed
    # OFF-vs-OFF just as surely as OFF-vs-ON.
    # ------------------------------------------------------------------
    torch.manual_seed(seed)
    objective = build_objective()
    counters = _Counters()
    pool = _FixedPool(prepared)

    original_forward = Stage1Objective.forward
    original_step = torch.optim.AdamW.step

    def counted_forward(self, batch):
        counters.forward += 1
        return original_forward(self, batch)

    def counted_step(self, *args, **kwargs):
        counters.step += 1
        return original_step(self, *args, **kwargs)

    def evaluate_fn(update):
        counters.evaluate += 1
        return point_for(update)

    original_next = DeterministicSampler.next_batch

    def counted_next(self, size):
        counters.next_batch += 1
        return original_next(self, size)

    Stage1Objective.forward = counted_forward
    torch.optim.AdamW.step = counted_step
    DeterministicSampler.next_batch = counted_next
    # Captured AFTER construction: deterministic, because the seed above is set
    # before anything consumes RNG.
    rng_before = torch.random.get_rng_state()
    initial = {
        "adapter": clone_tensors(objective.unmark_encoder.adapter.state_dict()),
        "encoder": clone_tensors(objective.unmark_encoder.encoder.state_dict()),
        "carried_adapter": clone_tensors(carried["adapter_state"]),
        "carried_sampler": dict(carried["sampler_state"]),
        "carried_points": [dict(p) for p in carried["points"]],
        "carried_global_update": carried["global_update"],
        "carried_cap": carried["cap"],
        "carried_provenance": dict(carried["provenance"]),
        "rng": rng_before.clone(),
    }
    try:
        result = train_run(
            objective=objective,
            provenance=provenance(),
            train_chunks=CHUNKS,
            tokenizer=None,
            corruption_policy=CorruptionRatePolicy(seed=CORRUPTION_SEED),
            truncation=TruncationPolicy.unbounded(),
            evaluate_fn=evaluate_fn,
            pad_token_id=1,
            cap=CAP,
            resume=carried,
            checkpoint_dir=tmp_path,
            execution=None,
            preparation_pool=pool,
            telemetry=sink,
            telemetry_identity={"stage": "lr_pilot", "label": "lr=0.0003"},
        )
    finally:
        Stage1Objective.forward = original_forward
        torch.optim.AdamW.step = original_step
        DeterministicSampler.next_batch = original_next

    adapter = objective.unmark_encoder.adapter
    # The checkpoint production actually wrote at update 20 000 -- real adapter
    # tensors, real AdamW moments, real sampler cursor, canonical points.
    written = load_training_checkpoint(tmp_path)
    return {
        "result": result,
        "counters": counters,
        "pool_calls": pool.calls,
        "adapter_state": {k: v.detach().clone() for k, v in adapter.state_dict().items()},
        "points": list(result.points),
        "cap": result.cap,
        "continued": result.continued,
        "budget_limited": result.budget_limited,
        "rng_before": rng_before,
        "rng_after": torch.random.get_rng_state(),
        "checkpoint": written,
        "initial": initial,
    }


# ---------------------------------------------------------------------------
# The equivalence proof
# ---------------------------------------------------------------------------

def execute(tmp_path, name, carried, prepared, *, telemetry: bool):
    """One execution, OFF or ON, from an independent deep copy of `carried`."""
    buffer = io.StringIO() if telemetry else None
    sink = JsonlSink(buffer) if telemetry else None
    outcome = run_once(tmp_path / name, sink=sink, prepared=prepared, carried=carried)
    return outcome, buffer


def canonical_start(tmp_path):
    """The one canonical starting state every execution in a test derives from."""
    return fixed_batch(), build_resume_payload(tmp_path / "seed")


def both_runs(tmp_path):
    """OFF and ON from the SAME real resume payload, so they start identical.

    The start-state equality is ASSERTED here rather than assumed -- that is the
    check whose absence let the Audit 042 fixture defect masquerade as a
    telemetry effect.
    """
    prepared, carried = canonical_start(tmp_path)
    off, _ = execute(tmp_path, "off", carried, prepared, telemetry=False)
    on, buffer = execute(tmp_path, "on", carried, prepared, telemetry=True)
    assert_same_start(off, on, "OFF vs ON")
    return off, on, buffer


def test_off_vs_off_is_reproducible(tmp_path):
    """OFF1 == OFF2. Without this, an OFF-vs-ON difference proves nothing.

    This is the counterfactual that identifies the Audit 042 root cause: under
    the old ordering the second execution built its frozen encoder from the RNG
    state the first had left behind, so even OFF vs OFF diverged.
    """
    prepared, carried = canonical_start(tmp_path)
    first, _ = execute(tmp_path, "off1", carried, prepared, telemetry=False)
    second, _ = execute(tmp_path, "off2", carried, prepared, telemetry=False)
    assert_same_start(first, second, "OFF vs OFF")
    assert_same_outcome(first, second, "OFF vs OFF")


def test_on_vs_on_is_reproducible(tmp_path):
    """ON1 == ON2. Telemetry itself must be deterministic."""
    prepared, carried = canonical_start(tmp_path)
    first, _ = execute(tmp_path, "on1", carried, prepared, telemetry=True)
    second, _ = execute(tmp_path, "on2", carried, prepared, telemetry=True)
    assert_same_start(first, second, "ON vs ON")
    assert_same_outcome(first, second, "ON vs ON")


def test_the_frozen_encoder_starts_identical_in_every_execution(tmp_path):
    """The exact tensor that diverged on CUDA.

    The encoder is restored from NO checkpoint -- v2 payloads are adapter-only
    by design -- so its start state depends entirely on the RNG at construction.
    """
    prepared, carried = canonical_start(tmp_path)
    off, _ = execute(tmp_path, "off", carried, prepared, telemetry=False)
    on, _ = execute(tmp_path, "on", carried, prepared, telemetry=True)
    assert same_tensor_state(off["initial"]["encoder"], on["initial"]["encoder"])
    assert off["initial"]["encoder"], "the stand-in encoder must carry real parameters"
    # And the adapter, which IS restored, matches the payload it came from.
    assert same_tensor_state(off["initial"]["adapter"], on["initial"]["adapter"])


def test_an_execution_does_not_mutate_the_canonical_starting_payload(tmp_path):
    """Running OFF must leave the payload ON will use untouched."""
    prepared, carried = canonical_start(tmp_path)
    before_adapter = clone_tensors(carried["adapter_state"])
    before = {
        "global_update": carried["global_update"],
        "cap": carried["cap"],
        "sampler_state": dict(carried["sampler_state"]),
        "points": [dict(p) for p in carried["points"]],
        "provenance": dict(carried["provenance"]),
    }
    execute(tmp_path, "off", carried, prepared, telemetry=False)
    execute(tmp_path, "on", carried, prepared, telemetry=True)

    assert same_tensor_state(before_adapter, carried["adapter_state"])
    assert carried["global_update"] == before["global_update"]
    assert carried["cap"] == before["cap"]
    assert carried["sampler_state"] == before["sampler_state"]
    assert [dict(p) for p in carried["points"]] == before["points"]
    assert dict(carried["provenance"]) == before["provenance"]


def test_the_fixture_resumes_under_the_real_locked_budget(tmp_path):
    """Guards the defect this file was repaired for.

    If someone reintroduces an arbitrary small cap, production raises
    `SelectionViolation` in `resolve_budget` and this test fails -- which is
    exactly what the first CUDA run did.
    """
    from unmark.stage1.protocol import EXTENDED_MAX_UPDATES

    assert CAP == INITIAL_MAX_UPDATES == 20_000
    assert CAP in (INITIAL_MAX_UPDATES, EXTENDED_MAX_UPDATES)
    assert START_UPDATE == 19_950 and UPDATES == 50
    carried = build_resume_payload(tmp_path / "seed")
    assert carried["cap"] == CAP
    assert carried["global_update"] == START_UPDATE
    assert any(p["update"] == 0 for p in carried["points"])


def test_telemetry_on_is_scientifically_identical_to_telemetry_off(tmp_path):
    off, on, buffer = both_runs(tmp_path)

    # Telemetry actually happened -- otherwise this test proves nothing.
    assert buffer.getvalue().count(PREFIX) > 0

    assert set(off["adapter_state"]) == set(on["adapter_state"])
    for name, tensor in off["adapter_state"].items():
        assert torch.equal(tensor, on["adapter_state"][name]), (
            f"adapter parameter {name} differs between telemetry OFF and ON"
        )

    assert off["result"].provenance.to_dict() == on["result"].provenance.to_dict()
    assert off["points"] == on["points"], "validation history differs"
    assert off["cap"] == on["cap"] == CAP
    assert off["continued"] == on["continued"] is False
    assert off["budget_limited"] == on["budget_limited"] is False
    assert off["result"].selected.to_dict() == on["result"].selected.to_dict()
    assert off["result"].selected.update == 0, (
        "update 0 is the best point in this fixture, so no continuation is due"
    )
    assert off["result"].to_dict() == on["result"].to_dict()
    # Same comparator the OFF/OFF and ON/ON reproducibility tests use, so the
    # three cells of the diagnostic matrix are held to one standard.
    assert_same_outcome(off, on, "OFF vs ON")


def test_the_written_checkpoint_is_scientifically_identical(tmp_path):
    """The REAL checkpoint production saved at update 20 000, compared field by
    field: adapter tensors, AdamW moments, sampler cursor, canonical points."""
    off, on, _ = both_runs(tmp_path)
    a, b = off["checkpoint"], on["checkpoint"]

    assert a["global_update"] == b["global_update"] == CAP
    assert a["cap"] == b["cap"] == CAP
    assert a["provenance"] == b["provenance"]
    assert a["sampler_state"] == b["sampler_state"], "sampler cursor differs"
    assert a["points"] == b["points"], "persisted validation history differs"

    assert set(a["adapter_state"]) == set(b["adapter_state"])
    for name, tensor in a["adapter_state"].items():
        assert torch.equal(tensor, b["adapter_state"][name]), name

    assert a["optimizer_state"]["param_groups"] == b["optimizer_state"]["param_groups"]
    moments_a, moments_b = a["optimizer_state"]["state"], b["optimizer_state"]["state"]
    assert set(moments_a) == set(moments_b)
    assert moments_a, "the optimizer must carry real moments after 50 updates"
    for key, entry in moments_a.items():
        for field, value in entry.items():
            other = moments_b[key][field]
            if torch.is_tensor(value):
                assert torch.equal(value, other), f"optimizer {key}.{field} differs"
            else:
                assert value == other, f"optimizer {key}.{field} differs"


def test_telemetry_adds_no_forward_backward_optimizer_sampling_or_evaluation(tmp_path):
    off, on, _ = both_runs(tmp_path)
    assert off["counters"].forward == on["counters"].forward == UPDATES
    assert off["counters"].step == on["counters"].step == UPDATES
    assert off["counters"].next_batch == on["counters"].next_batch == UPDATES
    assert off["pool_calls"] == on["pool_calls"] == UPDATES
    # One evaluation, at the 20 000 boundary. Update 0 was RESTORED from the
    # checkpoint, not re-measured, so it costs no evaluation.
    assert off["counters"].evaluate == on["counters"].evaluate == 1


def test_telemetry_consumes_no_torch_rng():
    """Emission must leave the torch RNG stream untouched."""
    torch.manual_seed(1234)
    before = torch.random.get_rng_state()
    sink = JsonlSink(io.StringIO())
    for update in range(500):
        sink.emit("train_progress", global_update=update, loss=0.5)
    sink.emit("validation", **point_for(0).to_dict())
    assert torch.equal(torch.random.get_rng_state(), before)


def test_the_run_leaves_the_same_rng_state_with_and_without_telemetry(tmp_path):
    off, on, _ = both_runs(tmp_path)
    assert torch.equal(off["rng_after"], on["rng_after"]), (
        "telemetry perturbed the torch RNG stream"
    )


# ---------------------------------------------------------------------------
# What telemetry actually reported
# ---------------------------------------------------------------------------
def test_the_emitted_stream_describes_the_real_run(tmp_path):
    import json

    off, on, buffer = both_runs(tmp_path)
    events = [json.loads(line[len(PREFIX):])
              for line in buffer.getvalue().splitlines() if line.startswith(PREFIX)]
    kinds = [e["event"] for e in events]

    # Cadence 50 from 19 950 lands exactly once, on the final update.
    progress = [e for e in events if e["event"] == "train_progress"]
    assert [e["global_update"] for e in progress] == [CAP], (
        f"expected one progress event at {CAP} under cadence {PROGRESS_EVERY_UPDATES}"
    )
    for event in progress:
        assert event["cap"] == CAP
        assert event["batch_size"] == BATCH_SIZE
        assert isinstance(event["loss"], float)
        assert event["stage"] == "lr_pilot" and event["label"] == "lr=0.0003"

    validations = [e for e in events if e["event"] == "validation"]
    assert [e["update"] for e in validations] == [CAP]
    for event in validations:
        assert set(event["distances"]) == set(VALIDATION_CONDITIONS)
        assert event["score"] == max(event["distances"].values())

    checkpoints = [e for e in events if e["event"] == "checkpoint"]
    assert [e["update"] for e in checkpoints] == [CAP]
    assert checkpoints[0]["checkpoint_name"] == CHECKPOINT_NAME

    assert "run_end" in kinds
    end = next(e for e in events if e["event"] == "run_end")
    assert end["global_update"] == CAP
    assert end["cap"] == on["cap"]
    assert end["selected_update"] == on["result"].selected.update

    # Ordering: the checkpoint event may only follow a successful save, and the
    # run ends last.
    assert kinds.index("validation") < kinds.index("checkpoint") < kinds.index("run_end")


def test_the_checkpoint_event_reports_state_that_is_really_on_disk(tmp_path):
    """A checkpoint event emitted before the save would name a file that does
    not yet hold this update."""
    import json

    _, on, buffer = both_runs(tmp_path)
    events = [json.loads(line[len(PREFIX):])
              for line in buffer.getvalue().splitlines() if line.startswith(PREFIX)]
    checkpoint = next(e for e in events if e["event"] == "checkpoint")
    assert on["checkpoint"]["global_update"] == checkpoint["update"] == CAP


def test_no_emitted_event_carries_corpus_text(tmp_path):
    """The prepared examples hold real text; none of it may reach telemetry."""
    import json

    prepared = fixed_batch()
    carried = build_resume_payload(tmp_path / "seed")
    buffer = io.StringIO()
    run_once(tmp_path / "on", sink=JsonlSink(buffer), prepared=prepared, carried=carried)
    raw = buffer.getvalue()
    for example in prepared[:8]:
        assert example.canonical_text not in raw
        assert example.corrupted_text not in raw
    for line in raw.splitlines():
        event = json.loads(line[len(PREFIX):])
        for key in event:
            assert "text" not in key
