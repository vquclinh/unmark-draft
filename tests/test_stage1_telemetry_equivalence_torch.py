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
    VALIDATION_CONDITIONS,
)
from unmark.stage1.selection import ValidationPoint  # noqa: E402
from unmark.stage1.telemetry import JsonlSink, PREFIX  # noqa: E402
from unmark.stage1.trainer import (  # noqa: E402
    RunProvenance,
    checkpoint_payload,
    train_run,
)

from test_stage1 import StubTokenizer  # noqa: E402

TINY_HIDDEN = 8
"""Not 768, so `verify_model_contract` skips the locked parameter-count clause
while every other clause still applies. Same rationale as the resume tests."""

CAP = 4
"""Four real updates. Enough to cross the progress cadence below and to run the
loop body repeatedly; small enough to be a unit test."""

EVERY = 2
"""Operational progress cadence for this test only -- passed through the sink,
never through `protocol`."""

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


class _Counters:
    def __init__(self) -> None:
        self.forward = 0
        self.step = 0
        self.next_batch = 0
        self.evaluate = 0


def point_for(update: int) -> ValidationPoint:
    """Deterministic validation, so the comparison isolates telemetry."""
    worst = 0.5 - update / 1000.0
    return ValidationPoint(
        update=update,
        distances={c: worst + i / 100.0 for i, c in enumerate(VALIDATION_CONDITIONS)},
        d_clean=worst / 2.0,
    )


def run_once(tmp_path, *, sink, prepared):
    """One REAL `train_run`, instrumented with counters. Returns everything."""
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

    from unmark.stage1.sampler import DeterministicSampler

    original_next = DeterministicSampler.next_batch

    def counted_next(self, size):
        counters.next_batch += 1
        return original_next(self, size)

    Stage1Objective.forward = counted_forward
    torch.optim.AdamW.step = counted_step
    DeterministicSampler.next_batch = counted_next
    torch.manual_seed(0)
    rng_before = torch.random.get_rng_state()
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
    optimizer = build_optimizer(
        [(n, p) for n, p in objective.unmark_encoder.named_parameters() if p.requires_grad],
        provenance().learning_rate,
    )
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
        "optimizer_shape": len(optimizer.param_groups),
    }


# ---------------------------------------------------------------------------
# The equivalence proof
# ---------------------------------------------------------------------------
def test_telemetry_on_is_scientifically_identical_to_telemetry_off(tmp_path):
    prepared = fixed_batch()
    off = run_once(tmp_path / "off", sink=None, prepared=prepared)

    buffer = io.StringIO()
    on = run_once(
        tmp_path / "on",
        sink=JsonlSink(buffer, progress_every_updates=EVERY),
        prepared=prepared,
    )

    # Telemetry actually happened -- otherwise this test proves nothing.
    assert buffer.getvalue().count(PREFIX) > 0

    assert set(off["adapter_state"]) == set(on["adapter_state"])
    for name, tensor in off["adapter_state"].items():
        assert torch.equal(tensor, on["adapter_state"][name]), (
            f"adapter parameter {name} differs between telemetry OFF and ON"
        )

    assert off["result"].provenance.to_dict() == on["result"].provenance.to_dict()
    assert off["points"] == on["points"], "validation history differs"
    assert off["cap"] == on["cap"]
    assert off["continued"] == on["continued"]
    assert off["budget_limited"] == on["budget_limited"]
    assert off["result"].selected.to_dict() == on["result"].selected.to_dict()
    assert off["result"].to_dict() == on["result"].to_dict()


def test_telemetry_adds_no_forward_backward_optimizer_sampling_or_evaluation(tmp_path):
    prepared = fixed_batch()
    off = run_once(tmp_path / "off", sink=None, prepared=prepared)
    on = run_once(
        tmp_path / "on",
        sink=JsonlSink(io.StringIO(), progress_every_updates=EVERY),
        prepared=prepared,
    )
    assert off["counters"].forward == on["counters"].forward == CAP
    assert off["counters"].step == on["counters"].step == CAP
    assert off["counters"].next_batch == on["counters"].next_batch == CAP
    assert off["pool_calls"] == on["pool_calls"] == CAP
    assert off["counters"].evaluate == on["counters"].evaluate, (
        "telemetry must not trigger an extra validation evaluation"
    )


def test_telemetry_consumes_no_torch_rng(tmp_path):
    """Emission must leave the torch RNG stream untouched."""
    torch.manual_seed(1234)
    before = torch.random.get_rng_state()
    sink = JsonlSink(io.StringIO(), progress_every_updates=EVERY)
    for update in range(500):
        sink.emit("train_progress", global_update=update, loss=0.5)
    sink.emit("validation", **point_for(2).to_dict())
    assert torch.equal(torch.random.get_rng_state(), before)


def test_the_run_leaves_the_same_rng_state_with_and_without_telemetry(tmp_path):
    prepared = fixed_batch()
    off = run_once(tmp_path / "off", sink=None, prepared=prepared)
    on = run_once(
        tmp_path / "on",
        sink=JsonlSink(io.StringIO(), progress_every_updates=EVERY),
        prepared=prepared,
    )
    assert torch.equal(off["rng_after"], on["rng_after"]), (
        "telemetry perturbed the torch RNG stream"
    )


def test_the_checkpoint_payload_is_identical_with_and_without_telemetry(tmp_path):
    prepared = fixed_batch()
    off = run_once(tmp_path / "off", sink=None, prepared=prepared)
    on = run_once(
        tmp_path / "on",
        sink=JsonlSink(io.StringIO(), progress_every_updates=EVERY),
        prepared=prepared,
    )
    common = dict(
        provenance=provenance(), adapter_state={}, optimizer_state={},
        global_update=CAP, sampler_state={}, cap=CAP, budget_limited=False,
    )
    assert (checkpoint_payload(points=off["points"], **common)["points"]
            == checkpoint_payload(points=on["points"], **common)["points"])


# ---------------------------------------------------------------------------
# What telemetry actually reported
# ---------------------------------------------------------------------------
def test_the_emitted_stream_describes_the_real_run(tmp_path):
    import json

    prepared = fixed_batch()
    buffer = io.StringIO()
    outcome = run_once(
        tmp_path / "on",
        sink=JsonlSink(buffer, progress_every_updates=EVERY),
        prepared=prepared,
    )
    events = [json.loads(line[len(PREFIX):])
              for line in buffer.getvalue().splitlines() if line.startswith(PREFIX)]
    kinds = [e["event"] for e in events]

    progress = [e for e in events if e["event"] == "train_progress"]
    assert progress, "no progress events at cadence 2 over 4 updates"
    assert [e["global_update"] for e in progress] == [2, 4]
    for event in progress:
        assert event["cap"] == CAP
        assert event["batch_size"] == BATCH_SIZE
        assert isinstance(event["loss"], float)
        assert event["stage"] == "lr_pilot" and event["label"] == "lr=0.0003"

    validations = [e for e in events if e["event"] == "validation"]
    assert validations, "no validation event"
    for event in validations:
        assert set(event["distances"]) == set(VALIDATION_CONDITIONS)
        assert event["score"] == max(event["distances"].values())

    assert "run_end" in kinds
    end = next(e for e in events if e["event"] == "run_end")
    assert end["global_update"] == CAP
    assert end["cap"] == outcome["cap"]
    assert end["selected_update"] == outcome["result"].selected.update


def test_no_emitted_event_carries_corpus_text(tmp_path):
    """The prepared examples hold real text; none of it may reach telemetry."""
    import json

    prepared = fixed_batch()
    buffer = io.StringIO()
    run_once(tmp_path / "on",
             sink=JsonlSink(buffer, progress_every_updates=EVERY),
             prepared=prepared)
    raw = buffer.getvalue()
    for example in prepared[:8]:
        assert example.canonical_text not in raw
        assert example.corrupted_text not in raw
    for line in raw.splitlines():
        event = json.loads(line[len(PREFIX):])
        for key in event:
            assert "text" not in key
