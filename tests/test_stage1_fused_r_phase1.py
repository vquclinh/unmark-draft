"""Operational r-phase1 acceleration contracts.

The fused path must not become a second scientific protocol. It is allowed only
for the locked r-phase1 schedule, and it may share preparation work only after
proving active candidates share the same sampler stream.
"""

from __future__ import annotations

import pathlib
import sys
import types

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from unmark.stage1.contracts import Stage1ContractViolation  # noqa: E402
from unmark.stage1.fused import (  # noqa: E402
    R_PHASE1_EXECUTION_ENV,
    R_PHASE1_EXECUTION_FUSED,
    R_PHASE1_EXECUTION_SEQUENTIAL,
    require_fused_r_phase1_schedule,
    resolve_r_phase1_execution,
)
from unmark.stage1.protocol import (  # noqa: E402
    R_PHASE1_GRID,
    SELECTION_SEED,
    VALIDATION_CONDITIONS,
)
from unmark.stage1.selection import (  # noqa: E402
    PlannedRun,
    r_phase1_schedule,
)


def test_r_phase1_execution_mode_is_an_environment_only_operational_knob():
    assert resolve_r_phase1_execution({}) == R_PHASE1_EXECUTION_SEQUENTIAL
    assert (
        resolve_r_phase1_execution({R_PHASE1_EXECUTION_ENV: "fused"})
        == R_PHASE1_EXECUTION_FUSED
    )
    assert (
        resolve_r_phase1_execution({R_PHASE1_EXECUTION_ENV: "off"})
        == R_PHASE1_EXECUTION_SEQUENTIAL
    )
    with pytest.raises(Stage1ContractViolation, match=R_PHASE1_EXECUTION_ENV):
        resolve_r_phase1_execution({R_PHASE1_EXECUTION_ENV: "fast"})


def test_fused_r_phase1_accepts_only_the_locked_shared_seed_schedule():
    require_fused_r_phase1_schedule(r_phase1_schedule(SELECTION_SEED, 1e-4))

    split_seed = list(r_phase1_schedule(SELECTION_SEED, 1e-4))
    split_seed[0] = PlannedRun("r_phase1", "r=0.25", 1e-4, 0.25, 999)
    with pytest.raises(Stage1ContractViolation, match="shared selection seed"):
        require_fused_r_phase1_schedule(split_seed)

    split_lr = list(r_phase1_schedule(SELECTION_SEED, 1e-4))
    split_lr[0] = PlannedRun("r_phase1", "r=0.25", 3e-4, 0.25, SELECTION_SEED)
    with pytest.raises(Stage1ContractViolation, match="one frozen learning rate"):
        require_fused_r_phase1_schedule(split_lr)

    wrong_stage = list(r_phase1_schedule(SELECTION_SEED, 1e-4))
    wrong_stage[0] = PlannedRun("lr_pilot", "r=0.25", 1e-4, 0.25, SELECTION_SEED)
    with pytest.raises(Stage1ContractViolation, match="only defined for r_phase1"):
        require_fused_r_phase1_schedule(wrong_stage)

    short = list(r_phase1_schedule(SELECTION_SEED, 1e-4))[:-1]
    with pytest.raises(Stage1ContractViolation, match=str(len(R_PHASE1_GRID))):
        require_fused_r_phase1_schedule(short)


def test_execute_stage_exposes_no_new_scientific_cli_flags():
    source = pathlib.Path("scripts/stage1_runner.py").read_text(encoding="utf-8")
    for forbidden in (
        "--r-phase1-execution",
        "--fused",
        "--parallel-r",
        "--preparation-workers",
    ):
        assert forbidden not in source


def test_fused_r_phase1_shares_one_preparation_batch_per_update(tmp_path, monkeypatch):
    pytest.importorskip("torch", reason="production modules imported by fused path need torch")

    import unmark.modeling.adapter as adapter_module
    import unmark.stage1.data as data_module
    import unmark.stage1.fused as fused
    import unmark.stage1.initialisation as init_module
    import unmark.stage1.objective as objective_module
    import unmark.stage1.optim as optim_module

    monkeypatch.setattr(fused, "BATCH_SIZE", 2)
    monkeypatch.setattr(fused, "INITIAL_MAX_UPDATES", 2)
    monkeypatch.setattr(fused, "EVAL_EVERY_UPDATES", 1)
    monkeypatch.setattr(fused, "CHECKPOINT_EVERY_UPDATES", 1)
    monkeypatch.setattr(fused, "load_training_checkpoint", lambda path: None)
    monkeypatch.setattr(fused, "resolve_budget", lambda result: result)
    monkeypatch.setattr(
        fused,
        "require_optimizer_parameter_identity",
        lambda optimizer, adapter: None,
    )
    monkeypatch.setattr(
        fused,
        "require_optimizer_state_device",
        lambda optimizer, device: None,
    )
    monkeypatch.setattr(
        fused,
        "verify_model_contract",
        lambda unmark_encoder: {"trainable_parameters": 1},
    )
    monkeypatch.setattr(
        fused,
        "gradient_report",
        lambda unmark_encoder: {
            "adapter_group_grad_norms": {
                "adapter.tone_embedding.weight": 1.0,
                "adapter.letter_embedding.weight": 1.0,
            },
            "encoder_grad_tensors": 0,
        },
    )
    monkeypatch.setattr(
        fused,
        "save_training_checkpoint",
        lambda directory, payload, is_best=False: pathlib.Path(directory) / "ckpt.pt",
    )
    monkeypatch.setattr(
        "unmark.stage1.execute.require_frozen_backbone_unchanged",
        lambda encoder, expected_hash, label: None,
    )

    class FakeScalar:
        def __init__(self, value):
            self.value = float(value)

        def backward(self):
            return None

        def detach(self):
            return self

        def __float__(self):
            return self.value

    class FakeLossResult:
        def __init__(self, value):
            self.loss = FakeScalar(value)
            self.loss_align = FakeScalar(value / 2.0)
            self.loss_clean = FakeScalar(value / 2.0)

    class FakeParam:
        requires_grad = True

    class FakeAdapter:
        def __init__(self, init_seed):
            self.init_seed = init_seed

        def to(self, device):
            return self

        def state_dict(self):
            return {"init_seed": self.init_seed}

        def load_state_dict(self, state, strict=True):
            self.init_seed = state["init_seed"]

        def named_parameters(self):
            return [("weight", FakeParam())]

    class FakeUnmarkEncoder:
        def __init__(self, encoder, adapter):
            self.encoder = encoder
            self.adapter = adapter

        def named_parameters(self):
            return [("adapter.weight", FakeParam())]

    objectives = []

    class FakeObjective:
        def __init__(self, unmark_encoder, weights):
            self.unmark_encoder = unmark_encoder
            self.weights = weights
            self.calls = []
            objectives.append(self)

        def train(self, mode=True):
            return self

        def __call__(self, batch):
            self.calls.append(tuple(batch["sample_ids"]))
            return FakeLossResult(len(self.calls))

    class FakeOptimizer:
        def __init__(self):
            self.steps = 0

        def zero_grad(self, set_to_none=True):
            return None

        def step(self):
            self.steps += 1

        def state_dict(self):
            return {"steps": self.steps}

        def load_state_dict(self, state):
            self.steps = state["steps"]

    class FakePrepared:
        channels_differ = True
        letter_channels_differ = True

        def __init__(self, sample_id):
            self.sample_id = sample_id

    class FakePool:
        def __init__(self):
            self.calls = []

        def prepare(self, tasks):
            self.calls.append(tuple((chunk_id, visit) for chunk_id, visit, _ in tasks))
            return [FakePrepared(chunk_id) for chunk_id, _visit, _text in tasks]

    def fake_point(update):
        score = 1.0 + update / 10.0
        return fused.ValidationPoint(
            update=update,
            distances={c: score for c in VALIDATION_CONDITIONS},
            d_clean=score,
        )

    monkeypatch.setattr(adapter_module, "UnmarkEncoder", FakeUnmarkEncoder)
    monkeypatch.setattr(objective_module, "Stage1Objective", FakeObjective)
    monkeypatch.setattr(optim_module, "build_optimizer", lambda named, lr: FakeOptimizer())
    monkeypatch.setattr(init_module, "fresh_adapter", lambda hidden, seed: FakeAdapter(seed))
    monkeypatch.setattr(init_module, "trainable_state", lambda adapter: adapter.state_dict())
    monkeypatch.setattr(init_module, "trainable_state_hash", lambda state: "fresh")
    monkeypatch.setattr(init_module, "expected_fresh_init_hash", lambda hidden, seed: "fresh")
    monkeypatch.setattr(
        data_module,
        "collate_stage1_batch",
        lambda prepared, pad: {"sample_ids": [item.sample_id for item in prepared]},
    )
    monkeypatch.setattr(data_module, "batch_to_device", lambda batch, device: batch)
    monkeypatch.setattr(data_module, "module_device", lambda objective: "device")
    monkeypatch.setattr("unmark.stage1.validation.evaluate", lambda *a, **k: fake_point(0))

    pool = FakePool()
    candidates = fused.train_fused_r_phase1(
        schedule=r_phase1_schedule(SELECTION_SEED, 1e-4),
        train_chunks={f"doc-{i}": f"text {i}" for i in range(6)},
        tokenizer=types.SimpleNamespace(),
        frozen_encoder=object(),
        hidden_size=8,
        encoder_state_hash="h",
        prepared_by_condition={},
        pad_token_id=1,
        device="device",
        execution=None,
        manifest_digest="d" * 64,
        repository_head="a" * 40,
        inventory=None,
        output_dir=tmp_path,
        preparation_pool=pool,
        resume=False,
        telemetry=types.SimpleNamespace(emit=lambda *a, **k: None),
    )

    assert len(candidates) == len(R_PHASE1_GRID)
    assert len(pool.calls) == 2, "preparation must be shared per update, not per r"
    assert len(objectives) == len(R_PHASE1_GRID)
    first_stream = objectives[0].calls
    assert all(obj.calls == first_stream for obj in objectives)
    for planned in r_phase1_schedule(SELECTION_SEED, 1e-4):
        assert (tmp_path / f"run-{planned.label.replace('=', '')}.json").is_file()
