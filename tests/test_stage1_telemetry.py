"""Stage-1 telemetry: schema, safety, and the monitor's parsing. **Torch-free.**

Audit 040. Telemetry is OPERATIONAL: it must make a silent 20-minute
preprocessing phase observable without touching a single scientific decision.
These tests hold it to that.

The scientific-equivalence proof through the real `train_run` lives in
`test_stage1_telemetry_equivalence_torch.py`, in a separate FILE because a
module-level `pytest.importorskip` would otherwise skip everything here too.
"""

from __future__ import annotations

import io
import json
import pathlib
import random
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from unmark.stage1 import telemetry as telemetry_module  # noqa: E402
from unmark.stage1.protocol import (  # noqa: E402
    BATCH_SIZE,
    EVAL_EVERY_UPDATES,
    VALIDATION_CONDITIONS,
)
from unmark.stage1.selection import ValidationPoint  # noqa: E402
from unmark.stage1.telemetry import (  # noqa: E402
    PREFIX,
    PROGRESS_EVERY_UPDATES,
    SCHEMA,
    JsonlSink,
    NullSink,
    TelemetrySink,
    phase,
    sink_from_environment,
)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from stage1_wandb_monitor import (  # noqa: E402
    CAMPAIGN_PROVENANCE_KEYS,
    SAFE_CONFIG_KEYS,
    MonitorState,
    candidate_run_name,
    derive_pass_metrics,
    parse_line,
    safe_config,
)


def sink_and_buffer() -> tuple[JsonlSink, io.StringIO]:
    buffer = io.StringIO()
    return JsonlSink(buffer), buffer


def events_in(buffer: io.StringIO) -> list[dict]:
    out = []
    for line in buffer.getvalue().splitlines():
        assert line.startswith(PREFIX), line
        out.append(json.loads(line[len(PREFIX):]))
    return out


# ---------------------------------------------------------------------------
# The default is silence
# ---------------------------------------------------------------------------
def test_the_default_sink_emits_nothing_and_is_disabled():
    """A caller that does not opt in gets the pre-telemetry code path."""
    sink = NullSink()
    assert sink.enabled is False
    assert sink.progress_every() == 0
    sink.emit("train_progress", global_update=1)  # must be a no-op, not an error


def test_environment_opt_in():
    assert isinstance(sink_from_environment({}), NullSink)
    assert isinstance(sink_from_environment({"UNMARK_TELEMETRY": "0"}), NullSink)
    for value in ("1", "true", "yes", "on"):
        assert isinstance(sink_from_environment({"UNMARK_TELEMETRY": value}), JsonlSink)


def test_the_base_sink_is_a_silent_no_op():
    TelemetrySink().emit("anything", a=1)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
def test_every_event_carries_the_versioned_schema_envelope():
    sink, buffer = sink_and_buffer()
    sink.emit("run_start", stage="lr_pilot", lr=1e-4)
    (event,) = events_in(buffer)
    for key in ("schema", "event", "seq", "wall_clock", "elapsed_s"):
        assert key in event, key
    assert event["schema"] == SCHEMA == "stage1-telemetry-v1"
    assert event["event"] == "run_start"
    assert isinstance(event["seq"], int) and event["seq"] == 1
    assert isinstance(event["elapsed_s"], float)
    assert event["stage"] == "lr_pilot" and event["lr"] == 1e-4


def test_sequence_numbers_are_monotonic():
    sink, buffer = sink_and_buffer()
    for i in range(5):
        sink.emit("train_progress", global_update=i)
    assert [e["seq"] for e in events_in(buffer)] == [1, 2, 3, 4, 5]


def test_phase_emits_start_and_done_with_elapsed():
    sink, buffer = sink_and_buffer()
    with phase(sink, "corpus_load", prepared_corpus="/x/y"):
        pass
    start, done = events_in(buffer)
    assert start["event"] == done["event"] == "stage_phase"
    assert (start["state"], done["state"]) == ("START", "DONE")
    assert start["phase"] == done["phase"] == "corpus_load"
    assert isinstance(done["elapsed_phase_s"], float)


def test_a_failing_phase_is_reported_and_the_error_still_propagates():
    """Fail-closed behaviour must be exactly what it was."""
    sink, buffer = sink_and_buffer()
    with pytest.raises(ValueError, match="boom"):
        with phase(sink, "backbone_load"):
            raise ValueError("boom")
    failed = events_in(buffer)[-1]
    assert failed["state"] == "FAILED"
    assert failed["error_type"] == "ValueError"


def test_a_null_sink_phase_is_transparent():
    with phase(NullSink(), "corpus_load"):
        pass
    with pytest.raises(ValueError):
        with phase(NullSink(), "corpus_load"):
            raise ValueError


# ---------------------------------------------------------------------------
# Telemetry must never break training
# ---------------------------------------------------------------------------
def test_a_broken_stream_cannot_raise_into_the_scientific_path():
    class Exploding(io.StringIO):
        def write(self, *args, **kwargs):
            raise OSError("broken pipe")

    JsonlSink(Exploding()).emit("train_progress", global_update=1)  # must not raise


def test_an_unserialisable_value_degrades_instead_of_raising():
    sink, buffer = sink_and_buffer()
    sink.emit("train_progress", weird=object())
    (event,) = events_in(buffer)
    assert isinstance(event["weird"], str)


def test_non_finite_values_are_visible_rather_than_fatal():
    sink, buffer = sink_and_buffer()
    sink.emit("train_progress", loss=float("nan"), other=float("inf"))
    (event,) = events_in(buffer)
    assert event["loss"] == "nan" and event["other"] == "inf"


# ---------------------------------------------------------------------------
# RNG equivalence -- telemetry consumes ZERO randomness
# ---------------------------------------------------------------------------
def test_emission_consumes_no_python_rng():
    random.seed(12345)
    before = random.getstate()
    sink, _ = sink_and_buffer()
    for i in range(200):
        sink.emit("train_progress", global_update=i, loss=0.5)
    with phase(sink, "corpus_load"):
        pass
    assert random.getstate() == before


def test_emission_does_not_perturb_a_random_sequence():
    random.seed(7)
    expected = [random.random() for _ in range(20)]
    random.seed(7)
    sink, _ = sink_and_buffer()
    observed = []
    for _ in range(20):
        sink.emit("train_progress", noise=1)
        observed.append(random.random())
    assert observed == expected


def _imported_modules(path: str) -> set[str]:
    """Top-level module names actually IMPORTED by a file.

    AST, not substring matching: an earlier draft of this test grepped for
    "wandb" and failed on this module's own docstring, which merely *explains*
    that wandb lives elsewhere. That is the recurring defect class in this
    repository -- a test matching its own prose -- so the check reads real
    import statements instead.
    """
    import ast

    tree = ast.parse(pathlib.Path(path).read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_the_module_imports_nothing_beyond_the_standard_library():
    """The scientific process must never depend on a monitoring package."""
    imported = _imported_modules(telemetry_module.__file__)
    for forbidden in ("wandb", "psutil", "rich", "tqdm", "requests", "urllib3",
                      "socket", "torch", "numpy"):
        assert forbidden not in imported, f"telemetry must not import {forbidden}"
    assert imported <= {"json", "os", "sys", "time", "contextlib", "typing", "__future__"}


# ---------------------------------------------------------------------------
# The cadence is operational, not scientific
# ---------------------------------------------------------------------------
def test_the_progress_cadence_is_not_a_protocol_constant():
    import unmark.stage1.protocol as protocol

    assert PROGRESS_EVERY_UPDATES == 50
    assert not hasattr(protocol, "PROGRESS_EVERY_UPDATES")
    # It must not be able to move the locked cadences.
    assert PROGRESS_EVERY_UPDATES != EVAL_EVERY_UPDATES
    assert EVAL_EVERY_UPDATES == 500


def test_the_cadence_divides_the_locked_eval_cadence():
    """So a progress line always lands on the evaluation boundary too."""
    assert EVAL_EVERY_UPDATES % PROGRESS_EVERY_UPDATES == 0


# ---------------------------------------------------------------------------
# Raw-text leakage
# ---------------------------------------------------------------------------
def test_long_strings_are_truncated_so_corpus_text_cannot_leak():
    sink, buffer = sink_and_buffer()
    sink.emit("train_progress", oops="x" * 5000)
    (event,) = events_in(buffer)
    assert len(event["oops"]) <= telemetry_module.MAX_STRING + len("...[truncated]")
    assert event["oops"].endswith("...[truncated]")


def test_no_production_telemetry_call_site_passes_corpus_text():
    """Every emitted field in production is a scalar, digest, label or path."""
    import ast

    for name in ("unmark/stage1/execute.py", "unmark/stage1/trainer.py"):
        tree = ast.parse(pathlib.Path(name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "emit"):
                continue
            for keyword in node.keywords:
                if keyword.arg is None:
                    continue
                assert "text" not in keyword.arg, (
                    f"{name}: telemetry field {keyword.arg!r} may carry corpus text"
                )
                assert keyword.arg not in ("chunks", "documents", "examples", "batch")


# ---------------------------------------------------------------------------
# ValidationPoint uses the canonical serialisation -- no second score
# ---------------------------------------------------------------------------
def test_validation_telemetry_uses_the_canonical_point_serialisation():
    point = ValidationPoint(
        update=500,
        distances={c: 0.4 + i / 100 for i, c in enumerate(VALIDATION_CONDITIONS)},
        d_clean=0.2,
    )
    sink, buffer = sink_and_buffer()
    sink.emit("validation", cap=20_000, **point.to_dict())
    (event,) = events_in(buffer)
    assert event["update"] == 500
    assert set(event["distances"]) == set(VALIDATION_CONDITIONS)
    # The score is the production-derived one, not recomputed here.
    assert event["score"] == point.score == max(point.distances.values())


def test_the_trainer_does_not_reimplement_the_score():
    import ast

    source = pathlib.Path("unmark/stage1/trainer.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "emit"):
            for keyword in node.keywords:
                if keyword.arg == "score":
                    raise AssertionError(
                        "telemetry must not compute a score; it emits point.to_dict()"
                    )


def test_the_validation_event_is_emitted_via_to_dict_unpacking():
    import ast

    tree = ast.parse(pathlib.Path("unmark/stage1/trainer.py").read_text(encoding="utf-8"))
    found = False
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "emit" and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == "validation"):
            starred = [k for k in node.keywords if k.arg is None]
            assert starred, "validation telemetry must unpack point.to_dict()"
            found = True
    assert found, "no validation telemetry call found"


# ---------------------------------------------------------------------------
# Checkpoint event ordering: only after a successful save
# ---------------------------------------------------------------------------
def test_the_checkpoint_event_follows_a_successful_save():
    """Structural: the emit uses the value `save_training_checkpoint` returned."""
    import ast

    tree = ast.parse(pathlib.Path("unmark/stage1/trainer.py").read_text(encoding="utf-8"))
    train_run = next(n for n in ast.walk(tree)
                     if isinstance(n, ast.FunctionDef) and n.name == "train_run")
    body = ast.unparse(train_run)
    save_at = body.index("save_training_checkpoint(")
    emit_at = body.index("'checkpoint'")
    assert save_at < emit_at, (
        "the checkpoint event must be emitted AFTER the save, never before"
    )
    assert "published = save_training_checkpoint(" in body, (
        "the event should report the path the save actually published"
    )


# ---------------------------------------------------------------------------
# Derived corpus-pass-equivalent semantics
# ---------------------------------------------------------------------------
def test_pass_metrics_are_sample_visit_equivalents():
    metrics = derive_pass_metrics(global_update=6750, batch_size=128, train_chunks=2_621_624)
    assert metrics["sample_visits_total"] == 6750 * 128 == 864_000
    assert metrics["corpus_pass_equivalent"] == pytest.approx(864_000 / 2_621_624)
    assert metrics["pass_index"] == 1
    assert metrics["sample_visits_in_current_pass"] == 864_000
    assert metrics["pass_percent"] == pytest.approx(100 * 864_000 / 2_621_624)
    assert metrics["pass_percent"] == pytest.approx(32.96, abs=0.01)


def test_pass_index_advances_after_a_full_corpus_pass():
    chunks = 2_621_624
    updates = (chunks // BATCH_SIZE) + 1
    metrics = derive_pass_metrics(updates, BATCH_SIZE, chunks)
    assert metrics["corpus_pass_equivalent"] > 1.0
    assert metrics["pass_index"] == 2
    assert metrics["sample_visits_in_current_pass"] < chunks


def test_pass_metrics_degrade_without_a_denominator():
    metrics = derive_pass_metrics(10, 128, 0)
    assert metrics["sample_visits_total"] == 1280
    assert metrics["corpus_pass_equivalent"] is None


# ---------------------------------------------------------------------------
# Monitor parsing
# ---------------------------------------------------------------------------
def test_the_monitor_ignores_ordinary_prose_and_foreign_schemas():
    assert parse_line("frozen backbone VERIFIED on cuda: ...") is None
    assert parse_line(PREFIX + "{not json") is None
    assert parse_line(PREFIX + json.dumps({"schema": "stage1-telemetry-v99"})) is None
    assert parse_line(PREFIX + json.dumps({"schema": SCHEMA, "event": "x"}))["event"] == "x"


CAMPAIGN = {
    "repository_head": "a" * 40,
    "protocol_version": "stage1-protocol-v1",
    "corpus_manifest_digest":
        "250859a57d745675c5dba2c7a35df08ccc123988bece873b0c9b29c6e78413d6",
    "corpus_dataset": "undertheseanlp/UVW-2026",
    "corpus_revision": "a0a79294e4568137e25828bb3f2a4cde8546e1fb",
    "encoder_checkpoint": "vinai/phobert-base",
    "encoder_revision": "01daacda68afe13d83023d16ec647239e344a1e6",
    "precision": "fp32",
    "inventory_source_name": "all-vietnamese-syllables.txt",
    "inventory_source_revision": "135a4d9716e49a981624474156d6f247b9b46f6a",
    "inventory_sha256":
        "78eeb840d50455b14bd564da5aed7318d96468b8deaad5986b77bf5c538315d2",
}
"""The exact safe provenance a dashboard must carry, as production emits it."""


def synthetic_stream() -> list[dict]:
    """A recorded-shape lr-pilot stream: phases, three candidates, validation."""
    stream = [
        {"event": "stage_start", "stage": "lr_pilot", "candidate_count": 3},
        {"event": "stage_phase", "phase": "corpus_load", "state": "START"},
        {"event": "stage_phase", "phase": "corpus_load", "state": "DONE",
         "elapsed_phase_s": 91.2},
        {"event": "corpus_loaded", "train_chunks": 2_621_624, "dev_chunks": 11_443},
        {"event": "campaign_identity", **CAMPAIGN},
    ]
    for index, lr in enumerate((1e-4, 3e-4, 1e-3), start=1):
        stream.append({
            "event": "run_start", "stage": "lr_pilot", "label": f"lr={lr:g}",
            "candidate_index": index, "candidate_count": 3, "lr": lr, "r": 1.0,
            "seed": 21230, "initial_global_update": 0, "cap": 20_000,
            "batch_size": 128, "train_chunks": 2_621_624,
            "repository_head": "a" * 40, "protocol_version": "stage1-protocol-v1",
        })
        stream.append({"event": "train_progress", "stage": "lr_pilot", "label": f"lr={lr:g}",
                       "global_update": 6750, "cap": 20_000, "loss": 0.62,
                       "loss_align": 0.31, "loss_clean": 0.31})
        stream.append({"event": "validation", "update": 7000, "cap": 20_000,
                       "distances": {"FULL": 0.4, "P50": 0.5, "P100": 0.6, "STRIP_ALL": 0.7},
                       "d_clean": 0.2, "score": 0.7})
        stream.append({"event": "checkpoint", "update": 7000, "cap": 20_000,
                       "is_best": True, "checkpoint_name": "training-checkpoint-last.pt"})
    return [dict(e, schema=SCHEMA) for e in stream]


def test_the_monitor_reconstructs_candidate_progress_and_derived_metrics():
    state = MonitorState()
    for event in synthetic_stream():
        state.consume(event)

    assert state.candidate["candidate_index"] == 3
    assert state.candidate["candidate_count"] == 3
    assert state.candidate["lr"] == 1e-3
    assert state.candidate_key == "lr_pilot/lr=0.001/seed-21230"
    assert state.train_chunks == 2_621_624
    assert state.global_update == 6750
    assert state.cap == 20_000
    assert state.run_fraction() == pytest.approx(0.3375)

    metrics = state.pass_metrics()
    assert metrics["sample_visits_total"] == 864_000
    assert metrics["pass_percent"] == pytest.approx(32.96, abs=0.01)

    assert state.last_validation["score"] == 0.7
    assert state.last_validation["distances"]["STRIP_ALL"] == 0.7
    assert state.latest_checkpoint_update == 7000
    assert state.checkpoints == 1, "per-candidate counters reset at run_start"
    assert state.next_eval_update() == 7000


def test_run_start_resets_per_candidate_state():
    state = MonitorState()
    for event in synthetic_stream():
        state.consume(event)
        if event["event"] == "run_start":
            assert state.validations == []
            assert state.checkpoints == 0
            assert state.latest_checkpoint_update is None


def test_phase_tracking_makes_a_silent_phase_visible():
    state = MonitorState()
    state.consume({"schema": SCHEMA, "event": "stage_phase",
                   "phase": "validation_batch_build", "state": "START"})
    assert state.phase == "validation_batch_build"
    assert state.phase_started is not None
    state.consume({"schema": SCHEMA, "event": "stage_phase",
                   "phase": "validation_batch_build", "state": "DONE"})
    assert state.phase is None


def test_candidate_run_names_are_deterministic_and_derived():
    event = {"stage": "lr_pilot", "label": "lr=0.0001", "seed": 21230}
    assert candidate_run_name(event) == "lr-pilot-lr-0.0001-seed-21230"
    assert candidate_run_name(event) == candidate_run_name(dict(event))
    other = {"stage": "r_phase1", "label": "r=0.5", "seed": 21230}
    assert candidate_run_name(other) == "r-phase1-r-0.5-seed-21230"


def test_diagnostics_are_neutral_and_never_claim_overfitting():
    state = MonitorState()
    for event in synthetic_stream():
        state.consume(event)
    state.consume({"schema": SCHEMA, "event": "train_progress",
                   "global_update": 6800, "cap": 20_000, "loss": 0.40})
    state.consume({"schema": SCHEMA, "event": "validation", "update": 7500,
                   "distances": {"FULL": 0.3, "P50": 0.3, "P100": 0.3, "STRIP_ALL": 0.3},
                   "d_clean": 0.1, "score": 0.3})
    diagnostics = state.diagnostics()
    assert set(diagnostics) <= {"train_trend", "validation_trend", "divergence_watch"}
    for key in diagnostics:
        assert "overfit" not in key


# ---------------------------------------------------------------------------
# W&B privacy and isolation
# ---------------------------------------------------------------------------
def test_only_safe_provenance_reaches_the_dashboard_config():
    # Selected by NAME, not by index: inserting an event upstream must not
    # silently change which one this test inspects.
    event = dict(next(e for e in synthetic_stream() if e["event"] == "run_start"))
    event["secret_text"] = "xin chao, this is corpus content"
    config = safe_config(event)
    assert "secret_text" not in config
    assert set(config) <= set(SAFE_CONFIG_KEYS)
    assert config["repository_head"] == "a" * 40
    assert config["lr"] == 1e-4 and config["seed"] == 21230
    # No VALUE may be free text. `train_chunks` is a legitimate key -- it is a
    # count, not content -- so the guard is on what is carried, not on names.
    for key, value in config.items():
        assert not (isinstance(value, str) and len(value) > 64), (
            f"{key} carries a long string; only scalars, digests and labels may "
            "reach a dashboard"
        )
    assert config["train_chunks"] == 2_621_624


def test_the_scientific_process_never_imports_wandb():
    """The isolation that makes a dashboard outage harmless.

    Checked on real imports. Prose that *mentions* the monitor is fine and
    expected -- what must never happen is a scientific module depending on it.
    """
    for name in ("unmark/stage1/telemetry.py", "unmark/stage1/trainer.py",
                 "unmark/stage1/execute.py", "scripts/stage1_runner.py"):
        imported = _imported_modules(name)
        assert "wandb" not in imported, f"{name} must not import wandb"
        assert "stage1_wandb_monitor" not in imported, (
            f"{name} must not import the external monitor"
        )
        assert "psutil" not in imported, f"{name} must not import psutil"


def test_wandb_is_not_in_the_scientific_requirements():
    experiment = pathlib.Path("requirements/experiment.txt").read_text(encoding="utf-8")
    assert "wandb" not in experiment
    monitoring = pathlib.Path("requirements/monitoring.txt").read_text(encoding="utf-8")
    assert "wandb" in monitoring
    assert "torch" not in monitoring.replace("# No torch here", "")


def test_the_wandb_bridge_is_a_no_op_without_wandb(tmp_path):
    """A missing dashboard degrades to console-only, never to a failure."""
    from stage1_wandb_monitor import WandbBridge

    bridge = WandbBridge("unmark-stage1", "lr-pilot", tmp_path, enabled=False)
    assert bridge.start_candidate({"stage": "lr_pilot", "label": "lr=1e-4", "seed": 1}, {}) is None
    bridge.log({"train/loss": 0.5})
    bridge.finish()


def test_wandb_run_ids_persist_so_a_restart_resumes_the_same_run(tmp_path):
    """Colab restart / scientific resume must not create a duplicate run."""
    from stage1_wandb_monitor import WandbBridge

    bridge = WandbBridge("unmark-stage1", "lr-pilot", tmp_path)
    bridge._remember("lr_pilot/lr=0.0001/seed-21230", "abc123")  # noqa: SLF001
    reopened = WandbBridge("unmark-stage1", "lr-pilot", tmp_path)
    assert reopened._ids()["lr_pilot/lr=0.0001/seed-21230"] == "abc123"  # noqa: SLF001

    event = {"stage": "lr_pilot", "label": "lr=0.0001", "seed": 21230}
    from stage1_wandb_monitor import candidate_key

    assert candidate_key(event) == "lr_pilot/lr=0.0001/seed-21230"


# ---------------------------------------------------------------------------
# Complete safe W&B provenance (Audit 040 final review, point 1)
# ---------------------------------------------------------------------------
def test_the_monitor_records_the_campaign_identity_from_production():
    state = MonitorState()
    for event in synthetic_stream():
        state.consume(event)
    assert state.campaign == CAMPAIGN, "campaign provenance was not captured verbatim"


def test_the_wandb_config_carries_every_requested_provenance_field():
    """The eight fields the design requires, with their exact locked values."""
    state = MonitorState()
    for event in synthetic_stream():
        state.consume(event)
    run_start = next(e for e in synthetic_stream() if e["event"] == "run_start")
    config = safe_config(run_start, state.campaign)

    assert config["encoder_checkpoint"] == "vinai/phobert-base"
    assert config["encoder_revision"] == "01daacda68afe13d83023d16ec647239e344a1e6"
    assert config["precision"] == "fp32"
    assert config["corpus_revision"] == "a0a79294e4568137e25828bb3f2a4cde8546e1fb"
    assert config["corpus_manifest_digest"] == (
        "250859a57d745675c5dba2c7a35df08ccc123988bece873b0c9b29c6e78413d6")
    assert config["inventory_sha256"] == (
        "78eeb840d50455b14bd564da5aed7318d96468b8deaad5986b77bf5c538315d2")
    assert config["repository_head"] == "a" * 40
    assert config["protocol_version"] == "stage1-protocol-v1"
    # ...and the per-candidate scalars alongside them.
    assert config["lr"] == 1e-4 and config["r"] == 1.0 and config["seed"] == 21230


def test_the_campaign_keys_are_exactly_the_production_identity_plus_corpus_pin():
    """No second identity definition: the monitor selects, it never defines."""
    from unmark.stage1.artifact import IDENTITY_FIELDS

    assert set(IDENTITY_FIELDS) <= set(CAMPAIGN_PROVENANCE_KEYS)
    extra = set(CAMPAIGN_PROVENANCE_KEYS) - set(IDENTITY_FIELDS)
    assert extra == {"corpus_dataset", "corpus_revision"}, extra


def test_execute_stage_builds_the_campaign_identity_exactly_once():
    """Telemetry and the artifact cannot disagree about the campaign."""
    import ast
    import inspect

    import unmark.stage1.execute as execute_module

    tree = ast.parse(inspect.getsource(execute_module))
    stage = next(n for n in ast.walk(tree)
                 if isinstance(n, ast.FunctionDef) and n.name == "execute_stage")
    built = [n for n in ast.walk(stage)
             if isinstance(n, ast.Call)
             and getattr(n.func, "attr", None) == "from_inputs"]
    assert len(built) == 1, (
        f"CampaignIdentity is constructed {len(built)} times; it must be built once "
        "and reused by both telemetry and the artifact"
    )


def test_the_corpus_pin_comes_from_the_verified_corpus_not_a_constant():
    """`verified` exists only if those bytes were re-hashed from disk."""
    import inspect

    import unmark.stage1.execute as execute_module

    source = inspect.getsource(execute_module.execute_stage)
    assert "verified.identity.corpus_revision" in source
    assert "verified.identity.corpus_dataset" in source


def test_nothing_text_bearing_survives_the_whitelist():
    state = MonitorState()
    for event in synthetic_stream():
        state.consume(event)
    hostile = dict(next(e for e in synthetic_stream() if e["event"] == "run_start"))
    hostile.update({
        "canonical_text": "xin chao, this is corpus content",
        "chunk_text": "raw document body",
        "checkpoint_blob": "x" * 5000,
        "code": "print('secret')",
    })
    config = safe_config(hostile, state.campaign)
    for banned in ("canonical_text", "chunk_text", "checkpoint_blob", "code"):
        assert banned not in config
    assert set(config) <= set(SAFE_CONFIG_KEYS)
    for key, value in config.items():
        assert not (isinstance(value, str) and len(value) > 64), key


def test_extra_config_values_are_whitelisted_too():
    """A caller cannot smuggle a field in through `extra`."""
    config = safe_config({"stage": "lr_pilot"}, {"secret_text": "corpus", "precision": "fp32"})
    assert "secret_text" not in config
    assert config["precision"] == "fp32"


# ---------------------------------------------------------------------------
# Persistent monitor-state contract (Audit 040 final review, point 4)
# ---------------------------------------------------------------------------
def test_the_state_directory_is_configurable_and_may_live_on_drive(tmp_path):
    """Nothing hard-codes a user-specific Drive path into production science."""
    from stage1_wandb_monitor import WandbBridge, build_parser

    drive_like = tmp_path / "drive" / "MyDrive" / "UNMARK" / "stage1-monitoring" / ("a" * 40)
    bridge = WandbBridge("unmark-stage1", "lr-pilot", drive_like, enabled=False)
    assert drive_like.is_dir(), "the monitor must create its state directory"
    bridge._remember("lr_pilot/lr=0.0001/seed-21230", "run-abc")  # noqa: SLF001
    assert (drive_like / "wandb_run_ids.json").is_file()

    # The default is a relative path, and the flag is documented for Drive use.
    action = next(a for a in build_parser()._actions  # noqa: SLF001
                  if a.dest == "state_dir")
    assert not str(action.default).startswith("/content")
    assert "drive" in action.help.lower()

    for name in ("unmark/stage1/telemetry.py", "unmark/stage1/execute.py",
                 "unmark/stage1/trainer.py", "scripts/stage1_runner.py"):
        source = pathlib.Path(name).read_text(encoding="utf-8")
        assert "/content/drive" not in source, f"{name} must not hard-code a Drive path"


def test_wandb_run_ids_survive_a_runtime_deletion(tmp_path):
    """Colab wipes /content; a Drive-backed state dir must still resume the run."""
    from stage1_wandb_monitor import WandbBridge

    drive = tmp_path / "drive_state"
    first = WandbBridge("unmark-stage1", "lr-pilot", drive, enabled=False)
    first._remember("lr_pilot/lr=0.0001/seed-21230", "run-abc")  # noqa: SLF001
    first._remember("lr_pilot/lr=0.0003/seed-21230", "run-def")  # noqa: SLF001

    # A brand-new process, new object, same Drive directory.
    revived = WandbBridge("unmark-stage1", "lr-pilot", drive, enabled=False)
    ids = revived._ids()  # noqa: SLF001
    assert ids["lr_pilot/lr=0.0001/seed-21230"] == "run-abc"
    assert ids["lr_pilot/lr=0.0003/seed-21230"] == "run-def"


def test_a_scientific_resume_maps_to_the_same_wandb_candidate(tmp_path):
    """`--resume` must continue the SAME W&B run, not create a duplicate."""
    from stage1_wandb_monitor import WandbBridge, candidate_key

    drive = tmp_path / "drive_state"
    bridge = WandbBridge("unmark-stage1", "lr-pilot", drive, enabled=False)
    fresh = next(e for e in synthetic_stream() if e["event"] == "run_start")
    bridge._remember(candidate_key(fresh), "run-abc")  # noqa: SLF001

    resumed = dict(fresh, resumed=True, initial_global_update=7000)
    assert candidate_key(resumed) == candidate_key(fresh), (
        "candidate identity must not depend on how far the run had progressed"
    )
    assert bridge._ids()[candidate_key(resumed)] == "run-abc"  # noqa: SLF001


def test_losing_monitor_state_cannot_invalidate_scientific_checkpoints(tmp_path):
    """The state dir holds NO scientific state -- only run ids and a mirror."""
    from stage1_wandb_monitor import WandbBridge

    drive = tmp_path / "drive_state"
    bridge = WandbBridge("unmark-stage1", "lr-pilot", drive, enabled=False)
    bridge._remember("lr_pilot/lr=0.0001/seed-21230", "run-abc")  # noqa: SLF001
    (drive / "telemetry.jsonl").write_text("{}\n", encoding="utf-8")

    written = {p.name for p in drive.iterdir()}
    assert written <= {"wandb_run_ids.json", "telemetry.jsonl"}, written

    # Deleting all of it is survivable: nothing here is read by the science.
    for path in drive.iterdir():
        path.unlink()
    revived = WandbBridge("unmark-stage1", "lr-pilot", drive, enabled=False)
    assert revived._ids() == {}  # noqa: SLF001
    assert revived.start_candidate({"stage": "lr_pilot", "label": "lr=1e-4",
                                    "seed": 21230}, {}) is None


def test_the_scientific_process_never_reads_the_monitor_state_directory():
    for name in ("unmark/stage1/telemetry.py", "unmark/stage1/execute.py",
                 "unmark/stage1/trainer.py", "scripts/stage1_runner.py"):
        source = pathlib.Path(name).read_text(encoding="utf-8")
        assert "wandb_run_ids" not in source
        assert "unmark-monitor" not in source
