"""Nominal-run independence and initialisation — structural half (§AE).

Audit 030 §AD found that `build_objective` was called **once, outside** the
schedule loop, so every nominal candidate trained one shared adapter and
candidates 2..N inherited trained weights. §AC.7 separately found initialisation
unseeded.

**Torch-free**, so it runs in the ML-free venv on every run. Its companion
`test_stage1_run_independence_runtime.py` proves the same contracts by executing
them, and its CUDA half needs a GPU.
"""

from __future__ import annotations

import ast
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from unmark.stage1.protocol import (
    ADAPTER_INIT_SEED_TAG,
    ADAPTER_INIT_SEEDS,
    ALL_SEEDS,
    SEED_ROOT_TAG,
    SELECTION_SEED,
    TRAIN_SEEDS,
    adapter_init_seed,
)

REPO = pathlib.Path(__file__).resolve().parents[1]
EXECUTE = (REPO / "unmark/stage1/execute.py").read_text(encoding="utf-8")
TRAINER = (REPO / "unmark/stage1/trainer.py").read_text(encoding="utf-8")


def function(source: str, name: str) -> ast.FunctionDef:
    return next(n for n in ast.walk(ast.parse(source))
                if isinstance(n, ast.FunctionDef) and n.name == name)


def calls(node: ast.AST) -> list[str]:
    return [getattr(n.func, "id", None) or getattr(n.func, "attr", None)
            for n in ast.walk(node) if isinstance(n, ast.Call)]


# ---------------------------------------------------------------------------
# 1. The init seed: domain-separated, and a function of run_seed ALONE
# ---------------------------------------------------------------------------
def test_the_init_tag_is_domain_separated_in_the_established_style():
    assert ADAPTER_INIT_SEED_TAG == f"{SEED_ROOT_TAG}|adapter-init"
    assert ADAPTER_INIT_SEED_TAG not in ALL_SEEDS


def test_the_locked_schedule_init_seeds():
    """The four values the FINAL CONFIGURATION FREEZE must carry."""
    assert ADAPTER_INIT_SEEDS == {21230: 3203, 36930: 51800, 7309: 45833, 5993: 15758}
    assert adapter_init_seed(SELECTION_SEED) == 3203
    assert tuple(adapter_init_seed(s) for s in TRAIN_SEEDS) == (51800, 45833, 15758)


def test_init_seeds_never_collide_with_a_role_seed():
    assert not set(ADAPTER_INIT_SEEDS.values()) & set(ALL_SEEDS.values())


def test_the_init_seed_differs_from_the_run_seed_it_derives_from():
    """Domain separation is the point: the two streams must not be one stream."""
    for run_seed, init in ADAPTER_INIT_SEEDS.items():
        assert init != run_seed, run_seed


def test_same_run_seed_gives_the_same_init_seed():
    assert adapter_init_seed(21230) == adapter_init_seed(21230)


def test_different_run_seeds_give_different_init_seeds():
    derived = [adapter_init_seed(s) for s in (21230, 36930, 7309, 5993)]
    assert len(set(derived)) == 4


def test_the_derivation_takes_run_seed_and_nothing_else():
    """LR, r, label and execution order cannot enter — the signature forbids it."""
    import inspect

    parameters = list(inspect.signature(adapter_init_seed).parameters)
    assert parameters == ["run_seed"], parameters


def test_all_eight_selection_candidates_share_one_init_seed():
    """Paired comparison: LR/r sweeps vary only their target (D-S1B-016)."""
    from unmark.stage1.selection import lr_pilot_schedule, r_phase1_schedule

    planned = [*lr_pilot_schedule(SELECTION_SEED), *r_phase1_schedule(SELECTION_SEED, 3e-4)]
    assert len(planned) == 8
    seeds = {adapter_init_seed(p.seed) for p in planned}
    assert seeds == {3203}, "the eight selection candidates must share one initialisation"


def test_the_three_final_runs_have_three_distinct_init_seeds():
    from unmark.stage1.selection import final_main_schedule

    planned = final_main_schedule(3e-4, 1.0)
    assert len({adapter_init_seed(p.seed) for p in planned}) == 3


def test_the_eleven_runs_form_exactly_four_init_groups_with_multiplicities_8_1_1_1():
    """**FOUR** groups, not two.

    Two *methodological categories* — paired selection, and seed-varied final-main
    — but four distinct initialisations: the eight selection candidates share one,
    and each final seed has its own. An earlier revision of §AD/§AE said "two hash
    groups"; that conflated the category count with the group count.
    """
    from collections import Counter

    from unmark.stage1.selection import (
        final_main_schedule,
        lr_pilot_schedule,
        r_phase1_schedule,
    )

    planned = [
        *lr_pilot_schedule(SELECTION_SEED),
        *r_phase1_schedule(SELECTION_SEED, 3e-4),
        *final_main_schedule(3e-4, 1.0),
    ]
    assert len(planned) == 11

    groups = Counter(adapter_init_seed(p.seed) for p in planned)
    assert len(groups) == 4, groups
    assert sorted(groups.values(), reverse=True) == [8, 1, 1, 1], groups
    assert groups == {3203: 8, 51800: 1, 45833: 1, 15758: 1}


def test_fresh_adapter_seeds_only_the_cpu_default_generator():
    """`torch.manual_seed` seeds **all devices**, and `fork_rng(devices=[])` does
    not snapshot CUDA state — so that pairing would perturb CUDA RNG without
    restoring it. AST, so a docstring naming the forbidden call cannot mislead."""
    import inspect

    from unmark.stage1 import initialisation

    node = function(inspect.getsource(initialisation), "fresh_adapter")
    seeding = [n for n in ast.walk(node) if isinstance(n, ast.Call)
               and getattr(n.func, "attr", None) == "manual_seed"]
    assert seeding, "fresh_adapter must seed something"
    for call in seeding:
        target = ast.unparse(call.func)
        assert target == "torch.default_generator.manual_seed", target

    called = calls(node)
    assert "manual_seed_all" not in called
    for name in ("cuda", "device", "to"):
        assert name not in called, f"fresh_adapter must not touch {name}()"


def test_fresh_adapter_forks_the_cpu_rng_only():
    import inspect

    from unmark.stage1 import initialisation

    node = function(inspect.getsource(initialisation), "fresh_adapter")
    fork = next(n for n in ast.walk(node) if isinstance(n, ast.Call)
                and getattr(n.func, "attr", None) == "fork_rng")
    devices = next(k for k in fork.keywords if k.arg == "devices")
    assert ast.unparse(devices.value) == "[]", ast.unparse(devices.value)


# ---------------------------------------------------------------------------
# 2. The sampler's data-order semantics are UNCHANGED
# ---------------------------------------------------------------------------
def test_the_sampler_is_still_seeded_by_run_seed_itself():
    """D-S1B-016 must not touch data order. Asserted on the real call."""
    node = function(TRAINER, "train_run")
    sampler_calls = [n for n in ast.walk(node) if isinstance(n, ast.Call)
                     and getattr(n.func, "id", None) == "DeterministicSampler"]
    assert sampler_calls, "train_run must construct a DeterministicSampler"
    keyword = next(k for k in sampler_calls[0].keywords if k.arg == "seed")
    assert ast.unparse(keyword.value) == "provenance.run_seed", ast.unparse(keyword.value)


def test_the_trainer_never_re_derives_an_init_seed():
    """AST, not text: the trainer's docstrings legitimately name the function."""
    called = calls(ast.parse(TRAINER))
    assert "adapter_init_seed" not in called, (
        "the trainer must not re-derive an init seed; execute_stage owns run identity "
        "and a continuation must reuse it, never recompute a new one"
    )


# ---------------------------------------------------------------------------
# 3. The construction boundary: fresh adapter per nominal run
# ---------------------------------------------------------------------------
def test_execute_stage_builds_the_backbone_once_outside_the_loop():
    node = function(EXECUTE, "execute_stage")
    loop = next(n for n in ast.walk(node) if isinstance(n, ast.For))
    built = [n.lineno for n in ast.walk(node) if isinstance(n, ast.Call)
             and getattr(n.func, "id", None) == "build_backbone"]
    assert built and all(line < loop.lineno for line in built), built


def test_a_fresh_adapter_is_constructed_inside_the_loop():
    """The §AD repair: what used to be shared is now per-candidate."""
    node = function(EXECUTE, "execute_stage")
    loop = next(n for n in ast.walk(node) if isinstance(n, ast.For))
    inside = calls(loop)
    # `objective_cls` until the consolidated repair -- a name this function
    # never bound. Requiring it here is what kept the `NameError` alive through
    # every green suite (Audit 031 B1). The real constructor is `Stage1Objective`,
    # and `test_stage1_name_resolution.py` proves the name actually resolves.
    for required in ("fresh_adapter", "UnmarkEncoder", "Stage1Objective",
                     "expected_fresh_init_hash", "trainable_state_hash"):
        assert required in inside, f"{required} must be called per nominal run"


def test_the_old_shared_objective_construction_is_gone():
    node = function(EXECUTE, "execute_stage")
    assert "build_objective" not in calls(node), (
        "execute_stage must not build one shared UnmarkEncoder for every candidate"
    )


def test_the_backbone_is_checked_after_every_nominal_run():
    node = function(EXECUTE, "execute_stage")
    loop = next(n for n in ast.walk(node) if isinstance(n, ast.For))
    assert "require_frozen_backbone_unchanged" in calls(loop)


def test_the_frozen_backbone_is_placed_once_not_per_candidate():
    """No CPU<->CUDA shuttling of ~135M parameters between candidates."""
    node = function(EXECUTE, "execute_stage")
    loop = next(n for n in ast.walk(node) if isinstance(n, ast.For))
    moved = [n for n in ast.walk(loop) if isinstance(n, ast.Call)
             and getattr(n.func, "attr", None) == "to"
             and ast.unparse(n.func).startswith("frozen_encoder")]
    assert moved == [], "the shared backbone must not be moved inside the run loop"


# ---------------------------------------------------------------------------
# 4. Device + numerics are established before any model work
# ---------------------------------------------------------------------------
def test_the_scientific_device_contract_precedes_the_backbone():
    node = function(EXECUTE, "execute_stage")
    order = {}
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            name = getattr(n.func, "id", None)
            if name in ("require_deterministic_cublas_workspace", "resolve_scientific_device",
                        "enforce_numerical_policy", "verify_numerical_policy",
                        "build_backbone", "verify_scientific_inputs"):
                order.setdefault(name, n.lineno)
    for earlier in ("verify_scientific_inputs", "require_deterministic_cublas_workspace",
                    "resolve_scientific_device", "enforce_numerical_policy",
                    "verify_numerical_policy"):
        assert order[earlier] < order["build_backbone"], earlier
    assert order["require_deterministic_cublas_workspace"] < order["resolve_scientific_device"], (
        "CUBLAS_WORKSPACE_CONFIG must be settled before CUDA is touched"
    )


def test_the_checkpoint_stores_adapter_only_state():
    node = function(TRAINER, "train_run")
    payload = next(n for n in ast.walk(node) if isinstance(n, ast.Call)
                   and getattr(n.func, "id", None) == "checkpoint_payload")
    keyword = next(k for k in payload.keywords if k.arg == "adapter_state")
    assert ast.unparse(keyword.value) == "adapter.state_dict()", ast.unparse(keyword.value)


def test_every_state_restore_in_the_trainer_is_strict():
    """AST over the real `load_state_dict` calls; comments mentioning the old
    `strict=False` behaviour must not satisfy or break this."""
    tree = ast.parse(TRAINER)
    loads = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "attr", None) == "load_state_dict"]
    assert loads, "expected at least one load_state_dict call"
    adapter_loads = [n for n in loads if "adapter" in ast.unparse(n.func)]
    assert adapter_loads, "the adapter state must be restored somewhere"
    for call in adapter_loads:
        strict = next((k for k in call.keywords if k.arg == "strict"), None)
        assert strict is not None, f"{ast.unparse(call)} does not state strict="
        assert strict.value.value is True, (
            f"{ast.unparse(call)} is fail-open: a key mismatch would silently "
            "restore nothing and training would continue from fresh weights"
        )


def test_the_checkpoint_schema_was_versioned_for_the_new_state_shape():
    from unmark.stage1.trainer import CHECKPOINT_SCHEMA_VERSION

    assert CHECKPOINT_SCHEMA_VERSION == "stage1-checkpoint-v2"


@pytest.mark.parametrize("helper", [
    "require_optimizer_parameter_identity", "require_optimizer_state_device",
])
def test_resume_asserts_the_optimizer_contracts(helper):
    node = function(TRAINER, "train_run")
    assert helper in calls(node), helper
