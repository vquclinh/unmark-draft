"""The pinned syllable inventory as a mandatory scientific input (Audit 030 §W).

The second real no-update smoke loaded the real PhoBERT encoder and *then* failed
closed in condition preparation, because the inventory D-B3A-001 pins is not
committed and a fresh Colab runtime had not provisioned it. The failure was
correct; its timing was not, and the run artifact could not have named the
inventory it used (D-S1A-008, "BLOCKING for scientific Stage-1 training").

These tests need no model, no network and no torch.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import inspect
import pathlib
import random
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from unmark.corruption.eligibility import (
    CorruptionPurpose,
    EligibilityPolicy,
    EligibilityUnresolved,
    active_eligibility_policy,
    require_resolved_eligibility,
)
from unmark.linguistics import (
    InventoryChecksumMismatch,
    InventoryUnavailable,
    clear_inventory_cache,
    load_inventory,
    load_manifest,
)
from unmark.stage1.preflight import (
    InventoryIdentity,
    ScientificInputsUnavailable,
    verify_scientific_inputs,
)

REPO = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = REPO / "configs/linguistics/vietnamese_syllables.yaml"
PROVENANCE = load_manifest(MANIFEST)


@pytest.fixture
def fake_repo(tmp_path):
    """A repo root whose cache we control. Always clears the memo."""
    clear_inventory_cache()
    manifest = tmp_path / "configs/linguistics/vietnamese_syllables.yaml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(MANIFEST.read_text(encoding="utf-8"), encoding="utf-8")
    cache = tmp_path / PROVENANCE.cache_relative_path
    cache.parent.mkdir(parents=True)
    yield manifest, cache, tmp_path
    clear_inventory_cache()


def real_bytes() -> bytes | None:
    path = REPO / PROVENANCE.cache_relative_path
    return path.read_bytes() if path.is_file() else None


provisioned = pytest.mark.skipif(
    real_bytes() is None,
    reason="the git-ignored inventory cache is not provisioned in this runtime",
)


# ---------------------------------------------------------------------------
# 1. The pin itself is exactly what D-B3A-001 and D-S1A-008 locked
# ---------------------------------------------------------------------------
def test_the_manifest_still_pins_the_locked_artifact():
    assert PROVENANCE.schema_version == "vn-syllables-v1"
    assert PROVENANCE.source_author == "hieuthi"
    assert PROVENANCE.source_revision == "135a4d9716e49a981624474156d6f247b9b46f6a"
    assert PROVENANCE.sha256 == (
        "78eeb840d50455b14bd564da5aed7318d96468b8deaad5986b77bf5c538315d2"
    )
    assert PROVENANCE.size_bytes == 116_290
    assert PROVENANCE.expected_entry_count == 17_974
    assert PROVENANCE.expected_unique_canonical_entry_count == 17_954
    assert PROVENANCE.expected_unique_stripped_form_count == 2_489
    assert PROVENANCE.license_status == "NO_EXPLICIT_LICENSE"


def test_the_raw_list_is_not_vendored():
    """No license statement upstream, so the bytes must not be in git."""
    import subprocess

    tracked = subprocess.run(
        ["git", "ls-files", PROVENANCE.cache_relative_path],
        cwd=REPO, capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert tracked == "", f"the unlicensed inventory is committed at {tracked}"


def test_the_pinned_url_is_an_immutable_revision_not_latest():
    assert PROVENANCE.source_revision in PROVENANCE.raw_url
    for mutable in ("/raw/master/", "/raw/main/", "HEAD"):
        assert mutable not in PROVENANCE.raw_url


# ---------------------------------------------------------------------------
# 2. Preflight rejects every way the artifact can be wrong
# ---------------------------------------------------------------------------
def test_a_missing_inventory_is_rejected(fake_repo):
    manifest, _cache, root = fake_repo
    with pytest.raises(ScientificInputsUnavailable) as caught:
        verify_scientific_inputs(manifest, root)
    message = str(caught.value)
    assert "not available" in message
    assert "fetch_vietnamese_syllable_inventory.py" in message
    assert "BEFORE the encoder was loaded" in message


@provisioned
def test_a_wrong_hash_is_rejected_and_diagnosed_as_such(fake_repo):
    """Present but wrong must not be reported as missing."""
    manifest, cache, root = fake_repo
    cache.write_bytes(real_bytes() + b"\nzzz\n")
    with pytest.raises(ScientificInputsUnavailable) as caught:
        verify_scientific_inputs(manifest, root)
    assert "NOT the pinned revision" in str(caught.value)


@provisioned
def test_a_truncated_inventory_is_rejected(fake_repo):
    manifest, cache, root = fake_repo
    cache.write_bytes(real_bytes()[: 116_290 // 2])
    with pytest.raises(ScientificInputsUnavailable):
        verify_scientific_inputs(manifest, root)


@provisioned
def test_a_removed_entry_is_rejected(fake_repo):
    manifest, cache, root = fake_repo
    lines = real_bytes().decode("utf-8").splitlines()
    cache.write_bytes("\n".join(lines[:-1]).encode("utf-8"))
    with pytest.raises(ScientificInputsUnavailable):
        verify_scientific_inputs(manifest, root)


def test_a_shape_change_that_preserves_the_hash_is_still_rejected(fake_repo):
    """The counts the manifest declares are now actually verified.

    Until this repair `load_manifest` parsed only `expected_entry_count`; the
    other two counts and `size_bytes` were declared under "verified on load" and
    checked by nothing. Simulated by pinning a manifest to bytes it does not
    describe, which is what a canonicalisation change would look like.
    """
    manifest, cache, root = fake_repo
    content = "\n".join(f"tie{n}" for n in range(50)) + "\n"
    raw = content.encode("utf-8")
    text = manifest.read_text(encoding="utf-8")
    text = text.replace(PROVENANCE.sha256, hashlib.sha256(raw).hexdigest())
    text = text.replace("size_bytes: 116290", f"size_bytes: {len(raw)}")
    text = text.replace("expected_entry_count: 17974", "expected_entry_count: 50")
    manifest.write_text(text, encoding="utf-8")
    cache.write_bytes(raw)
    # entry count now agrees, but the two other declared counts do not.
    with pytest.raises(ScientificInputsUnavailable) as caught:
        verify_scientific_inputs(manifest, root)
    assert "changed shape" in str(caught.value)


def test_self_check_can_never_reach_the_scientific_preflight():
    with pytest.raises(ScientificInputsUnavailable) as caught:
        verify_scientific_inputs(MANIFEST, REPO, purpose=CorruptionPurpose.SELF_CHECK)
    assert "SELF_CHECK must never reach a scientific route" in str(caught.value)


def test_unresolved_eligibility_is_rejected_by_the_corruption_guard():
    with pytest.raises(EligibilityUnresolved):
        require_resolved_eligibility(policy=EligibilityPolicy.UNRESOLVED)


# ---------------------------------------------------------------------------
# 3. The happy path, and what it returns
# ---------------------------------------------------------------------------
@provisioned
def test_the_provisioned_inventory_verifies_against_the_pin():
    inputs = verify_scientific_inputs(MANIFEST, REPO)
    assert inputs.report["eligibility_policy"] == "VIETNAMESE_SYLLABLE_INVENTORY"
    shape = inputs.report["inventory_shape"]
    assert shape["raw_entry_count"] == 17_974
    assert shape["unique_canonical_entry_count"] == 17_954
    assert shape["unique_stripped_form_count"] == 2_489
    assert shape["collisions_after_stripping"] == 15_465


@provisioned
def test_the_identity_is_exactly_the_seven_fields_d_s1a_008_requires():
    identity = verify_scientific_inputs(MANIFEST, REPO).inventory
    assert set(identity.to_dict()) == {
        "inventory_schema_version", "source_name", "source_author",
        "source_revision", "sha256", "size_bytes", "license_status",
    }
    assert identity.sha256 == PROVENANCE.sha256
    assert identity.license_status == "NO_EXPLICIT_LICENSE"


@provisioned
def test_the_parsed_membership_digest_is_deterministic():
    first = verify_scientific_inputs(MANIFEST, REPO).report["parsed_membership_digest"]
    clear_inventory_cache()
    second = verify_scientific_inputs(MANIFEST, REPO).report["parsed_membership_digest"]
    assert first == second and len(first) == 64


# ---------------------------------------------------------------------------
# 4. Preflight must precede model load, in every path
# ---------------------------------------------------------------------------
def statement_order(source: str, function: str, first: str, second: str):
    tree = ast.parse(source)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == function)
    lines = {}
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if name in (first, second):
                lines.setdefault(name, node.lineno)
    return lines.get(first), lines.get(second)


@pytest.mark.parametrize("module,function,builder", [
    # execute_stage now builds the shared frozen backbone (D-S1B-017); the
    # preflight must still precede it.
    ("unmark/stage1/execute.py", "execute_stage", "build_backbone"),
    ("unmark/stage1/execute.py", "smoke_check", "build_objective"),
    ("scripts/stage1_pretrain_measurements.py", "validation_timing", "build"),
])
def test_preflight_runs_before_the_encoder_is_built(module, function, builder):
    source = (REPO / module).read_text(encoding="utf-8")
    preflight, model = statement_order(source, function, "verify_scientific_inputs", builder)
    assert preflight is not None, f"{function} does not call verify_scientific_inputs"
    assert model is not None, f"{function} does not call {builder}"
    assert preflight < model, (
        f"{function}: the encoder is built at line {model} before the scientific "
        f"preflight at line {preflight}; a missing inventory would again be found "
        "only after the model was downloaded and resident"
    )


def test_the_preflight_reuses_the_authoritative_loader():
    """No second verifier: it must call `load_inventory`, not re-hash itself."""
    from unmark.stage1 import preflight

    tree = ast.parse(inspect.getsource(preflight))
    called = {getattr(n.func, "id", None) or getattr(n.func, "attr", None)
              for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert "load_inventory" in called
    assert "try_load_inventory" not in called, (
        "try_load_inventory swallows InventoryChecksumMismatch and would report a "
        "corrupted cache as a missing one"
    )


# ---------------------------------------------------------------------------
# 5. Validation and training share ONE eligibility path
# ---------------------------------------------------------------------------
def test_stage1_corruption_never_asks_for_self_check():
    for module in ("unmark/stage1/data.py", "unmark/stage1/validation.py",
                   "unmark/stage1/execute.py", "scripts/stage1_pretrain_measurements.py"):
        source = (REPO / module).read_text(encoding="utf-8")
        assert "SELF_CHECK" not in source, f"{module} can reach the provisional fallback"


def test_the_scientific_guard_fires_before_any_condition_is_dispatched():
    """Including FULL: the guard must not sit behind a per-condition branch."""
    from unmark.corruption import corrupt as corrupt_module

    tree = ast.parse(inspect.getsource(corrupt_module))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "corrupt")
    guard = next(n.lineno for n in ast.walk(fn) if isinstance(n, ast.Call)
                 and getattr(n.func, "id", None) == "require_resolved_eligibility")
    dispatch = next(n.lineno for n in ast.walk(fn) if isinstance(n, ast.Call)
                    and getattr(n.func, "id", None) == "get_condition")
    assert guard < dispatch


def test_validation_and_training_build_the_classifier_the_same_way():
    source = (REPO / "unmark/stage1/execute.py").read_text(encoding="utf-8")
    assert source.count("make_classifier(try_load_inventory())") >= 2
    assert "make_classifier" not in (REPO / "unmark/stage1/validation.py").read_text(
        encoding="utf-8"
    ), "validation must receive the classifier, not build a second one"


# ---------------------------------------------------------------------------
# 6. Stage 6 does NOT depend on the inventory (so no rerun is required)
# ---------------------------------------------------------------------------
def test_safe_cut_offsets_never_reads_its_classifier_argument():
    from unmark.stage1 import chunking

    tree = ast.parse(inspect.getsource(chunking))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "safe_cut_offsets")
    reads = [n.id for n in ast.walk(ast.Module([*fn.body], []))
             if isinstance(n, ast.Name) and n.id == "classifier"]
    assert reads == [], (
        "safe_cut_offsets reads its classifier; chunk boundaries would depend on "
        "the syllable inventory and the prepared corpus would not be reproducible "
        "without it (D-S1B-013 requires the predicate to be lexicon-free)"
    )


def test_chunk_boundaries_are_identical_under_any_classifier():
    """Empirical twin of the structural test, over adversarial text."""
    from unmark.stage1.chunking import safe_cut_offsets

    def always(_):
        return type("E", (), {"name": "VIETNAMESE_CANDIDATE"})

    def never(_):
        return type("E", (), {"name": "NOT_APPLICABLE"})

    rng = random.Random(20260823)
    alphabet = "aăâbcdđeêghiklmnoôơpqrstuưvxy ,.-—«»/()0123456789̣̀́̃̉한글"
    for _ in range(2_000):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 60)))
        base = safe_cut_offsets(text)
        assert base == safe_cut_offsets(text, always) == safe_cut_offsets(text, never), text


def test_the_stage6_prepare_worker_does_not_gate_on_the_inventory():
    """It builds a classifier, but with `try_load_inventory`, and never fails on it."""
    from unmark.stage1 import parallel

    source = inspect.getsource(parallel)
    assert "try_load_inventory()" in source
    assert "require_resolved_eligibility" not in source
    assert "verify_scientific_inputs" not in source


# ---------------------------------------------------------------------------
# 7. Provenance binds the inventory (D-S1A-008) and resume rejects drift
# ---------------------------------------------------------------------------
def identity(**overrides) -> InventoryIdentity:
    base = dict(
        inventory_schema_version="vn-syllables-v1",
        source_name="all-vietnamese-syllables.txt",
        source_author="hieuthi",
        source_revision=PROVENANCE.source_revision,
        sha256=PROVENANCE.sha256,
        size_bytes=116_290,
        license_status="NO_EXPLICIT_LICENSE",
    )
    base.update(overrides)
    return InventoryIdentity(**base)


def provenance(**overrides):
    from unmark.stage1.trainer import RunProvenance

    base = dict(
        run_seed=36930, init_seed=51800, corruption_seed=35422, learning_rate=3e-4, r=1.0,
        corpus_manifest_digest="d" * 64, repository_head="a" * 40,
        inventory=identity(),
    )
    base.update(overrides)
    return RunProvenance(**base)


def payload_for(p):
    from unmark.stage1.trainer import checkpoint_payload

    return checkpoint_payload(
        provenance=p, adapter_state={}, optimizer_state={}, global_update=1,
        sampler_state={}, cap=2, budget_limited=False, points=[],
    )


def test_provenance_records_the_inventory_identity():
    recorded = provenance().to_dict()["inventory"]
    assert recorded["sha256"] == PROVENANCE.sha256
    assert recorded["source_revision"] == PROVENANCE.source_revision
    assert recorded["license_status"] == "NO_EXPLICIT_LICENSE"


@pytest.mark.parametrize("field,value", [
    ("sha256", "0" * 64),
    ("source_revision", "b" * 40),
    ("inventory_schema_version", "vn-syllables-v2"),
    ("size_bytes", 999),
])
def test_resume_rejects_a_different_inventory(field, value):
    """Two runs differing only in the inventory are different experiments."""
    from unmark.stage1.trainer import TrainerContractViolation, verify_checkpoint

    mine = provenance()
    theirs = provenance(inventory=identity(**{field: value}))
    with pytest.raises(TrainerContractViolation) as caught:
        verify_checkpoint(payload_for(theirs), mine)
    assert "'inventory'" in str(caught.value)


def test_a_checkpoint_with_no_inventory_cannot_resume_a_scientific_run():
    from unmark.stage1.trainer import TrainerContractViolation, verify_checkpoint

    with pytest.raises(TrainerContractViolation):
        verify_checkpoint(payload_for(provenance(inventory=None)), provenance())


def test_the_matching_inventory_still_resumes():
    from unmark.stage1.trainer import verify_checkpoint

    verify_checkpoint(payload_for(provenance()), provenance())


def test_execute_stage_binds_the_verified_identity_not_a_hand_built_one():
    source = (REPO / "unmark/stage1/execute.py").read_text(encoding="utf-8")
    assert "inventory=inputs.inventory" in source
