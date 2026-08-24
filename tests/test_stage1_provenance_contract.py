"""The `RunProvenance` serialization contract (Audit 030 §V).

The first real Colab no-update smoke stopped here: six `test_a_foreign_run_cannot_resume`
cases died in **setup** on

    RunProvenance(**mine.to_dict())
    TypeError: unexpected keyword argument 'lambda_align'

`to_dict()` emits the two derived weights on top of the constructor fields, so it
is not constructor-round-trippable -- by design, not by accident. This file pins
that contract from both directions, so neither the assumption that broke the test
nor the absence of a check on the derived keys can come back.

Torch-free: it runs in the ML-free venv on every run, which is exactly where the
defect should have been caught.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from unmark.stage1.preflight import InventoryIdentity
from unmark.stage1.protocol import LAMBDA_SCALE_SUM, lambdas_for_r
from unmark.stage1.trainer import (
    RunProvenance,
    TrainerContractViolation,
    checkpoint_payload,
    verify_checkpoint,
)

CONSTRUCTOR_FIELDS = (
    # D-S1B-016 added `init_seed`: it determines the run's initial weights, so it
    # is scientific identity and must be gated exactly like the other fields.
    "run_seed", "init_seed", "corruption_seed", "learning_rate", "r",
    "corpus_manifest_digest", "repository_head",
    "backbone_checkpoint", "backbone_revision", "protocol_version", "precision",
    # D-S1A-008, added by Audit 030 §W: the pinned syllable inventory decides
    # every corruption denominator, so a run artifact must name the one it used.
    "inventory",
)


INVENTORY = InventoryIdentity(
    inventory_schema_version="vn-syllables-v1",
    source_name="all-vietnamese-syllables.txt",
    source_author="hieuthi",
    source_revision="135a4d9716e49a981624474156d6f247b9b46f6a",
    sha256="78eeb840d50455b14bd564da5aed7318d96468b8deaad5986b77bf5c538315d2",
    size_bytes=116_290,
    license_status="NO_EXPLICIT_LICENSE",
)


def provenance(**overrides) -> RunProvenance:
    base = dict(
        run_seed=36930, init_seed=51800, corruption_seed=35422, learning_rate=3e-4, r=1.0,
        corpus_manifest_digest="d" * 64, repository_head="a" * 40,
        inventory=INVENTORY,
    )
    base.update(overrides)
    return RunProvenance(**base)


def payload_for(p: RunProvenance) -> dict:
    return checkpoint_payload(
        provenance=p, adapter_state={}, optimizer_state={}, global_update=7,
        sampler_state={"cursor": 3, "visit": 1}, cap=20, budget_limited=False, points=[],
    )


# ---------------------------------------------------------------------------
# 1. What the two forms actually contain
# ---------------------------------------------------------------------------
def test_the_constructor_takes_exactly_the_declared_identity_fields():
    fields = tuple(f.name for f in dataclasses.fields(RunProvenance))
    assert fields == CONSTRUCTOR_FIELDS, fields


def test_to_dict_emits_the_constructor_fields_plus_exactly_the_derived_keys():
    keys = set(provenance().to_dict())
    assert keys == set(CONSTRUCTOR_FIELDS) | set(RunProvenance.DERIVED_KEYS)
    assert RunProvenance.DERIVED_KEYS == ("lambda_align", "lambda_clean")


def test_the_derived_keys_are_not_constructor_parameters():
    """`DERIVED_KEYS` must stay disjoint from the fields, or the contract is moot."""
    fields = {f.name for f in dataclasses.fields(RunProvenance)}
    assert fields.isdisjoint(RunProvenance.DERIVED_KEYS)


def test_to_dict_is_deliberately_not_constructor_round_trippable():
    """The exact failure the real smoke hit, pinned as the CONTRACT.

    If someone ever makes this pass by dropping the derived keys from `to_dict`,
    artifacts stop recording the weights their objective used, and this test
    should be the thing that forces that to be a deliberate decision.
    """
    with pytest.raises(TypeError) as caught:
        RunProvenance(**provenance().to_dict())
    assert "lambda_align" in str(caught.value)


def test_dataclasses_replace_is_the_authoritative_derivation():
    """The supported way to make one provenance from another."""
    mine = provenance()
    theirs = dataclasses.replace(mine, r=4.0)
    assert theirs.r == 4.0
    assert theirs.run_seed == mine.run_seed
    assert theirs.corpus_manifest_digest == mine.corpus_manifest_digest
    # Derived weights follow `r` automatically; they cannot be set independently.
    assert theirs.to_dict()["lambda_align"] == lambdas_for_r(4.0)[0]


def test_no_production_code_reconstructs_provenance_from_a_mapping():
    """Identity comes from the plan, never from the artifact being resumed.

    Asserted on the call graph across `unmark/` and `scripts/`: a
    `RunProvenance(**something)` anywhere in production would mean a checkpoint
    could define which experiment it belongs to.
    """
    offenders = []
    for path in [*pathlib.Path("unmark").rglob("*.py"), *pathlib.Path("scripts").rglob("*.py")]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if getattr(node.func, "id", None) != "RunProvenance":
                continue
            if any(kw.arg is None for kw in node.keywords):
                offenders.append(f"{path}:{node.lineno}")
    assert offenders == [], offenders


# ---------------------------------------------------------------------------
# 2. The real production lifecycle, end to end
# ---------------------------------------------------------------------------
def test_the_production_lifecycle_preserves_every_identity_field():
    """p -> to_dict -> checkpoint payload -> verify_checkpoint against p."""
    mine = provenance()
    recorded = payload_for(mine)["provenance"]
    for field in CONSTRUCTOR_FIELDS:
        expected = getattr(mine, field)
        # `inventory` is a nested identity; it serialises through its own to_dict.
        if field == "inventory":
            expected = expected.to_dict() if expected is not None else None
        assert recorded[field] == expected, field
    verify_checkpoint(payload_for(mine), mine)  # must not raise


def test_the_lifecycle_survives_a_json_round_trip():
    """The result artifact is JSON; floats must come back bit-exact."""
    mine = provenance(learning_rate=3e-4, r=0.3)
    payload = payload_for(mine)
    payload["provenance"] = json.loads(json.dumps(payload["provenance"]))
    verify_checkpoint(payload, mine)
    assert payload["provenance"]["learning_rate"] == 3e-4
    assert payload["provenance"]["r"] == 0.3


@pytest.mark.parametrize("field,value", [
    ("run_seed", 7309),
    ("init_seed", 45833),
    ("corruption_seed", 1),
    ("learning_rate", 1e-3),
    ("r", 4.0),
    ("corpus_manifest_digest", "e" * 64),
    ("repository_head", "b" * 40),
    ("backbone_checkpoint", "vinai/phobert-large"),
    ("backbone_revision", "deadbeef"),
    ("protocol_version", "stage1-not-this-one"),
    ("precision", "bf16"),
    ("inventory", None),
])
def test_every_constructor_field_is_gated_on_resume(field, value):
    """Not just the six: no identity field may be silently unguarded."""
    mine = provenance()
    theirs = dataclasses.replace(mine, **{field: value})
    with pytest.raises(TrainerContractViolation) as caught:
        verify_checkpoint(payload_for(theirs), mine)
    message = str(caught.value)
    assert "checkpoint provenance mismatch" in message, message
    assert repr(field) in message, message


# ---------------------------------------------------------------------------
# 3. r / lambda -- the derived science must be coherent
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("r", [0.0, 0.3, 1.0, 3.0, 4.0])
def test_the_recorded_weights_obey_the_locked_relationship(r):
    recorded = provenance(r=r).to_dict()
    align, clean = recorded["lambda_align"], recorded["lambda_clean"]
    assert (align, clean) == lambdas_for_r(r)
    assert align + clean == pytest.approx(LAMBDA_SCALE_SUM)
    if align:
        assert clean / align == pytest.approx(r)


@pytest.mark.parametrize("key", ["lambda_align", "lambda_clean"])
def test_a_checkpoint_claiming_one_r_with_inconsistent_weights_is_refused(key):
    """A hand-edited or corrupted artifact cannot misdescribe its own objective."""
    mine = provenance(r=1.0)
    payload = payload_for(mine)
    payload["provenance"][key] = 99.0
    with pytest.raises(TrainerContractViolation) as caught:
        verify_checkpoint(payload, mine)
    assert "internally inconsistent" in str(caught.value)
    assert key in str(caught.value)


@pytest.mark.parametrize("key", ["lambda_align", "lambda_clean"])
def test_a_checkpoint_missing_a_derived_key_is_refused(key):
    mine = provenance()
    payload = payload_for(mine)
    payload["provenance"].pop(key)
    with pytest.raises(TrainerContractViolation) as caught:
        verify_checkpoint(payload, mine)
    assert "missing the derived key" in str(caught.value)


def test_the_weights_property_and_the_serialized_weights_agree():
    mine = provenance(r=0.3)
    assert mine.to_dict()["lambda_align"] == mine.weights.lambda_align
    assert mine.to_dict()["lambda_clean"] == mine.weights.lambda_clean


def test_an_extra_unknown_key_does_not_smuggle_identity_past_the_gate():
    """Unknown keys are ignored, but every gated field is still compared."""
    mine = provenance()
    payload = payload_for(mine)
    payload["provenance"]["invented_field"] = "whatever"
    verify_checkpoint(payload, mine)  # tolerated
    payload["provenance"]["run_seed"] = 1
    with pytest.raises(TrainerContractViolation):
        verify_checkpoint(payload, mine)
