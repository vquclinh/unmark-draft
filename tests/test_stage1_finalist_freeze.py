"""The Stage-1 FINALIST FREEZE, checked against the code (Audit 048).

`docs/spec/stage1-adapter-finalists.json` freezes exactly two already-trained
adapters and records that the final downstream adapter is still UNSELECTED. This
file is what makes that mechanical rather than decorative.

Two halves:

* the freeze contract itself, which needs no torch and therefore runs anywhere;
* checkpoint-payload verification, which inspects real tensors and is skipped
  where torch is absent -- following the existing convention in
  `tests/test_stage1_fused_r_phase1.py`. Those tests are meaningful only on the
  machine that actually holds the checkpoints.

Values are compared against the **imported constants**, never against literals
re-typed here; re-typing would only prove the test agrees with itself.
"""

from __future__ import annotations

import copy
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from unmark.stage1.finalists import (  # noqa: E402
    ADAPTER_STATE_KEYS,
    ADAPTER_TENSOR_COUNT,
    ADJUDICATION_ALLOWED_EVIDENCE,
    BINDING_FIELDS,
    CORPUS_MANIFEST_DIGEST,
    FINALIST_A,
    FINALIST_B,
    FINALIST_COUNT,
    FINALIST_FREEZE_SCHEMA_VERSION,
    FINALISTS,
    FREEZE_PATH,
    FROZEN_LEARNING_RATE,
    FROZEN_R,
    INVENTORY_SHA256,
    INVENTORY_SIZE_BYTES,
    INVENTORY_SOURCE_REVISION,
    SHA256_PENDING,
    VERIFIED_PROVENANCE_FIELDS,
    FinalistFreezeViolation,
    FinalistIdentity,
    expected_run_provenance,
    finalist_for,
    freeze_is_complete,
    load_freeze,
    pending_finalists,
    require_sha256,
    resolve_inventory,
    validate_finalists,
    validate_freeze_payload,
    verify_finalist_checkpoint,
)
from unmark.stage1.protocol import (  # noqa: E402
    ADAPTER_TRAINABLE_PARAMETERS,
    ENCODER_CHECKPOINT,
    ENCODER_REVISION,
    PRECISION,
    STAGE1_PROTOCOL_VERSION,
)
from unmark.stage1.trainer import CHECKPOINT_SCHEMA_VERSION  # noqa: E402


def freeze() -> dict:
    return json.loads(FREEZE_PATH.read_text(encoding="utf-8"))


def finalists_from(payload: dict) -> list[dict]:
    return payload["adjudication"]["finalists"]["value"]


# ==========================================================================================
# The committed artifact
# ==========================================================================================

def test_the_committed_freeze_validates():
    payload = load_freeze()
    assert payload["schema_version"] == FINALIST_FREEZE_SCHEMA_VERSION


def test_the_freeze_records_stage1_as_closed():
    payload = freeze()
    for field in ("training", "candidate_generation", "learning_rate_selection", "r_selection"):
        assert payload["stage1"][field]["value"] == "CLOSED", field


def test_the_freeze_agrees_with_the_imported_model_constants():
    model = freeze()["model"]
    assert model["backbone"]["value"] == ENCODER_CHECKPOINT
    assert model["revision"]["value"] == ENCODER_REVISION
    assert model["protocol_version"]["value"] == STAGE1_PROTOCOL_VERSION
    assert model["adapter_trainable_parameters"]["value"] == ADAPTER_TRAINABLE_PARAMETERS
    assert model["adapter_tensor_count"]["value"] == ADAPTER_TENSOR_COUNT
    assert model["adapter_dtype"]["value"] == PRECISION
    assert model["checkpoint_schema_version"]["value"] == CHECKPOINT_SCHEMA_VERSION
    assert tuple(model["adapter_state_keys"]["value"]) == ADAPTER_STATE_KEYS


def test_the_adapter_contract_matches_the_real_architecture():
    """8 tensors and 3,551,232 parameters, derived -- not transcribed."""
    from unmark.modeling.config import AdapterConfig

    counts = AdapterConfig(hidden_size=768).parameter_count()
    assert counts.total == ADAPTER_TRAINABLE_PARAMETERS
    assert len(ADAPTER_STATE_KEYS) == ADAPTER_TENSOR_COUNT


def test_every_field_carries_a_classification():
    """The convention `docs/spec/stage1-final-freeze.json` established."""
    allowed = {
        "scientific_identity", "scientific_protocol", "scientific_input",
        "operational_acceptance", "operational_provenance", "safety_gate",
    }
    payload = freeze()
    for section, block in payload.items():
        if section in {"schema_version", "notes"}:
            continue
        for field, entry in block.items():
            assert "classification" in entry, f"{section}.{field}"
            assert entry["classification"] in allowed, f"{section}.{field}"


# ==========================================================================================
# Exactly two finalists, and no drift
# ==========================================================================================

def test_exactly_two_finalists():
    assert len(FINALISTS) == FINALIST_COUNT == 2
    validate_finalists(list(FINALISTS))
    assert {f.key for f in FINALISTS} == {"A", "B"}


def test_a_third_finalist_is_refused():
    extra = FinalistIdentity(
        key="C",
        role="stage1_stability_rule_winner",
        source_stage="final_main",
        run_seed=36930,
        update=17000,          # the attractive-looking later checkpoint
        learning_rate=FROZEN_LEARNING_RATE,
        r=FROZEN_R,
        validation_score=0.08,
        source_repository_head=FINALIST_A.source_repository_head,
        checkpoint_sha256="c" * 64,
    )
    with pytest.raises(FinalistFreezeViolation, match="exactly 2 candidates"):
        validate_finalists([FINALIST_A, FINALIST_B, extra])


def test_a_dropped_finalist_is_refused():
    with pytest.raises(FinalistFreezeViolation, match="exactly 2 candidates"):
        validate_finalists([FINALIST_A])


def test_duplicate_finalists_are_refused():
    twin = FinalistIdentity(**{**FINALIST_A.__dict__, "key": "B", "role": FINALIST_B.role})
    with pytest.raises(FinalistFreezeViolation):
        validate_finalists([FINALIST_A, twin])


def test_the_same_checkpoint_cannot_be_both_finalists():
    same_hash = FinalistIdentity(
        **{**FINALIST_B.__dict__, "checkpoint_sha256": FINALIST_A.checkpoint_sha256}
    )
    with pytest.raises(FinalistFreezeViolation, match="same checkpoint sha256"):
        validate_finalists([FINALIST_A, same_hash])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("run_seed", 7309),
        ("update", 17000),
        ("source_repository_head", "d" * 40),
        ("validation_score", 0.05),
        ("source_stage", "r_phase1"),
        ("role", "mature_historical_lr_pilot_alternative"),
    ],
)
def test_mutating_a_pinned_finalist_field_is_refused(field, value):
    mutated = FinalistIdentity(**{**FINALIST_A.__dict__, field: value})
    with pytest.raises(FinalistFreezeViolation):
        validate_finalists([mutated, FINALIST_B])


@pytest.mark.parametrize("field,value", [("learning_rate", 3e-4), ("r", 2.0)])
def test_a_finalist_at_other_hyperparameters_is_refused(field, value):
    """LR and r are CLOSED; a checkpoint from elsewhere cannot enter here."""
    mutated = FinalistIdentity(**{**FINALIST_A.__dict__, field: value})
    with pytest.raises(FinalistFreezeViolation):
        validate_finalists([mutated, FINALIST_B])


def test_an_update_off_the_validation_cadence_is_refused():
    """The identity pin catches this first; the cadence guard backs it up."""
    mutated = FinalistIdentity(**{**FINALIST_A.__dict__, "update": 3501})
    with pytest.raises(FinalistFreezeViolation, match="not the frozen 3500"):
        validate_finalists([mutated, FINALIST_B])


def test_both_frozen_updates_lie_on_the_validation_cadence():
    from unmark.stage1.protocol import EVAL_EVERY_UPDATES

    for finalist in FINALISTS:
        assert finalist.update > 0
        assert finalist.update % EVAL_EVERY_UPDATES == 0, finalist.key


def test_finalist_lookup_refuses_an_unknown_key():
    with pytest.raises(FinalistFreezeViolation, match="unknown finalist"):
        finalist_for("C")


# ==========================================================================================
# Digest evidence: full or explicitly pending, never truncated
# ==========================================================================================

def test_a_full_digest_is_accepted():
    assert require_sha256("a" * 64, "x") == "a" * 64


def test_the_pending_sentinel_is_accepted_where_allowed():
    assert require_sha256(SHA256_PENDING, "x") == SHA256_PENDING


@pytest.mark.parametrize(
    "rejected",
    [
        "9405bd76c0493964",              # the observed prefix
        "9405bd76c0493964...",
        "a" * 63,
        "a" * 65,
        "A" * 64,                        # uppercase
        "g" * 64,                        # non-hex
        "",
        None,
        12345,
    ],
)
def test_a_truncated_or_malformed_digest_is_refused(rejected):
    with pytest.raises(FinalistFreezeViolation):
        require_sha256(rejected, "x")


def test_a_pending_digest_is_refused_where_evidence_is_required():
    with pytest.raises(FinalistFreezeViolation, match="requires the authoritative digest"):
        require_sha256(SHA256_PENDING, "x", allow_pending=False)


def test_finalist_b_digest_is_bound_to_authoritative_evidence():
    """Bound from the real checkpoint, never from the observed 16-char prefix.

    Until the authoritative Colab verification this was `SHA256_PENDING`. The
    prefix `9405bd76c0493964` had been visible the whole time and was deliberately
    not expanded; the bound value begins with it, which is a consistency check on
    the evidence, not the source of it.
    """
    assert FINALIST_B.sha256_bound
    assert FINALIST_B.checkpoint_sha256 == (
        "9405bd76c04939641170cb71507ce8eb669eb2987016b86b495a403ceafcb9d2"
    )
    assert len(FINALIST_B.checkpoint_sha256) == 64
    assert FINALIST_B.checkpoint_sha256.startswith("9405bd76c0493964")


def test_both_finalist_digests_are_bound_and_the_freeze_is_complete():
    for finalist in FINALISTS:
        assert finalist.sha256_bound, finalist.key
        assert len(finalist.checkpoint_sha256) == 64, finalist.key
    assert pending_finalists() == ()
    assert freeze_is_complete() is True


def test_finalist_a_digest_is_bound():
    assert FINALIST_A.sha256_bound
    assert len(FINALIST_A.checkpoint_sha256) == 64


def test_the_artifact_and_the_code_agree_about_what_is_pending():
    payload = freeze()
    assert payload["evidence"]["pending_finalist_digests"]["value"] == list(pending_finalists())
    assert payload["evidence"]["freeze_complete"]["value"] is freeze_is_complete()


def test_declaring_the_freeze_complete_while_a_digest_is_pending_is_refused(monkeypatch):
    """The generic guard, exercised by UN-binding B rather than by B's real state."""
    set_b_digest_in_code(monkeypatch, SHA256_PENDING)
    # The pending list is declared CORRECTLY, so the freeze_complete guard is the
    # one that must speak; declaring it wrongly would trip the earlier check.
    payload = set_b_digest_in_artifact(
        freeze(), SHA256_PENDING, complete=True, pending=["B"]
    )
    with pytest.raises(FinalistFreezeViolation, match="disagrees with the pending digests"):
        validate_freeze_payload(payload)


def test_a_pending_digest_must_be_declared_pending(monkeypatch):
    set_b_digest_in_code(monkeypatch, SHA256_PENDING)
    payload = set_b_digest_in_artifact(freeze(), SHA256_PENDING, complete=False, pending=[])
    with pytest.raises(FinalistFreezeViolation, match="pending digests"):
        validate_freeze_payload(payload)


# ==========================================================================================
# The adjudication boundary
# ==========================================================================================

def test_adjudication_is_open_and_no_adapter_is_selected():
    payload = freeze()["adjudication"]
    assert payload["status"]["value"] == "OPEN"
    assert payload["final_adapter_selected"]["value"] is False
    assert payload["winner"]["value"] is None
    assert payload["candidate_universe_extensible"]["value"] is False
    assert payload["downstream_results_seen"]["value"] is False
    assert payload["allowed_evidence"]["value"] == ADJUDICATION_ALLOWED_EVIDENCE


def test_a_winner_cannot_be_recorded_without_an_adjudication_contract():
    payload = freeze()
    payload["adjudication"]["winner"]["value"] = "A"
    with pytest.raises(FinalistFreezeViolation, match="no adjudication protocol"):
        validate_freeze_payload(payload)


def test_marking_the_final_adapter_selected_is_refused():
    payload = freeze()
    payload["adjudication"]["final_adapter_selected"]["value"] = True
    with pytest.raises(FinalistFreezeViolation, match="records a selected final adapter"):
        validate_freeze_payload(payload)


def test_claiming_downstream_results_were_seen_is_refused():
    payload = freeze()
    payload["adjudication"]["downstream_results_seen"]["value"] = True
    with pytest.raises(FinalistFreezeViolation, match="BEFORE any downstream result"):
        validate_freeze_payload(payload)


def test_reopening_the_candidate_universe_is_refused():
    payload = freeze()
    payload["adjudication"]["candidate_universe_extensible"]["value"] = True
    with pytest.raises(FinalistFreezeViolation, match="candidate universe as closed"):
        validate_freeze_payload(payload)


def test_a_third_finalist_in_the_artifact_is_refused():
    payload = freeze()
    entries = finalists_from(payload)
    third = copy.deepcopy(entries[0])
    third.update({"key": "C", "update": 17000, "checkpoint_sha256": "c" * 64})
    entries.append(third)
    with pytest.raises(FinalistFreezeViolation, match="exactly 2 candidates"):
        validate_freeze_payload(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [("run_seed", 7309), ("update", 17000), ("source_repository_head", "e" * 40)],
)
def test_editing_a_finalist_in_the_artifact_is_refused(field, value):
    payload = freeze()
    finalists_from(payload)[0][field] = value
    with pytest.raises(FinalistFreezeViolation):
        validate_freeze_payload(payload)


def test_an_unknown_finalist_field_is_refused():
    payload = freeze()
    finalists_from(payload)[0]["downstream_dev_accuracy"] = 0.91
    with pytest.raises(FinalistFreezeViolation, match="unknown field"):
        validate_freeze_payload(payload)


def test_a_partial_finalist_record_is_refused():
    payload = freeze()
    del finalists_from(payload)[0]["source_repository_head"]
    with pytest.raises(FinalistFreezeViolation, match="missing"):
        validate_freeze_payload(payload)


# ==========================================================================================
# Sealing and history
# ==========================================================================================

def test_the_official_test_is_sealed():
    sealing = freeze()["test_sealing"]
    assert sealing["official_uit_vsfc_test"]["value"] == "SEALED"
    assert sealing["official_test_used"]["value"] is False
    assert sealing["test_used_for_finalist_selection"]["value"] is False


@pytest.mark.parametrize("field", ["official_test_used", "test_used_for_finalist_selection"])
def test_marking_test_used_is_refused(field):
    payload = freeze()
    payload["test_sealing"][field]["value"] = True
    with pytest.raises(FinalistFreezeViolation, match=field):
        validate_freeze_payload(payload)


def test_unsealing_the_official_test_is_refused():
    payload = freeze()
    payload["test_sealing"]["official_uit_vsfc_test"]["value"] = "OPEN"
    with pytest.raises(FinalistFreezeViolation, match="SEALED"):
        validate_freeze_payload(payload)


def test_provisional_canonical_v1_is_preserved_and_not_final():
    block = freeze()["provisional_canonical_v1"]
    assert block["preserved"]["value"] is True
    assert block["is_final_adapter"]["value"] is False
    assert block["source_sha256"]["value"] == FINALIST_A.checkpoint_sha256
    assert block["source_finalist"]["value"] == "A"


def test_canonical_v1_cannot_masquerade_as_the_final_adapter():
    payload = freeze()
    payload["provisional_canonical_v1"]["is_final_adapter"]["value"] = True
    with pytest.raises(FinalistFreezeViolation, match="masquerade"):
        validate_freeze_payload(payload)


def test_deleting_canonical_v1_history_is_refused():
    payload = freeze()
    payload["provisional_canonical_v1"]["preserved"]["value"] = False
    with pytest.raises(FinalistFreezeViolation, match="preserved, not deleted"):
        validate_freeze_payload(payload)


@pytest.mark.parametrize(
    "field", ["training", "candidate_generation", "learning_rate_selection", "r_selection"]
)
def test_reopening_a_closed_stage1_decision_is_refused(field):
    payload = freeze()
    payload["stage1"][field]["value"] = "OPEN"
    with pytest.raises(FinalistFreezeViolation):
        validate_freeze_payload(payload)


def test_a_wrong_schema_version_is_refused():
    payload = freeze()
    payload["schema_version"] = "stage1-finalist-freeze-v2"
    with pytest.raises(FinalistFreezeViolation, match="schema"):
        validate_freeze_payload(payload)


@pytest.mark.parametrize("field", ["revision", "protocol_version", "adapter_trainable_parameters"])
def test_a_wrong_model_identity_in_the_artifact_is_refused(field):
    payload = freeze()
    payload["model"][field]["value"] = "wrong" if isinstance(
        payload["model"][field]["value"], str
    ) else 1
    with pytest.raises(FinalistFreezeViolation):
        validate_freeze_payload(payload)


def test_a_missing_freeze_file_is_refused(tmp_path):
    with pytest.raises(FinalistFreezeViolation, match="missing"):
        load_freeze(tmp_path / "absent.json")


def test_a_malformed_freeze_file_is_refused(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(FinalistFreezeViolation, match="malformed"):
        load_freeze(bad)


# ==========================================================================================
# The two finalists came from DIFFERENT repository HEADs -- that must stay visible
# ==========================================================================================

def test_the_finalists_do_not_share_a_repository_head():
    """Repository HEAD is part of Stage-1 campaign identity in this project.

    The freeze must not quietly present A and B as one campaign identity. Audit
    048 states what equivalence evidence exists and what is not proven.
    """
    assert FINALIST_A.source_repository_head != FINALIST_B.source_repository_head
    heads = {entry["source_repository_head"] for entry in finalists_from(freeze())}
    assert len(heads) == 2


def test_the_provenance_caveat_is_documented():
    notes = " ".join(freeze()["notes"])
    assert "DIFFERENT repository HEADs" in notes


# ==========================================================================================
# Checkpoint verification gates that run BEFORE torch is needed
#
# The tensor-level half lives in `test_stage1_finalist_checkpoint_torch.py`: a
# module-level `importorskip` here would skip every structural check above it,
# which is exactly how an earlier repair in this repository lost its torch-free
# coverage (see `test_stage1_device_contract.py`).
# ==========================================================================================

def test_a_missing_checkpoint_is_refused(tmp_path):
    with pytest.raises(FinalistFreezeViolation, match="missing"):
        verify_finalist_checkpoint(tmp_path / "nope.pt", FINALIST_A)


def test_a_sha_mismatch_is_refused_before_torch_is_needed(tmp_path):
    """The digest gate runs first, so this is meaningful without torch."""
    path = tmp_path / "wrong.pt"
    path.write_bytes(b"not the frozen checkpoint")
    with pytest.raises(FinalistFreezeViolation, match="sha256 mismatch"):
        verify_finalist_checkpoint(path, FINALIST_A)


def test_a_pending_finalist_cannot_be_verified_without_its_digest(tmp_path):
    """The generic rule survives B's binding: an unbound finalist has nothing to
    verify against. Exercised on a synthetic unbound identity, because both real
    finalists are now bound."""
    unbound = FinalistIdentity(
        **{**FINALIST_B.__dict__, "checkpoint_sha256": SHA256_PENDING}
    )
    path = tmp_path / "b.pt"
    path.write_bytes(b"anything")
    with pytest.raises(FinalistFreezeViolation, match="no bound sha256"):
        verify_finalist_checkpoint(path, unbound)


def test_finalist_b_is_no_longer_blocked():
    """Replaces the former `blocked today` fact after authoritative verification."""
    assert FINALIST_B.sha256_bound
    assert pending_finalists() == ()
    load_freeze()          # the committed freeze validates in its bound state


# ==========================================================================================
# The verifier must not be WEAKER than the existing Stage-1 provenance contract
# ==========================================================================================

def test_the_verifier_covers_every_field_require_match_compares():
    """Read from `require_match` itself, so it cannot drift ahead of this gate."""
    import ast

    source = (pathlib.Path(__file__).resolve().parents[1]
              / "unmark/stage1/trainer.py").read_text(encoding="utf-8")
    function = next(
        node for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.FunctionDef) and node.name == "require_match"
    )
    compared = set()
    for node in ast.walk(function):
        if isinstance(node, ast.For) and isinstance(node.iter, ast.Tuple):
            compared |= {e.value for e in node.iter.elts if isinstance(e, ast.Constant)}

    from unmark.stage1.trainer import RunProvenance

    compared |= set(RunProvenance.DERIVED_KEYS)
    assert compared, "could not read the compared field list out of require_match"
    assert not compared - set(VERIFIED_PROVENANCE_FIELDS), (
        f"the finalist verifier does not cover {sorted(compared - set(VERIFIED_PROVENANCE_FIELDS))}"
    )
    assert not set(VERIFIED_PROVENANCE_FIELDS) - compared, (
        "the finalist verifier advertises fields require_match does not compare"
    )


def test_the_verifier_delegates_to_the_authoritative_checkpoint_gate():
    """AST, not grep: prose mentioning verify_checkpoint would match a substring."""
    import ast

    source = (pathlib.Path(__file__).resolve().parents[1]
              / "unmark/stage1/finalists.py").read_text(encoding="utf-8")
    called = {
        node.func.id
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "verify_checkpoint" in called, (
        "verify_finalist_checkpoint must reuse the authoritative checkpoint gate, "
        "not re-implement a second, weaker parser"
    )


def test_expected_provenance_is_derived_not_read_from_the_artifact():
    from unmark.stage1.protocol import CORRUPTION_SEED, adapter_init_seed, lambdas_for_r

    inventory = resolve_inventory()
    expected = expected_run_provenance(FINALIST_A, inventory=inventory)
    assert expected.run_seed == FINALIST_A.run_seed
    assert expected.init_seed == adapter_init_seed(FINALIST_A.run_seed)
    assert expected.corruption_seed == CORRUPTION_SEED
    assert expected.corpus_manifest_digest == CORPUS_MANIFEST_DIGEST
    assert expected.repository_head == FINALIST_A.source_repository_head
    lambda_align, lambda_clean = lambdas_for_r(FINALIST_A.r)
    recorded = expected.to_dict()
    assert recorded["lambda_align"] == lambda_align
    assert recorded["lambda_clean"] == lambda_clean
    assert set(recorded) == set(VERIFIED_PROVENANCE_FIELDS)


def test_the_two_finalists_get_different_init_seeds():
    """A pinned init_seed is what stops one finalist's identity matching the other."""
    a = expected_run_provenance(FINALIST_A, inventory=resolve_inventory())
    b = expected_run_provenance(FINALIST_B, inventory=resolve_inventory())
    assert a.init_seed != b.init_seed
    assert (a.init_seed, b.init_seed) == (51800, 3203)


# ==========================================================================================
# Corpus and inventory identity
# ==========================================================================================

def test_the_corpus_digest_agrees_with_the_pre_training_freeze():
    """The two committed specs must not drift apart."""
    pre = json.loads(
        (pathlib.Path(__file__).resolve().parents[1]
         / "docs/spec/stage1-final-freeze.json").read_text(encoding="utf-8")
    )
    assert pre["data"]["chunk_membership_digest"]["value"] == CORPUS_MANIFEST_DIGEST
    assert pre["data"]["syllable_inventory_sha256"]["value"] == INVENTORY_SHA256
    assert freeze()["corpus"]["chunk_membership_digest"]["value"] == CORPUS_MANIFEST_DIGEST
    assert freeze()["inventory"]["sha256"]["value"] == INVENTORY_SHA256


def test_the_resolved_inventory_matches_the_pins():
    inventory = resolve_inventory()
    assert inventory.sha256 == INVENTORY_SHA256
    assert inventory.source_revision == INVENTORY_SOURCE_REVISION
    assert inventory.size_bytes == INVENTORY_SIZE_BYTES


@pytest.mark.parametrize("field", ["sha256", "source_revision", "size_bytes"])
def test_a_wrong_inventory_pin_in_the_artifact_is_refused(field):
    payload = freeze()
    payload["inventory"][field]["value"] = "wrong" if field != "size_bytes" else 1
    with pytest.raises(FinalistFreezeViolation, match="inventory"):
        validate_freeze_payload(payload)


def test_a_wrong_corpus_digest_in_the_artifact_is_refused():
    payload = freeze()
    payload["corpus"]["chunk_membership_digest"]["value"] = "0" * 64
    with pytest.raises(FinalistFreezeViolation, match="corpus membership digest"):
        validate_freeze_payload(payload)


def test_the_artifact_may_not_advertise_a_different_verified_field_set():
    payload = freeze()
    payload["evidence"]["provenance_fields_verified"]["value"] = ["run_seed"]
    with pytest.raises(FinalistFreezeViolation, match="provenance_fields_verified"):
        validate_freeze_payload(payload)


# ==========================================================================================
# Binding finalist B is a FOUR-field edit; partial binding is refused
# ==========================================================================================

OTHER_DIGEST = "b" * 64   # a well-formed digest that is not either finalist's


def set_b_digest_in_code(monkeypatch, digest):
    """Point the code constant for B at `digest` (bound or the pending sentinel)."""
    import unmark.stage1.finalists as module

    replaced = FinalistIdentity(**{**FINALIST_B.__dict__, "checkpoint_sha256": digest})
    monkeypatch.setattr(module, "FINALIST_B", replaced)
    return replaced


def set_b_digest_in_artifact(payload, digest, *, pending=None, complete=None):
    for entry in finalists_from(payload):
        if entry["key"] == "B":
            entry["checkpoint_sha256"] = digest
    if pending is not None:
        payload["evidence"]["pending_finalist_digests"]["value"] = pending
    if complete is not None:
        payload["evidence"]["freeze_complete"]["value"] = complete
    return payload


def test_the_fully_bound_freeze_validates():
    """The committed artifact is now bound in all four places."""
    validate_freeze_payload(freeze())         # must not raise


def test_the_binding_workflow_is_enumerated_in_code_and_artifact():
    assert len(BINDING_FIELDS) == 4
    assert freeze()["evidence"]["binding_workflow"]["value"] == list(BINDING_FIELDS)


def test_a_digest_present_in_the_artifact_but_not_the_code_is_refused():
    payload = set_b_digest_in_artifact(freeze(), OTHER_DIGEST)
    with pytest.raises(FinalistFreezeViolation, match="field edit"):
        validate_freeze_payload(payload)


def test_a_digest_present_in_the_code_but_not_the_artifact_is_refused(monkeypatch):
    set_b_digest_in_code(monkeypatch, OTHER_DIGEST)
    with pytest.raises(FinalistFreezeViolation, match="field edit"):
        validate_freeze_payload(freeze())


def test_a_stale_pending_entry_beside_a_bound_digest_is_refused():
    """Every digest is bound, but the artifact still declares B pending."""
    payload = freeze()
    payload["evidence"]["pending_finalist_digests"]["value"] = ["B"]
    with pytest.raises(FinalistFreezeViolation, match="pending digests"):
        validate_freeze_payload(payload)


def test_a_bound_freeze_that_denies_being_complete_is_refused():
    payload = freeze()
    payload["evidence"]["freeze_complete"]["value"] = False
    with pytest.raises(FinalistFreezeViolation, match="disagrees with the pending digests"):
        validate_freeze_payload(payload)


def test_a_bound_freeze_reports_itself_complete():
    assert freeze_is_complete(list(FINALISTS)) is True
    assert pending_finalists(list(FINALISTS)) == ()


def test_a_truncated_digest_is_refused_even_when_consistent(monkeypatch):
    """The exact prefix that was visible before the authoritative digest existed."""
    set_b_digest_in_code(monkeypatch, "9405bd76c0493964")
    payload = set_b_digest_in_artifact(
        freeze(), "9405bd76c0493964", pending=[], complete=True
    )
    with pytest.raises(FinalistFreezeViolation, match="64 lowercase hex"):
        validate_freeze_payload(payload)


# ==========================================================================================
# Torch-free guard on the AUTHORITATIVE failure messages
#
# The real-torch Colab run of Audit 048 failed two tests whose regexes still
# named the pre-delegation wording ("no provenance" / "no adapter_state"). The
# verifier was correct; the expectations were stale. `verify_checkpoint` and
# `require_match` need no torch, so that drift was locally detectable and simply
# was not checked. These tests close that gap: if the contract's message shape
# changes, it fails here rather than on the authoritative host.
# ==========================================================================================

def test_a_missing_required_key_is_reported_by_the_authoritative_contract():
    from unmark.stage1.trainer import (
        CHECKPOINT_SCHEMA_VERSION,
        REQUIRED_CHECKPOINT_KEYS,
        TrainerContractViolation,
        verify_checkpoint,
    )

    expected = expected_run_provenance(FINALIST_A, inventory=resolve_inventory())

    def payload():
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "provenance": expected.to_dict(),
            "adapter_state": {"tone_embedding.weight": 1},
            "optimizer_state": {}, "global_update": FINALIST_A.update,
            "sampler_state": {}, "cap": 20000, "points": [], "execution": {},
        }

    for key in REQUIRED_CHECKPOINT_KEYS:
        broken = payload()
        del broken[key]
        with pytest.raises(TrainerContractViolation) as excinfo:
            verify_checkpoint(broken, expected)
        message = str(excinfo.value)
        assert "checkpoint is missing" in message, (key, message)
        assert key in message, (key, message)


def test_the_torch_tests_assert_the_contract_wording_not_the_old_local_wording():
    """Guards against reintroducing the exact regexes the Colab run rejected."""
    import ast

    path = pathlib.Path(__file__).resolve().parents[1] / (
        "tests/test_stage1_finalist_checkpoint_torch.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    patterns = {
        kw.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for kw in node.keywords
        if kw.arg == "match" and isinstance(kw.value, ast.Constant)
        and isinstance(kw.value.value, str)
    }
    for retired in ("no provenance", "no adapter_state"):
        assert retired not in patterns, (
            f"{retired!r} is the pre-delegation wording; the authoritative gate now "
            "reports 'checkpoint is missing [...]'"
        )
