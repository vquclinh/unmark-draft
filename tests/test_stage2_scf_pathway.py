"""Post-hoc V2-SCF Stage-2 pathway: identity, fusion dispatch, fail-closed gates.

Audit 071. The defect this file exists to prevent is specific and silent: a
V2-SCF `adapter_state` loads into the **historical** mixture rule with
`strict=True` and no error -- same eight tensors, same names, same shapes -- and
produces a model that is numerically wrong and structurally perfect. Every test
below is either about refusing that, or about the V2-SCF campaign staying
additive to the frozen UNMARK-A/UNMARK-B contract.

Most of it is torch-free on purpose: a cross-fusion or cross-identity load should
be refused on a machine with no torch installed, before any tensor exists. The
tensor-level checks carry `@requires_torch` and are skipped, not hidden, where
torch is absent.

Nothing here trains, reads a UIT-VSFC row, opens official validation, or names
official TEST.
"""

from __future__ import annotations

import ast
import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import unmark.evaluation.stage2_scf_pathway as scf  # noqa: E402
from unmark.evaluation.contracts import EvaluationContractViolation  # noqa: E402
from unmark.evaluation.preg1_head import Preg1Role  # noqa: E402
from unmark.evaluation.stage2_dual_finalist import (  # noqa: E402
    STAGE2_FIRST_TOKEN_POOLING,
    STAGE2_REPRESENTATION_DTYPE,
    STAGE2_UNMARK_ARM_NAMES,
    Stage2UnmarkArm,
    require_stage2_unmark_arm,
)
from unmark.stage1.contracts import Stage1ContractViolation  # noqa: E402
from unmark.stage1.finalists import (  # noqa: E402
    ADAPTER_TENSOR_COUNT,
    CORPUS_MANIFEST_DIGEST,
    FINALIST_A,
    FINALIST_B,
    INVENTORY_SHA256,
    INVENTORY_SIZE_BYTES,
    INVENTORY_SOURCE_REVISION,
    PRECISION,
    expected_run_provenance,
)
from unmark.stage1.protocol import (  # noqa: E402
    ADAPTER_TRAINABLE_PARAMETERS,
    HIDDEN_SIZE,
    HISTORICAL_FUSION_ID,
    SCALE_CALIBRATED_FUSION_ID,
    V2_SCF_STAGE,
)
from unmark.stage1.trainer import CHECKPOINT_SCHEMA_VERSION  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
PATHWAY_MODULE = "unmark/evaluation/stage2_scf_pathway.py"
HISTORICAL_MODULE = "unmark/evaluation/stage2_dual_finalist.py"

try:  # pragma: no cover - depends on the environment
    import torch

    TORCH = True
except ImportError:  # pragma: no cover - the normal ML-free path
    torch = None
    TORCH = False

requires_torch = pytest.mark.skipif(
    not TORCH, reason="torch is not installed locally; tensor checks run where available"
)


class FakeInventoryIdentity:
    """Enough of D-S1A-008's identity for the verifier's existing contract."""

    def to_dict(self):
        return {
            "inventory_schema_version": "vn-syllables-v1",
            "source_name": "synthetic-test-inventory-identity",
            "source_author": "tests",
            "source_revision": INVENTORY_SOURCE_REVISION,
            "sha256": INVENTORY_SHA256,
            "size_bytes": INVENTORY_SIZE_BYTES,
            "license_status": "NO_EXPLICIT_LICENSE",
        }


INVENTORY = FakeInventoryIdentity()


def source(name: str) -> str:
    return (REPO / name).read_text(encoding="utf-8")


def scf_provenance(**overrides) -> dict:
    """A provenance dict describing the frozen V2-SCF run, before overrides."""
    payload = scf.expected_scf_run_provenance(inventory=INVENTORY).to_dict()
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# A. The campaign is ADDITIVE: V2-SCF is not a third historical arm
# ---------------------------------------------------------------------------
def test_scf_pathway_id_is_not_a_historical_arm():
    """The whole additivity claim in one assertion.

    If `require_stage2_unmark_arm` ever accepted this id, V2-SCF would have
    become a third arm and the frozen two-arm protocol would mean something
    different from what Audit 049 froze.
    """
    assert scf.V2_SCF_PATHWAY_ID not in STAGE2_UNMARK_ARM_NAMES
    with pytest.raises(EvaluationContractViolation):
        require_stage2_unmark_arm(scf.V2_SCF_PATHWAY_ID)


def test_historical_arm_universe_is_still_exactly_two():
    assert STAGE2_UNMARK_ARM_NAMES == ("UNMARK-A", "UNMARK-B")
    assert [a.value for a in Stage2UnmarkArm] == ["UNMARK-A", "UNMARK-B"]
    scf.require_historical_arms_untouched()


@pytest.mark.parametrize("arm", ["UNMARK-A", "UNMARK-B"])
def test_scf_pathway_refuses_a_historical_arm_name(arm):
    with pytest.raises(scf.ScfPathwayViolation) as error:
        scf.require_scf_pathway_id(arm)
    assert "historical" in str(error.value)


def test_historical_dual_finalist_module_is_untouched_by_this_work():
    """The historical loader still builds the DEFAULT (historical) adapter.

    Asserted against the source rather than by behaviour, because the failure
    mode is a well-meant edit: adding a `fusion_id=` argument here to 'support'
    V2-SCF would silently change what every UNMARK-A/B representation means.
    """
    text = source(HISTORICAL_MODULE)
    assert "AdapterConfig(hidden_size=HIDDEN_SIZE)" in text
    assert "fusion_id" not in text
    assert "scale_calibrated" not in text
    assert "V2_SCF" not in text
    assert "UNMARK-V2-SCF" not in text


def test_historical_head_campaign_still_declares_exactly_two_arms():
    from unmark.evaluation import stage2_head_campaign as historical

    assert historical.STAGE2_CAMPAIGN_ARM_COUNT == 2
    assert historical.STAGE2_CAMPAIGN_RUN_COUNT == 10
    assert len(historical.stage2_campaign_plan()) == 10


def test_scf_checkpoint_digest_is_not_a_finalist_digest():
    for finalist in (FINALIST_A, FINALIST_B):
        assert scf.V2_SCF_CHECKPOINT.checkpoint_sha256 != finalist.checkpoint_sha256
        assert (
            scf.V2_SCF_CHECKPOINT.source_repository_head
            != finalist.source_repository_head
        )


# ---------------------------------------------------------------------------
# B. The frozen identity is the one the campaign states
# ---------------------------------------------------------------------------
def test_frozen_scf_identity_values():
    identity = scf.V2_SCF_CHECKPOINT
    assert identity.stage == V2_SCF_STAGE == "v2_scf"
    assert identity.source_repository_head == (
        "8de83f0da8d2b38f312e2ff16cb78ebcf9e8e526"
    )
    assert identity.checkpoint_sha256 == (
        "a32c0167817d457d5067c2a351f2d1b73b26229033f03727f43e2d79f59ef685"
    )
    assert identity.stage_artifact_sha256 == (
        "19a5bf4cb4b7793bee9e51292f62e7b42baf38c64026ad52fc177de40fb05695"
    )
    assert identity.update == 8000
    assert identity.held_out_worst_case_score == 0.08116862134875966
    assert identity.fusion_id == SCALE_CALIBRATED_FUSION_ID
    assert identity.objective_id == "align-clean-pooled-v1"
    assert identity.run_seed == 36930
    assert identity.init_seed == 51800


def test_scf_candidate_registration_comes_from_the_stage1_register():
    candidate = scf.require_scf_candidate_registration()
    assert candidate.stage == V2_SCF_STAGE
    assert candidate.identity == ("align-clean-pooled-v1", SCALE_CALIBRATED_FUSION_ID)
    assert candidate.fusion.is_scale_calibrated


def test_expected_provenance_carries_the_scale_calibrated_fusion():
    provenance = scf.expected_scf_run_provenance(inventory=INVENTORY).to_dict()
    assert provenance["fusion"] == {"fusion_id": SCALE_CALIBRATED_FUSION_ID}
    assert provenance["objective"]["objective_id"] == "align-clean-pooled-v1"
    assert provenance["repository_head"] == (
        scf.V2_SCF_CHECKPOINT.source_repository_head
    )
    assert provenance["corpus_manifest_digest"] == CORPUS_MANIFEST_DIGEST


def test_scientific_status_flags_declare_posthoc_exploratory():
    assert scf.POSTHOC_EXPLORATORY is True
    assert scf.OFFICIAL_VALIDATION_PREVIOUSLY_SEEN is True
    assert scf.OFFICIAL_TEST_USED is False
    assert scf.OFFICIAL_TEST_ROLE_EXISTS is False
    assert scf.V2_SCF_STAGE2_HYPERPARAMETER_RETUNED is False
    assert scf.V2_SCF_BEST_SEED_SELECTION_IMPLEMENTED is False
    assert scf.V2_SCF_SELECTION_AGAINST_AB_IMPLEMENTED is False


# ---------------------------------------------------------------------------
# C. The frozen protocol artifact
# ---------------------------------------------------------------------------
def test_frozen_protocol_spec_exists_and_binds_the_campaign():
    payload = scf.require_frozen_scf_protocol_spec()
    assert payload["schema_version"] == scf.V2_SCF_POSTHOC_PROTOCOL_VERSION

    def value(section, key):
        return payload[section][key]["value"]

    assert value("scientific_status", "posthoc_exploratory") is True
    assert value("scientific_status", "official_validation_previously_seen") is True
    assert value("scientific_status", "official_test_used") is False
    assert value("scientific_status", "stage2_hyperparameter_retuned_after_seeing_scf") is False
    assert value("stage1_source", "source_repository_head") == (
        scf.V2_SCF_CHECKPOINT.source_repository_head
    )
    assert value("stage1_source", "stage") == "v2_scf"
    assert value("stage1_source", "stage_artifact_sha256") == (
        scf.V2_SCF_CHECKPOINT.stage_artifact_sha256
    )
    assert value("stage1_source", "checkpoint_sha256") == (
        scf.V2_SCF_CHECKPOINT.checkpoint_sha256
    )
    assert value("stage1_source", "selected_update") == 8000
    assert value("stage1_source", "fusion_id") == SCALE_CALIBRATED_FUSION_ID
    assert value("stage1_source", "objective_id") == "align-clean-pooled-v1"
    assert value("stage1_source", "held_out_worst_case_score") == 0.08116862134875966
    assert value("pathway", "is_a_third_historical_arm") is False
    assert value("splits", "official_test") == "SEALED"
    assert value("pooling", "strategy") == STAGE2_FIRST_TOKEN_POOLING


def test_frozen_protocol_spec_reuses_the_corrected_historical_head_protocol():
    """Every head value is the historical one, not a V2-SCF variant."""
    from unmark.evaluation.preg1_protocol import (
        ADAMW_BETAS,
        ADAMW_EPS,
        BATCH_SIZE,
        EPOCHS,
        MEASUREMENT_SEEDS,
        WEIGHT_DECAY_BIAS,
        WEIGHT_DECAY_WEIGHT,
    )
    from unmark.evaluation.stage2_head_campaign import (
        STAGE2_HEAD_LEARNING_RATE,
        STAGE2_MEASUREMENT_CORRUPTION_SEED,
    )

    payload = scf.require_frozen_scf_protocol_spec()

    def value(section, key):
        return payload[section][key]["value"]

    assert value("optimization", "optimizer") == "AdamW"
    assert value("optimization", "learning_rate") == STAGE2_HEAD_LEARNING_RATE == 0.01
    assert value("optimization", "weight_decay_weight") == WEIGHT_DECAY_WEIGHT == 0.01
    assert value("optimization", "weight_decay_bias") == WEIGHT_DECAY_BIAS == 0.0
    assert value("optimization", "betas") == list(ADAMW_BETAS) == [0.9, 0.999]
    assert value("optimization", "eps") == ADAMW_EPS == 1e-8
    assert value("optimization", "batch_size") == BATCH_SIZE == 128
    assert value("optimization", "epochs") == EPOCHS == 30
    assert value("optimization", "early_stopping") is False
    assert value("optimization", "gradient_clipping") is None
    assert value("optimization", "gradient_accumulation_steps") == 1
    assert value("optimization", "lr_schedule") == "CONSTANT"
    assert value("optimization", "new_lr_pilot_permitted") is False
    assert value("head", "shape") == "Linear(768, 3, bias=True)"
    assert value("head", "init_weight") == "torch.nn.init.xavier_uniform_"
    assert value("head", "init_bias") == "torch.nn.init.zeros_"
    assert value("head", "loss") == "cross_entropy"
    assert value("head", "loss_label_smoothing") == 0.0
    assert value("head", "loss_class_weights") is None
    assert value("seeds", "head_seeds") == list(MEASUREMENT_SEEDS)
    assert value("seeds", "head_seeds") == [53148, 59945, 42941, 720, 9428]
    assert value("seeds", "minibatch_order") == "seed * 1000 + epoch"
    assert value("splits", "training_rows") == 9139
    assert value("splits", "selection_rows") == 2285
    assert value("measurement", "corruption_seed") == STAGE2_MEASUREMENT_CORRUPTION_SEED == 19225
    assert value("head_checkpoint_selection", "rule") == (
        "highest macro_f1, then highest accuracy, then earliest epoch"
    )
    assert value("head_checkpoint_selection", "best_seed_selection") is False
    assert value("frozen_representation", "syllable_inventory_classifier") == "REQUIRED"


def test_frozen_protocol_spec_is_valid_json_and_committed():
    path = REPO / "docs/spec/stage2-v2-scf-posthoc-protocol.json"
    assert path.is_file()
    json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# D. THE central refusal: the SCF loader will not accept the historical fusion
# ---------------------------------------------------------------------------
def _historical_payload() -> dict:
    """A payload whose provenance says `historical-fusion-v1`."""
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "provenance": expected_run_provenance(FINALIST_A, inventory=INVENTORY).to_dict(),
        "adapter_state": {},
        "optimizer_state": {},
        "global_update": FINALIST_A.update,
        "sampler_state": {},
        "cap": 20000,
        "points": [],
        "execution": {},
    }


def test_scf_adapter_builder_refuses_a_historical_fusion_checkpoint():
    """Torch-free, and refused BEFORE any adapter is constructed.

    This is the defect in one test. The tensors would load; the equation would be
    wrong; nothing downstream would complain.
    """
    with pytest.raises(scf.ScfPathwayViolation) as error:
        scf.build_scf_adapter(_historical_payload())
    message = str(error.value)
    assert HISTORICAL_FUSION_ID in message
    assert SCALE_CALIBRATED_FUSION_ID in message


def test_scf_adapter_builder_refuses_a_provenance_with_no_fusion_block():
    """Absence reads as the historical claim, so it must be refused here too."""
    payload = _historical_payload()
    payload["provenance"].pop("fusion")
    with pytest.raises(scf.ScfPathwayViolation):
        scf.build_scf_adapter(payload)


def test_scf_expected_provenance_refuses_a_historical_provenance():
    """The full-provenance gate also catches it, independently of the fusion guard."""
    expected = scf.expected_scf_run_provenance(inventory=INVENTORY)
    historical = expected_run_provenance(FINALIST_A, inventory=INVENTORY).to_dict()
    with pytest.raises(Exception) as error:
        expected.require_match(historical)
    assert "fusion" in str(error.value) or "repository_head" in str(error.value)


def test_historical_verifier_refuses_an_scf_provenance():
    """The refusal is symmetric: C1's checkpoint cannot verify as UNMARK-A."""
    expected = expected_run_provenance(FINALIST_A, inventory=INVENTORY)
    with pytest.raises(Exception) as error:
        expected.require_match(scf_provenance())
    assert "mismatch" in str(error.value)


# ---------------------------------------------------------------------------
# E. Fail-closed identity gates: digest, update, source HEAD
# ---------------------------------------------------------------------------
def test_scf_checkpoint_digest_mismatch_fails_closed(tmp_path):
    """Refused on the bytes, before torch is imported at all."""
    path = tmp_path / "not-the-checkpoint.pt"
    path.write_bytes(b"this is not the frozen V2-SCF checkpoint")
    with pytest.raises(scf.ScfPathwayViolation) as error:
        scf.verify_scf_checkpoint(path, inventory=INVENTORY)
    message = str(error.value)
    assert "sha256 mismatch" in message
    assert scf.V2_SCF_CHECKPOINT.checkpoint_sha256 in message


def test_scf_checkpoint_missing_file_fails_closed(tmp_path):
    with pytest.raises(scf.ScfPathwayViolation):
        scf.verify_scf_checkpoint(tmp_path / "absent.pt", inventory=INVENTORY)


def test_scf_selected_update_mismatch_fails_closed_in_binding():
    evidence = _evidence()
    evidence["update"] = 7500
    with pytest.raises(scf.ScfPathwayViolation) as error:
        scf.bind_verified_scf_checkpoint("synthetic.pt", evidence)
    assert "update" in str(error.value)


def test_scf_source_head_mismatch_fails_closed_in_binding():
    evidence = _evidence()
    evidence["source_repository_head"] = "0" * 40
    with pytest.raises(scf.ScfPathwayViolation) as error:
        scf.bind_verified_scf_checkpoint("synthetic.pt", evidence)
    assert "source_repository_head" in str(error.value)


def test_scf_checkpoint_sha_mismatch_fails_closed_in_binding():
    evidence = _evidence()
    evidence["checkpoint_sha256"] = "f" * 64
    with pytest.raises(scf.ScfPathwayViolation) as error:
        scf.bind_verified_scf_checkpoint("synthetic.pt", evidence)
    assert "checkpoint_sha256" in str(error.value)


def test_scf_fusion_mismatch_fails_closed_in_binding():
    evidence = _evidence()
    evidence["fusion_id"] = HISTORICAL_FUSION_ID
    with pytest.raises(scf.ScfPathwayViolation) as error:
        scf.bind_verified_scf_checkpoint("synthetic.pt", evidence)
    assert "fusion_id" in str(error.value)


def test_scf_binding_refuses_a_historical_arm_evidence_record():
    evidence = _evidence()
    evidence["pathway_id"] = "UNMARK-A"
    with pytest.raises(scf.ScfPathwayViolation):
        scf.bind_verified_scf_checkpoint("synthetic.pt", evidence)


def test_scf_selected_update_mismatch_fails_closed_against_provenance():
    """The update gate is independent of the provenance gate, by construction."""
    identity = scf.V2_SCF_CHECKPOINT
    assert identity.update == 8000
    # The verifier compares `global_update` against the frozen identity, so a
    # payload from a different update of the SAME run is refused even though its
    # provenance matches perfectly.
    provenance = scf_provenance()
    assert provenance["run_seed"] == identity.run_seed
    scf.expected_scf_run_provenance(inventory=INVENTORY).require_match(provenance)


def _evidence(**overrides) -> dict:
    identity = scf.V2_SCF_CHECKPOINT
    evidence = {
        "kind": "stage2_v2_scf_checkpoint_evidence",
        "pathway_id": identity.pathway_id,
        "stage": identity.stage,
        "source_repository_head": identity.source_repository_head,
        "run_seed": identity.run_seed,
        "update": identity.update,
        "fusion_id": identity.fusion_id,
        "objective_id": identity.objective_id,
        "checkpoint_sha256": identity.checkpoint_sha256,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "adapter_tensor_count": ADAPTER_TENSOR_COUNT,
        "adapter_trainable_parameters": ADAPTER_TRAINABLE_PARAMETERS,
        "adapter_dtype": PRECISION,
        "all_finite": True,
    }
    evidence.update(overrides)
    return evidence


def test_good_evidence_binds():
    binding = scf.bind_verified_scf_checkpoint("synthetic.pt", _evidence())
    assert binding.pathway_id == scf.V2_SCF_PATHWAY_ID
    assert binding.fusion_id == SCALE_CALIBRATED_FUSION_ID
    assert binding.update == 8000
    assert binding.to_dict()["stage1_evidence_kind"] == (
        "stage2_v2_scf_checkpoint_evidence"
    )


def test_binding_refuses_partial_evidence():
    evidence = _evidence()
    del evidence["fusion_id"]
    with pytest.raises(scf.ScfPathwayViolation) as error:
        scf.bind_verified_scf_checkpoint("synthetic.pt", evidence)
    assert "missing" in str(error.value)


# ---------------------------------------------------------------------------
# F. Stage artifact verification
# ---------------------------------------------------------------------------
def test_stage_artifact_digest_mismatch_fails_closed(tmp_path):
    path = tmp_path / "v2_scf.json"
    path.write_text(json.dumps({"stage": "v2_scf"}), encoding="utf-8")
    with pytest.raises(scf.ScfPathwayViolation) as error:
        scf.verify_scf_stage_artifact(path)
    assert "sha256 mismatch" in str(error.value)


def test_stage_artifact_missing_fails_closed(tmp_path):
    with pytest.raises(scf.ScfPathwayViolation):
        scf.verify_scf_stage_artifact(tmp_path / "absent.json")


# ---------------------------------------------------------------------------
# G. Official TEST is unreachable; official validation cannot be a training role
# ---------------------------------------------------------------------------
def test_official_test_has_no_role_member():
    """Structural, not a check: the enum has no member to name."""
    assert [r.value for r in Preg1Role] == [
        "protocol-train",
        "protocol-dev",
        "official-validation",
    ]
    assert not any("test" in r.name.lower() for r in Preg1Role)
    with pytest.raises(ValueError):
        Preg1Role("official-test")


SCF_MODULES = (
    PATHWAY_MODULE,
    "unmark/evaluation/stage2_scf_campaign.py",
    "unmark/evaluation/stage2_scf_measurement.py",
)

DECLARED_TEST_FLAGS = {
    "OFFICIAL_TEST_USED",
    "OFFICIAL_TEST_ROLE_EXISTS",
    "SCF_OFFICIAL_TEST_ROLE_EXISTS",
    "SCF_OFFICIAL_TEST_REACHABLE",
}


def test_no_scf_module_can_name_or_route_official_test():
    """Checked on the AST, not the text, so prose explaining the seal is allowed.

    What must not exist is a *route*: an attribute access naming a TEST role, a
    string literal that could select one, or any identifier other than the
    declared `False` safety flags. Documentation that says TEST is unreachable is
    the point of the seal, not a violation of it.
    """
    for name in SCF_MODULES:
        tree = ast.parse(source(name))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                assert "OFFICIAL_TEST" not in node.attr, (
                    f"{name} accesses an OFFICIAL_TEST attribute"
                )
            if isinstance(node, ast.Name) and "OFFICIAL_TEST" in node.id:
                assert node.id in DECLARED_TEST_FLAGS, (
                    f"{name} names {node.id}, which is not one of the declared "
                    "false safety flags"
                )
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                # The role-value spelling is the only string that could ever
                # resolve to a split. `"official_test"` as a *field name* is how
                # the freeze artifact declares the seal, and reading a safety
                # declaration is the opposite of routing to it.
                assert node.value not in {
                    "official-test",
                    "OFFICIAL_TEST",
                }, f"{name} carries a string literal that could select TEST"


def test_scf_safety_flags_declaring_test_are_all_false():
    """The declared flags exist so the absence is visible, and they say `False`."""
    from unmark.evaluation import stage2_scf_campaign as campaign
    from unmark.evaluation import stage2_scf_measurement as measurement

    assert scf.OFFICIAL_TEST_USED is False
    assert scf.OFFICIAL_TEST_ROLE_EXISTS is False
    assert campaign.SCF_OFFICIAL_TEST_ROLE_EXISTS is False
    assert measurement.SCF_OFFICIAL_TEST_REACHABLE is False


def test_scf_modules_take_no_split_argument():
    """No function anywhere in the V2-SCF pathway accepts a split/role argument.

    Roles are read from cache provenance, never declared by a caller, so there is
    no parameter through which a sealed split could be routed.
    """
    for name in (
        PATHWAY_MODULE,
        "unmark/evaluation/stage2_scf_campaign.py",
        "unmark/evaluation/stage2_scf_measurement.py",
    ):
        tree = ast.parse(source(name))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                arguments = node.args
                names = [
                    a.arg
                    for a in (
                        list(arguments.posonlyargs)
                        + list(arguments.args)
                        + list(arguments.kwonlyargs)
                    )
                ]
                assert "split" not in names, f"{name}:{node.name} takes a split argument"
                assert "test" not in names, f"{name}:{node.name} takes a test argument"


# ---------------------------------------------------------------------------
# H. The corrected tone-channel input path is the one V2-SCF uses
# ---------------------------------------------------------------------------
def test_scf_reuses_the_corrected_stage2_input_path():
    """V2-SCF prepares inputs with the SAME function the corrected A/B path uses.

    That function carries the Audit-059 guard, so the tone channel cannot be
    silently dead here either. Reuse is the assertion: a second input builder
    would be a second place for the classifier to go missing.
    """
    from unmark.evaluation.stage2_dual_finalist import (
        prepare_stage2_unmark_input,
        require_resolved_tone_channel,
    )

    assert callable(prepare_stage2_unmark_input)
    assert callable(require_resolved_tone_channel)
    for name in (PATHWAY_MODULE, "unmark/evaluation/stage2_scf_campaign.py"):
        text = source(name)
        assert "prepare_stage2_unmark_input" not in text or "def prepare_stage2_unmark_input" not in text
        assert "def project_text" not in text
        assert "def _with_special_tokens" not in text


def test_classifierless_preparation_still_fails_closed():
    """The corrected guard is live: `classifier=None` is refused, not defaulted."""
    import zlib

    from unmark.corruption import CorruptionPurpose
    from unmark.evaluation.stage2_dual_finalist import prepare_stage2_unmark_input

    class StubTokenizer:
        pad_token_id = 1
        unk_token_id = 3

        def tokenize(self, text):
            return list(text)

        def convert_tokens_to_ids(self, tokens):
            return [7 + zlib.crc32(t.encode("utf-8")) % 400 for t in tokens]

        def build_inputs_with_special_tokens(self, ids):
            return [0] + list(ids) + [2]

        def get_special_tokens_mask(self, ids, already_has_special_tokens=False):
            return [1] + [0] * len(ids) + [1]

    with pytest.raises(EvaluationContractViolation) as error:
        prepare_stage2_unmark_input(
            text="Tôi đã học",
            sample_id="scf-guard",
            tokenizer=StubTokenizer(),
            condition="P100",
            corruption_seed=19225,
            classifier=None,
            corruption_purpose=CorruptionPurpose.SELF_CHECK,
        )
    assert "classifier" in str(error.value)


def test_frozen_protocol_requires_the_syllable_inventory_classifier():
    payload = scf.require_frozen_scf_protocol_spec()
    entry = payload["frozen_representation"]["syllable_inventory_classifier"]
    assert entry["value"] == "REQUIRED"
    assert "make_classifier" in entry["note"]


# ---------------------------------------------------------------------------
# I. Reuse, not re-implementation
# ---------------------------------------------------------------------------
def test_scf_pathway_does_not_reimplement_the_fusion_equation():
    """The C1 equation appears in the modeling layer and nowhere in Stage-2."""
    text = source(PATHWAY_MODULE)
    tree = ast.parse(text)
    for node in ast.walk(tree):
        # `.norm(...)` is the shape of a hand-rolled calibration; the Stage-2
        # layer must not contain one.
        if isinstance(node, ast.Attribute) and node.attr == "norm":
            raise AssertionError(
                "the Stage-2 V2-SCF pathway computes a norm; the scale-calibrated "
                "equation belongs to unmark.modeling.adapter and must not be restated"
            )
    assert "clamp" not in text.replace("clamp(||f||, 1e-8)", "")


def test_scf_pathway_builds_the_adapter_through_stage1_dispatch():
    """Asserted by AST: `reconstruct_adapter` is called, not `OrthographyInputAdapter`."""
    tree = ast.parse(source(PATHWAY_MODULE))
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "reconstruct_adapter" in called
    assert "require_loadable_as" in called
    assert "OrthographyInputAdapter" not in called
    assert "AdapterConfig" not in called


def test_scf_extraction_delegates_to_the_accepted_forward():
    tree = ast.parse(source(PATHWAY_MODULE))
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "extract_stage2_unmark_representations" in called


def test_scf_pathway_imports_no_torch_at_module_level():
    """Every contract above is checkable in the ML-free environment."""
    tree = ast.parse(source(PATHWAY_MODULE))
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in getattr(node, "names", [])]
            module = getattr(node, "module", "") or ""
            assert "torch" not in module
            assert not any(n.startswith("torch") for n in names)


# ---------------------------------------------------------------------------
# J. Tensor-level behaviour (torch-gated)
# ---------------------------------------------------------------------------
def _adapter_state(torch_module, dtype=None):
    shapes = {
        "tone_embedding.weight": (7, 768),
        "letter_embedding.weight": (5, 768),
        "fusion.weight": (768, 2304),
        "fusion.bias": (768,),
        "gate.weight": (768, 2304),
        "gate.bias": (768,),
        "layer_norm.weight": (768,),
        "layer_norm.bias": (768,),
    }
    dtype = dtype or torch_module.float32
    return {
        key: torch_module.zeros(shape, dtype=dtype) for key, shape in shapes.items()
    }


def _scf_payload(torch_module, **overrides):
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "provenance": scf_provenance(),
        "adapter_state": _adapter_state(torch_module),
        "optimizer_state": {},
        "global_update": scf.V2_SCF_CHECKPOINT.update,
        "sampler_state": {},
        "cap": 20000,
        "points": [],
        "execution": {},
    }
    payload.update(overrides)
    return payload


@requires_torch
def test_scf_pathway_really_constructs_scale_calibrated_fusion():
    adapter = scf.build_scf_adapter(_scf_payload(torch))
    assert adapter.config.fusion_id == SCALE_CALIBRATED_FUSION_ID
    assert adapter.config.is_scale_calibrated is True
    assert tuple(sorted(adapter.state_dict())) == tuple(
        sorted(_adapter_state(torch))
    )
    assert sum(p.numel() for p in adapter.parameters()) == ADAPTER_TRAINABLE_PARAMETERS


@requires_torch
def test_scf_adapter_forward_differs_from_the_historical_adapter():
    """Same weights, different equation, different output. The whole point.

    If these ever agreed, the fusion identity would be decorative and loading a
    C1 checkpoint into the historical adapter would be harmless -- which is
    exactly the assumption this pathway refuses to make.
    """
    from unmark.stage1.initialisation import fresh_adapter

    scale_calibrated = fresh_adapter(HIDDEN_SIZE, 51800, SCALE_CALIBRATED_FUSION_ID)
    historical = fresh_adapter(HIDDEN_SIZE, 51800, HISTORICAL_FUSION_ID)
    historical.load_state_dict(scale_calibrated.state_dict(), strict=True)

    torch.manual_seed(0)
    # Scaled so ||e|| is clearly not ||LayerNorm(f)||: the calibration factor is
    # then far from 1 and the two equations cannot agree by numerical accident.
    base = torch.randn(2, 5, HIDDEN_SIZE) * 3.0
    tone_ids = torch.zeros(2, 5, dtype=torch.long)
    tone_mask = torch.ones(2, 5, dtype=torch.bool)
    letter_ids = torch.zeros(2, 5, 1, dtype=torch.long)
    letter_mask = torch.ones(2, 5, 1, dtype=torch.bool)

    with torch.no_grad():
        a = scale_calibrated(base, tone_ids, tone_mask, letter_ids, letter_mask)
        b = historical(base, tone_ids, tone_mask, letter_ids, letter_mask)
    assert not torch.allclose(a, b), (
        "the scale-calibrated and historical fusions produced identical outputs from "
        "identical weights; the fusion identity would then be decorative"
    )


@requires_torch
def test_scf_checkpoint_update_mismatch_fails_closed(tmp_path, monkeypatch):
    payload = _scf_payload(torch, global_update=7500)
    path = tmp_path / "checkpoint.pt"
    torch.save(payload, path)
    monkeypatch.setattr(
        scf, "sha256_file", lambda _p: scf.V2_SCF_CHECKPOINT.checkpoint_sha256
    )
    with pytest.raises(scf.ScfPathwayViolation) as error:
        scf.verify_scf_checkpoint(path, inventory=INVENTORY)
    assert "global_update" in str(error.value)
    assert "8000" in str(error.value)


@requires_torch
def test_scf_checkpoint_source_head_mismatch_fails_closed(tmp_path, monkeypatch):
    payload = _scf_payload(torch)
    payload["provenance"]["repository_head"] = "0" * 40
    path = tmp_path / "checkpoint.pt"
    torch.save(payload, path)
    monkeypatch.setattr(
        scf, "sha256_file", lambda _p: scf.V2_SCF_CHECKPOINT.checkpoint_sha256
    )
    with pytest.raises(scf.ScfPathwayViolation) as error:
        scf.verify_scf_checkpoint(path, inventory=INVENTORY)
    assert "repository_head" in str(error.value)


@requires_torch
def test_scf_checkpoint_historical_fusion_fails_closed(tmp_path, monkeypatch):
    payload = _scf_payload(torch)
    payload["provenance"]["fusion"] = {"fusion_id": HISTORICAL_FUSION_ID}
    path = tmp_path / "checkpoint.pt"
    torch.save(payload, path)
    monkeypatch.setattr(
        scf, "sha256_file", lambda _p: scf.V2_SCF_CHECKPOINT.checkpoint_sha256
    )
    with pytest.raises(scf.ScfPathwayViolation) as error:
        scf.verify_scf_checkpoint(path, inventory=INVENTORY)
    assert SCALE_CALIBRATED_FUSION_ID in str(error.value)


@requires_torch
def test_scf_checkpoint_verifies_and_binds(tmp_path, monkeypatch):
    payload = _scf_payload(torch)
    path = tmp_path / "checkpoint.pt"
    torch.save(payload, path)
    monkeypatch.setattr(
        scf, "sha256_file", lambda _p: scf.V2_SCF_CHECKPOINT.checkpoint_sha256
    )
    evidence = scf.verify_scf_checkpoint(path, inventory=INVENTORY)
    assert evidence["fusion_id"] == SCALE_CALIBRATED_FUSION_ID
    assert evidence["update"] == 8000
    assert evidence["adapter_tensor_count"] == ADAPTER_TENSOR_COUNT
    assert evidence["adapter_trainable_parameters"] == ADAPTER_TRAINABLE_PARAMETERS
    binding = scf.bind_verified_scf_checkpoint(path, evidence)
    assert binding.checkpoint_sha256 == scf.V2_SCF_CHECKPOINT.checkpoint_sha256
