from viunmark.provenance import load_historical_aliases, load_reproduction_manifest, load_uit_vsfc_reproduction


def test_reproduction_record():
    reproduction = load_uit_vsfc_reproduction()
    assert reproduction.head_count == 20
    assert reproduction.viunmark_calibration.additive_logit_bias == 1.25
    assert reproduction.adapted_only_calibration.additive_logit_bias == 0.75
    assert reproduction.viunmark_config().stage2_parameter_count == 6955580


def test_manifest_and_aliases_are_provenance_only():
    manifest = load_reproduction_manifest()
    aliases = load_historical_aliases()
    assert manifest["method_protocol_recovery_not_bit_exact_execution_recovery"] is True
    assert manifest["public_code_license"] == "UNRESOLVED"
    assert aliases["public_api"] is False
