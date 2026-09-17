"""Public naming: research-stage identifiers stay in provenance, out of the API.

The ViUnMark package and its final-system spec must read in the terms of the
method. Historical research labels are allowed in exactly one committed place --
`docs/spec/viunmark-historical-aliases-v1.json` -- and in tests like this one
that check the mapping. Torch-free.
"""

from __future__ import annotations

import ast
import json
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import unmark.viunmark as viunmark  # noqa: E402
import unmark.viunmark.diagnostics as diagnostics  # noqa: E402
from unmark.viunmark.provenance import load_historical_aliases  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[1]
PACKAGE = REPO / "unmark" / "viunmark"
FINAL_SPEC = REPO / "docs" / "spec" / "viunmark-final-system-v1.json"
ALIAS_SPEC = REPO / "docs" / "spec" / "viunmark-historical-aliases-v1.json"

HISTORICAL_IDENTIFIER_PATTERNS: tuple[str, ...] = (
    r"\bD[1-4]\b",
    r"\bR[1-6](?:-[UW])?(?:-MLP)?\b",
    r"\bSYS ?[12]\b",
    r"\bSYS2-[12]\b",
    r"\bUNMARK[-_][AB]\b",
    r"\bV2[-_]SCF\b",
    r"\bV2_[UW]\b",
    r"\bVANILLA\b",
    r"\bOPT\d",
    r"\bCOMP-D[12]\b",
    r"\bCAT-U\b",
    r"\bAUGMENTED_6COND\b",
)

EXPECTED_ALIASES = {
    "UNMARK-A": "ViUnMark-Gate",
    "V2-SCF": "ViUnMark-Scale",
    "R4-W-MLP": "Gate Robust Readout",
    "R6-U-MLP": "Scale Unweighted Readout",
    "R6-W-MLP": "Scale Weighted Readout",
    "VANILLA CAT-U-MLP": "PhoBERT Readout",
    "SYS1": "Adapted-Only Fusion",
    "SYS2-1": "PhoBERT Robust Readout Selection",
    "SYS2-2": "ViUnMark",
    "D1": "Scale Pathway Preflight",
    "D2": "Checkpoint-Free Pooling Bridge",
    "D3": "Matched-Head Pooling Comparison",
    "D4": "Native-Adapted Decision Geometry",
    "COMP-D1": "Matched-Recipe Complementarity Analysis",
    "COMP-D2": "Robustness Gain Factorization",
    "V2_U": "Scale Unweighted branch ensemble",
    "V2_W": "Scale Weighted branch ensemble",
    "seed_ensemble": "within-branch head-logit mean",
    "OPT1": "Stage-II Readout and Augmentation Screen",
    "OPT2": "Stage-II Class-Balance Loss Optimization",
    "OPT3": "adapted-pathway robust readout optimization",
    "R4": "screen candidate 4: MASKED_MEAN, AUGMENTED_6COND",
    "R6": "screen candidate 6: [FIRST_TOKEN;MASKED_MEAN], AUGMENTED_6COND",
    "F": "FOCAL_GAMMA_2",
}


def package_sources() -> list[pathlib.Path]:
    return sorted(PACKAGE.rglob("*.py"))


def hits(text: str) -> list[str]:
    return [
        match.group(0)
        for pattern in HISTORICAL_IDENTIFIER_PATTERNS
        for match in re.finditer(pattern, text)
    ]


def test_the_package_exists_and_has_sources():
    names = {path.name for path in package_sources()}
    assert {"config.py", "fusion.py", "system.py", "readout.py", "heads.py",
            "losses.py", "provenance.py"} <= names


def test_no_historical_identifier_in_any_package_source():
    offenders = {
        str(path.relative_to(REPO)): found
        for path in package_sources()
        if (found := hits(path.read_text(encoding="utf-8")))
    }
    assert offenders == {}, offenders


def test_no_historical_identifier_in_the_final_system_spec():
    assert hits(FINAL_SPEC.read_text(encoding="utf-8")) == []


def test_the_pattern_set_actually_detects_historical_identifiers():
    """Guards the guard: each pattern must fire on the label it exists for."""
    for label in ("D1", "D4", "R4-W-MLP", "R6-U", "SYS1", "SYS2-2", "UNMARK-A", "V2-SCF",
                  "V2_SCF", "V2_U", "VANILLA", "OPT3", "COMP-D1", "CAT-U", "AUGMENTED_6COND"):
        assert hits(f"x {label} y"), label
    for clean in ("VIUNMARK_ADAPTED_WEIGHT", "ViUnMark-Gate", "gate_robust_readout", "d1ce"):
        assert hits(clean) == [], clean


def test_no_module_is_named_after_a_research_stage():
    for path in package_sources():
        assert not re.fullmatch(r"(d[1-4]|r[1-6].*|sys\d.*|opt\d.*|comp_?d\d)\.py", path.name), path


def test_public_exports_carry_no_historical_identifier():
    for module in (viunmark, diagnostics):
        assert hits(" ".join(module.__all__)) == []


def test_defined_names_carry_no_historical_identifier():
    for path in package_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            name = getattr(node, "name", None) or getattr(node, "id", None)
            if isinstance(name, str):
                assert hits(name.replace("_", "-")) == [], (path, name)


def test_alias_spec_maps_every_historical_name_to_its_descriptive_name():
    payload = load_historical_aliases()
    mapping = {entry["historical_id"]: entry["descriptive_name"] for entry in payload["aliases"]}
    for historical, descriptive in EXPECTED_ALIASES.items():
        assert mapping.get(historical) == descriptive, historical


def test_alias_spec_is_declared_not_public_api():
    payload = json.loads(ALIAS_SPEC.read_text(encoding="utf-8"))
    assert payload["public_api"] is False
    development_stage = next(a for a in payload["aliases"] if a["historical_id"] == "SYS2-1")
    assert development_stage["public_api"] is False
    assert development_stage["python_name"] is None


def test_no_public_name_exposes_the_development_stage():
    exported = " ".join(viunmark.__all__ + diagnostics.__all__)
    assert "Selection" not in exported
    assert "SYS" not in exported


def test_final_spec_points_at_the_alias_spec_as_non_public():
    payload = json.loads(FINAL_SPEC.read_text(encoding="utf-8"))
    assert payload["historical_aliases"] == {
        "spec": "docs/spec/viunmark-historical-aliases-v1.json",
        "public_api": False,
    }
