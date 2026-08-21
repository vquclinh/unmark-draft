"""Pre-G1 internal split: hardened splitter and deterministic materialiser.

ML-free and network-free. Every fixture is synthetic — no corpus text, no
downloads, no real membership. The real derived csv is never read here; where a
test needs to exercise the digest gate, it builds a file and checks the refusal.

These tests are deliberately **executable** rather than source-matching. Audit
022 shipped a schema defect past a test that grepped a module for a version
string while the program wrote a different one; the lesson is applied here by
running the splitter and the materialiser and reading their outputs.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys
from collections import Counter

import pytest

from unmark.evaluation import EvaluationContractViolation
from unmark.evaluation.preg1_protocol import (
    DERIVED_TRAIN_CSV_SHA256,
    DERIVED_TRAIN_LABEL_COUNTS,
    DERIVED_TRAIN_SIZE,
    INTERNAL_SPLIT_FRACTIONS,
    LABEL_MAPPING,
    SPLIT_SEED,
    SPLIT_SEED_TAG,
    SPLITTER_STATUS,
)
from unmark.evaluation.preg1_split import (
    ID_FILE_NAMES,
    PROTOCOL_DEV,
    PROTOCOL_TRAIN,
    SPLIT_SCHEMA_VERSION,
    DerivedPool,
    expected_split_counts,
    expected_split_totals,
    load_derived_pool,
    materialize_split,
)
from unmark.evaluation.profiling import (
    SPLIT_ALLOCATION_ORDER_RULE,
    stratified_group_split,
)

REPO = pathlib.Path(__file__).resolve().parents[1]
LOCKED = dict(INTERNAL_SPLIT_FRACTIONS)


# ---------------------------------------------------------------------------
# Synthetic fixtures — never real corpus text
# ---------------------------------------------------------------------------
def synthetic_pool(counts=None):
    """A singleton-group pool with the locked derived class counts."""
    counts = counts or dict(DERIVED_TRAIN_LABEL_COUNTS)
    records, index = [], 0
    for label in sorted(counts):
        for k in range(counts[label]):
            index += 1
            records.append((f"s{index:06d}", f"synthetic-{label}-{k}", LABEL_MAPPING[label]))
    return records


def small_records(n=40):
    return [
        (f"id{i:03d}", f"synthetic text {i}", i % 3)
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# 1-4: determinism and order invariance
# ---------------------------------------------------------------------------
def test_repeated_calls_are_deterministic():
    records = small_records()
    assert stratified_group_split(records, LOCKED, SPLIT_SEED) == stratified_group_split(
        records, LOCKED, SPLIT_SEED
    )


def test_record_input_order_does_not_change_membership():
    records = small_records()
    shuffled = records[7:] + records[:7]
    assert stratified_group_split(shuffled, LOCKED, SPLIT_SEED) == stratified_group_split(
        records, LOCKED, SPLIT_SEED
    )


def test_fraction_mapping_insertion_order_does_not_change_membership():
    """S23-F1. A dict literal's keystroke order was a scientific variable."""
    records = small_records()
    forward = {"protocol-train": 0.80, "protocol-dev": 0.20}
    reversed_mapping = {"protocol-dev": 0.20, "protocol-train": 0.80}
    assert list(forward) != list(reversed_mapping)  # the mappings really do differ
    assert stratified_group_split(records, forward, SPLIT_SEED) == stratified_group_split(
        records, reversed_mapping, SPLIT_SEED
    )


def test_allocation_order_is_descending_fraction_then_name():
    """Equal fractions must still have a total order, or the rule is incomplete."""
    records = small_records()
    even = {"b-part": 0.5, "a-part": 0.5}
    flipped = {"a-part": 0.5, "b-part": 0.5}
    assert stratified_group_split(records, even, SPLIT_SEED) == stratified_group_split(
        records, flipped, SPLIT_SEED
    )
    assert "descending fraction" in SPLIT_ALLOCATION_ORDER_RULE


def test_canonical_duplicates_with_one_label_stay_atomic():
    records = small_records() + [
        ("dupA", "synthetic text 3", 0),
        ("dupB", "SYNTHETIC TEXT 3", 0),  # canon-equal after normalisation? not required
    ]
    records.append(("dupC", "synthetic text 3", 0))
    parts = stratified_group_split(records, LOCKED, SPLIT_SEED)
    train, dev = set(parts[PROTOCOL_TRAIN]), set(parts[PROTOCOL_DEV])
    group = {"id003", "dupA", "dupC"}  # same canonical text
    assert group <= train or group <= dev, "a canonical group straddled the split"


# ---------------------------------------------------------------------------
# 5-7: fail-closed
# ---------------------------------------------------------------------------
def test_conflicting_label_canonical_group_raises():
    """S23-F2. Majority voting would manufacture a gold label nobody assigned."""
    records = small_records() + [
        ("conflictA", "a contested sentence", 0),
        ("conflictB", "a contested sentence", 2),
    ]
    with pytest.raises(EvaluationContractViolation, match="conflicting-label"):
        stratified_group_split(records, LOCKED, SPLIT_SEED)


def test_conflict_error_does_not_leak_raw_text():
    secret = "a very distinctive contested sentence"
    records = small_records() + [("cA", secret, 0), ("cB", secret, 2)]
    with pytest.raises(EvaluationContractViolation) as caught:
        stratified_group_split(records, LOCKED, SPLIT_SEED)
    message = str(caught.value)
    assert secret not in message
    assert "cA" in message and "cB" in message  # ids are allowed
    assert "digest=" in message


def test_conflict_is_not_resolved_by_majority():
    """Three-to-one must fail exactly as two-to-two does."""
    records = small_records() + [
        ("m1", "majority sentence", 0),
        ("m2", "majority sentence", 0),
        ("m3", "majority sentence", 0),
        ("m4", "majority sentence", 2),
    ]
    with pytest.raises(EvaluationContractViolation, match="conflicting-label"):
        stratified_group_split(records, LOCKED, SPLIT_SEED)


def test_duplicate_sample_ids_raise():
    """S23-F3. A membership artifact cannot say which of two rows was assigned."""
    records = small_records() + [("id005", "a different sentence", 1)]
    with pytest.raises(EvaluationContractViolation, match="globally unique"):
        stratified_group_split(records, LOCKED, SPLIT_SEED)


def test_duplicate_id_error_does_not_leak_raw_text():
    secret = "another very distinctive sentence"
    records = small_records() + [("id005", secret, 1)]
    with pytest.raises(EvaluationContractViolation) as caught:
        stratified_group_split(records, LOCKED, SPLIT_SEED)
    assert secret not in str(caught.value)
    assert "id005" in str(caught.value)


def test_empty_input_raises():
    with pytest.raises(EvaluationContractViolation, match="empty record set"):
        stratified_group_split([], LOCKED, SPLIT_SEED)


# ---------------------------------------------------------------------------
# 9-10: fraction validation
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "fractions, match",
    [
        ({}, "non-empty mapping"),
        ({"": 1.0}, "non-empty strings"),
        ({"a": float("nan"), "b": 1.0}, "finite"),
        ({"a": float("inf")}, "finite"),
        ({"a": 0.0, "b": 1.0}, "strictly positive"),
        ({"a": -0.2, "b": 1.2}, "strictly positive"),
        ({"a": "0.8", "b": 0.2}, "must be a number"),
        ({"a": 0.5, "b": 0.2}, "sum to 1.0"),
        ({"a": 0.7, "b": 0.7}, "sum to 1.0"),
    ],
)
def test_invalid_fractions_raise(fractions, match):
    with pytest.raises(EvaluationContractViolation, match=match):
        stratified_group_split(small_records(), fractions, SPLIT_SEED)


# ---------------------------------------------------------------------------
# 11-15: partition properties
# ---------------------------------------------------------------------------
def test_parts_are_disjoint_complete_and_duplicate_free():
    records = small_records(90)
    parts = stratified_group_split(records, LOCKED, SPLIT_SEED)
    emitted = [i for name in parts for i in parts[name]]
    assert len(emitted) == len(set(emitted)) == len(records)
    assert set(emitted) == {sample_id for sample_id, _, _ in records}
    assert not set(parts[PROTOCOL_TRAIN]) & set(parts[PROTOCOL_DEV])


def test_output_ids_are_deterministically_ordered():
    parts = stratified_group_split(small_records(50), LOCKED, SPLIT_SEED)
    for ids in parts.values():
        assert ids == sorted(ids)


def test_changing_the_seed_changes_membership():
    records = small_records(200)
    assert stratified_group_split(records, LOCKED, SPLIT_SEED) != stratified_group_split(
        records, LOCKED, SPLIT_SEED + 1
    )


def test_no_global_rng_dependency():
    """Seeding `random` differently must not move a single id."""
    import random

    records = small_records(120)
    random.seed(1)
    first = stratified_group_split(records, LOCKED, SPLIT_SEED)
    random.seed(999)
    assert stratified_group_split(records, LOCKED, SPLIT_SEED) == first


def test_splitter_module_imports_no_rng_or_ml():
    import ast

    tree = ast.parse((REPO / "unmark/evaluation/profiling.py").read_text(encoding="utf-8"))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
    assert not modules & {"random", "numpy", "torch", "sklearn", "transformers"}


# ---------------------------------------------------------------------------
# 16: the precommitted aggregate counts
# ---------------------------------------------------------------------------
def test_expected_counts_are_computed_from_locked_aggregates():
    assert expected_split_counts() == {
        PROTOCOL_TRAIN: {"negative": 4259, "neutral": 366, "positive": 4514},
        PROTOCOL_DEV: {"negative": 1065, "neutral": 92, "positive": 1128},
    }
    assert expected_split_totals() == {PROTOCOL_TRAIN: 9139, PROTOCOL_DEV: 2285}
    assert sum(expected_split_totals().values()) == DERIVED_TRAIN_SIZE


def test_synthetic_locked_pool_reproduces_the_exact_expected_counts():
    """The precommitment is executable: a synthetic pool with the real class
    totals must land on exactly 9139 / 2285 with the locked seed."""
    records = synthetic_pool()
    assert len(records) == DERIVED_TRAIN_SIZE
    parts = stratified_group_split(records, LOCKED, SPLIT_SEED)
    inverse = {index: name for name, index in LABEL_MAPPING.items()}
    label_of = {sample_id: inverse[label] for sample_id, _, label in records}

    assert len(parts[PROTOCOL_TRAIN]) == 9139
    assert len(parts[PROTOCOL_DEV]) == 2285
    assert dict(Counter(label_of[i] for i in parts[PROTOCOL_TRAIN])) == {
        "negative": 4259, "neutral": 366, "positive": 4514
    }
    assert dict(Counter(label_of[i] for i in parts[PROTOCOL_DEV])) == {
        "negative": 1065, "neutral": 92, "positive": 1128
    }


# ---------------------------------------------------------------------------
# Materialiser: input gate
# ---------------------------------------------------------------------------
def write_csv(path, records):
    inverse = {index: name for name, index in LABEL_MAPPING.items()}
    lines = ["id,text,label"]
    for sample_id, text, label in records:
        name = inverse.get(label, label)
        lines.append(f"{sample_id},{text},{LABEL_MAPPING[name]}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def load_relaxed(path, records):
    """Load with the expectations relaxed to the synthetic fixture's own shape."""
    return load_derived_pool(
        path, "text", "label", "id",
        expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        expected_rows=len(records),
        expected_label_counts=dict(
            Counter(
                {index: name for name, index in LABEL_MAPPING.items()}[label]
                for _, _, label in records
            )
        ),
    )


def test_materializer_refuses_a_wrong_input_digest(tmp_path):
    """The digest gate is what ties the split to the approved corpus."""
    path = write_csv(tmp_path / "derived.csv", small_records(9))
    with pytest.raises(EvaluationContractViolation, match="digest mismatch"):
        load_derived_pool(path, "text", "label", "id")  # expects the locked SHA


def test_materializer_refuses_a_missing_file(tmp_path):
    with pytest.raises(EvaluationContractViolation, match="not found"):
        load_derived_pool(tmp_path / "absent.csv", "text", "label", "id")


def test_materializer_refuses_a_wrong_row_count(tmp_path):
    records = small_records(9)
    path = write_csv(tmp_path / "derived.csv", records)
    with pytest.raises(EvaluationContractViolation, match="must have 10 rows"):
        load_derived_pool(
            path, "text", "label", "id",
            expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            expected_rows=10,
            expected_label_counts={"negative": 3, "neutral": 3, "positive": 3},
        )


def test_materializer_refuses_wrong_label_counts(tmp_path):
    records = small_records(9)
    path = write_csv(tmp_path / "derived.csv", records)
    with pytest.raises(EvaluationContractViolation, match="label counts must be"):
        load_derived_pool(
            path, "text", "label", "id",
            expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            expected_rows=9,
            expected_label_counts={"negative": 9},
        )


def test_materializer_refuses_duplicate_ids(tmp_path):
    records = small_records(9) + [("id000", "a distinct sentence", 1)]
    path = write_csv(tmp_path / "derived.csv", records)
    with pytest.raises(EvaluationContractViolation, match="duplicate sample ids"):
        load_relaxed(path, records)


def test_materializer_refuses_empty_text(tmp_path):
    path = tmp_path / "derived.csv"
    path.write_text("id,text,label\na,x,0\nb, ,1\n", encoding="utf-8")
    with pytest.raises(EvaluationContractViolation, match="empty text"):
        load_derived_pool(
            path, "text", "label", "id",
            expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            expected_rows=2, expected_label_counts={"negative": 1, "neutral": 1},
        )


def test_materializer_refuses_a_missing_column(tmp_path):
    path = tmp_path / "derived.csv"
    path.write_text("id,body,label\na,x,0\n", encoding="utf-8")
    with pytest.raises(EvaluationContractViolation, match="missing"):
        load_derived_pool(
            path, "text", "label", "id",
            expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            expected_rows=1, expected_label_counts={"negative": 1},
        )


def test_materializer_refuses_conflicting_canonical_groups(tmp_path):
    records = small_records(9) + [
        ("cA", "a contested sentence", 0),
        ("cB", "a contested sentence", 2),
    ]
    path = write_csv(tmp_path / "derived.csv", records)
    pool = load_relaxed(path, records)
    with pytest.raises(EvaluationContractViolation, match="conflicting-label"):
        materialize_split(pool, tmp_path / "out")


# ---------------------------------------------------------------------------
# Materialiser: artifacts
# ---------------------------------------------------------------------------
def materialize_synthetic(tmp_path, name="out", counts=None):
    counts = counts or {"negative": 40, "neutral": 10, "positive": 50}
    records = synthetic_pool(counts)
    path = write_csv(tmp_path / f"{name}.csv", records)
    pool = load_relaxed(path, records)
    manifest = materialize_split(pool, tmp_path / name, repository_head="deadbeef")
    return tmp_path / name, manifest, pool


def test_materializer_writes_the_expected_artifacts(tmp_path):
    out, manifest, _ = materialize_synthetic(tmp_path)
    for filename in (*ID_FILE_NAMES.values(), "split-manifest.json", "report.md"):
        assert (out / filename).is_file(), filename
    assert manifest["schema_version"] == SPLIT_SCHEMA_VERSION == "preg1-split-v1"


def test_materializer_refuses_to_overwrite(tmp_path):
    out, _, pool = materialize_synthetic(tmp_path)
    with pytest.raises(EvaluationContractViolation, match="refusing to overwrite"):
        materialize_split(pool, out)


def test_a_failed_run_leaves_no_authoritative_looking_directory(tmp_path):
    records = synthetic_pool({"negative": 8, "neutral": 2, "positive": 10}) + [
        ("cA", "a contested sentence", 0),
        ("cB", "a contested sentence", 2),
    ]
    path = write_csv(tmp_path / "bad.csv", records)
    pool = load_relaxed(path, records)
    with pytest.raises(EvaluationContractViolation):
        materialize_split(pool, tmp_path / "bad-out")
    assert not (tmp_path / "bad-out").exists()
    assert not (tmp_path / "bad-out.partial").exists()


def test_repeated_materialization_is_byte_identical(tmp_path):
    """Determinism at the file level, not just the data structure level."""
    first, _, _ = materialize_synthetic(tmp_path, "run1")
    second, _, _ = materialize_synthetic(tmp_path, "run2")
    for filename in (*ID_FILE_NAMES.values(), "split-manifest.json", "report.md"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes(), filename


def test_manifest_carries_no_runtime_varying_fields(tmp_path):
    """Checks KEYS and VALUES structurally.

    A substring scan would fail on the manifest's own documentation, which names
    the very fields it promises to exclude — the prose-matching trap.
    """
    out, _, _ = materialize_synthetic(tmp_path)
    manifest = json.loads((out / "split-manifest.json").read_text(encoding="utf-8"))

    banned_keys = {"timestamp", "created_at", "generated_at", "run_id", "uuid",
                   "hostname", "user", "cwd", "path", "elapsed", "duration"}
    seen_keys, values = set(), []

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                seen_keys.add(key)
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)
        else:
            values.append(node)

    walk(manifest)
    assert not seen_keys & banned_keys, seen_keys & banned_keys
    # No value may carry an absolute machine path.
    assert not any(isinstance(v, str) and str(tmp_path) in v for v in values)


def test_runtime_evidence_is_kept_out_of_the_membership_artifact(tmp_path):
    records = synthetic_pool({"negative": 8, "neutral": 2, "positive": 10})
    path = write_csv(tmp_path / "d.csv", records)
    pool = load_relaxed(path, records)
    out = tmp_path / "withruntime"
    materialize_split(pool, out, runtime_evidence={"python": "3.12.13"})
    assert (out / "runtime-environment.json").is_file()
    manifest = json.loads((out / "split-manifest.json").read_text(encoding="utf-8"))
    assert "3.12.13" not in json.dumps(manifest)
    assert "python" not in json.dumps(manifest)


def test_manifest_records_the_locked_protocol_values(tmp_path):
    out, manifest, _ = materialize_synthetic(tmp_path)
    assert manifest["split"]["seed"] == SPLIT_SEED == 17486
    assert manifest["split"]["seed_tag"] == SPLIT_SEED_TAG
    assert manifest["split"]["fractions"] == dict(INTERNAL_SPLIT_FRACTIONS)
    assert manifest["boundaries"] == {
        "raw_text_persisted": False,
        "raw_text_rule": manifest["boundaries"]["raw_text_rule"],
        "official_validation_used": False,
        "official_test_used": False,
        "downstream_score_used": False,
        "head_trained": False,
    }
    assert manifest["input"]["duplicate_id_count"] == 0
    assert manifest["input"]["canonical_conflicting_label_groups"] == 0
    assert manifest["result"]["canonical_cross_part_leakage"] == 0
    assert manifest["repository_head"] == "deadbeef"


def test_id_file_digests_match_the_files(tmp_path):
    out, manifest, _ = materialize_synthetic(tmp_path)
    for filename, digest in manifest["result"]["id_file_sha256"].items():
        actual = hashlib.sha256((out / filename).read_bytes()).hexdigest()
        assert actual == digest, filename


def test_artifacts_contain_no_raw_text(tmp_path):
    """The synthetic texts are distinctive; none may appear in any artifact."""
    out, _, pool = materialize_synthetic(tmp_path)
    texts = {text for _, text, _ in pool.records}
    for artifact in out.iterdir():
        blob = artifact.read_text(encoding="utf-8")
        for text in texts:
            assert text not in blob, f"{artifact.name} leaked corpus text"


def test_id_files_are_sorted_and_complete(tmp_path):
    out, manifest, pool = materialize_synthetic(tmp_path)
    emitted = []
    for filename in ID_FILE_NAMES.values():
        ids = (out / filename).read_text(encoding="utf-8").split()
        assert ids == sorted(ids)
        emitted.extend(ids)
    assert sorted(emitted) == sorted(s for s, _, _ in pool.records)


def test_assignment_digest_is_stable_and_membership_sensitive(tmp_path):
    _, first, _ = materialize_synthetic(tmp_path, "a")
    _, second, _ = materialize_synthetic(tmp_path, "b")
    assert first["result"]["assignment_digest"] == second["result"]["assignment_digest"]
    _, other, _ = materialize_synthetic(
        tmp_path, "c", counts={"negative": 41, "neutral": 10, "positive": 50}
    )
    assert other["result"]["assignment_digest"] != first["result"]["assignment_digest"]


# ---------------------------------------------------------------------------
# Boundaries
# ---------------------------------------------------------------------------
def test_official_validation_and_test_cannot_enter_the_interface():
    """Structural: the CLI exposes no flag by which they could be supplied."""
    import ast

    source = (REPO / "scripts/materialize_preg1_split.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    flags = {
        node.args[0].value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    }
    assert "--derived-train" in flags
    assert not {"--validation", "--test", "--dev", "--official-test"} & flags
    # ... and no override of a precommitted scientific constant.
    assert not {"--seed", "--fractions", "--split-seed"} & flags


def test_materializer_imports_locked_values_rather_than_restating_them():
    import ast

    source = (REPO / "scripts/materialize_preg1_split.py").read_text(encoding="utf-8")
    assert "17486" not in source
    assert "0.80" not in source and "0.20" not in source
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert {"SPLIT_SEED", "SPLIT_SEED_TAG"} <= imported


def test_split_module_restates_no_locked_literal():
    """AST, not substring: the module may DOCUMENT the numbers, never compute with them."""
    import ast

    tree = ast.parse((REPO / "unmark/evaluation/preg1_split.py").read_text(encoding="utf-8"))
    numeric = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
    }
    forbidden = {17486, 11424, 5324, 458, 5642, 4259, 366, 4514, 9139, 1065, 1128, 2285}
    assert not numeric & forbidden, f"restated locked values: {sorted(numeric & forbidden)}"


def test_derived_input_digest_constant_is_the_locked_one():
    assert DERIVED_TRAIN_CSV_SHA256 == (
        "a20c0f7760f32dc48263a79d73ddf5363526c17e9de2afc32d8346b23444d301"
    )


def test_splitter_status_defers_run_state_to_the_audit():
    assert "Audit 023" in SPLITTER_STATUS
    assert "FAIL-CLOSED" in SPLITTER_STATUS


def test_cli_refuses_a_wrong_digest_with_a_nonzero_exit(tmp_path):
    """End to end, through the actual entry point."""
    records = small_records(9)
    path = write_csv(tmp_path / "derived.csv", records)
    result = subprocess.run(
        [
            sys.executable, str(REPO / "scripts/materialize_preg1_split.py"),
            "--derived-train", str(path),
            "--text-column", "text", "--label-column", "label", "--id-column", "id",
            "--output-dir", str(tmp_path / "out"),
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "SPLIT REFUSED" in result.stderr
    assert not (tmp_path / "out").exists()


def test_locked_digest_gate_is_live_not_dead_code():
    """A pool claiming the locked digest MUST hit the precommitted aggregates.

    Proven by constructing a pool that carries the locked digest but the wrong
    contents: the gate must fire. Without this, the strongest guarantee in the
    module could be unreachable and every other test would still pass.
    """
    from unmark.evaluation.preg1_split import validate_assignment

    records = tuple(synthetic_pool({"negative": 8, "neutral": 2, "positive": 10}))
    impostor = DerivedPool(records=records, source_sha256=DERIVED_TRAIN_CSV_SHA256)
    parts = stratified_group_split(list(records), LOCKED, SPLIT_SEED)
    with pytest.raises(EvaluationContractViolation, match="precommitted split"):
        validate_assignment(impostor, parts)


def test_rule_based_validation_passes_for_a_non_locked_pool():
    """The same assignment is fine when the pool does not claim the locked digest."""
    from unmark.evaluation.preg1_split import validate_assignment

    records = tuple(synthetic_pool({"negative": 8, "neutral": 2, "positive": 10}))
    pool = DerivedPool(records=records, source_sha256="not-the-locked-digest")
    parts = stratified_group_split(list(records), LOCKED, SPLIT_SEED)
    result = validate_assignment(pool, parts)
    assert sum(result["totals"].values()) == len(records)


def test_a_locked_sized_pool_reproduces_the_precommitment_through_validation():
    """End to end through validate_assignment on a locked-shaped synthetic pool."""
    from unmark.evaluation.preg1_split import validate_assignment

    records = tuple(synthetic_pool())
    pool = DerivedPool(records=records, source_sha256=DERIVED_TRAIN_CSV_SHA256)
    parts = stratified_group_split(list(records), LOCKED, SPLIT_SEED)
    result = validate_assignment(pool, parts)
    assert result["totals"] == {PROTOCOL_TRAIN: 9139, PROTOCOL_DEV: 2285}
    assert result["label_counts"][PROTOCOL_TRAIN] == {
        "negative": 4259, "neutral": 366, "positive": 4514
    }
    assert result["label_counts"][PROTOCOL_DEV] == {
        "negative": 1065, "neutral": 92, "positive": 1128
    }
