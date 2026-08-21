"""Pre-G1 internal split materialisation — deterministic, ML-free, fail-closed.

This turns the **derived** pre-G1 train pool into a protocol-train /
protocol-dev membership. It is the last step before a head could be trained, and
it is deliberately the most suspicious code in the pre-G1 path: a split that is
subtly wrong produces results that look fine.

Three rules shape everything here.

**Nothing is restated.** The fractions, the seed, the seed tag, the expected row
count, the expected class counts and the input digest are all imported from
`preg1_protocol`. A literal `0.8` typed here could drift from the locked value
without any test noticing.

**Every check runs before assignment, and again after.** The preflight refuses
an input that is not byte-identical to the approved derived corpus. The
postflight refuses to write an artifact whose membership does not satisfy every
invariant. A failed run must not leave a directory that looks authoritative.

**Membership artifacts are byte-deterministic.** No timestamps, no run uuids, no
absolute paths in the manifest -- the same input, code and seed must produce the
same bytes, or "reproducible" means nothing. Runtime evidence lives in a
separate file that is explicitly *not* part of the scientific artifact.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from unmark.evaluation.contracts import EvaluationContractViolation
from unmark.evaluation.preg1_protocol import (
    CONFLICTING_GROUP_POLICY,
    DERIVED_TRAIN_CSV_SHA256,
    DERIVED_TRAIN_LABEL_COUNTS,
    DERIVED_TRAIN_SIZE,
    INTERNAL_SPLIT_FRACTIONS,
    LABEL_MAPPING,
    PRIMARY_DATASET,
    PRIMARY_DATASET_VERSION,
    PRIMARY_TASK,
    SPLIT_SEED,
    SPLIT_SEED_TAG,
)
from unmark.evaluation.profiling import (
    SPLIT_ALLOCATION_ORDER_RULE,
    SPLIT_GROUPING_RULE,
    SPLIT_STRATIFICATION_RULE,
    analyse_duplicates,
    file_sha256,
    profile_split,
    stratified_group_split,
)

SPLIT_SCHEMA_VERSION = "preg1-split-v1"
"""The one authoritative split-artifact schema constant.

Audit 022 shipped a defect where this kind of version lived in two places and
they drifted. It is defined here and imported everywhere else -- never restated.
"""

PROTOCOL_TRAIN = "protocol-train"
PROTOCOL_DEV = "protocol-dev"

ID_FILE_NAMES: dict[str, str] = {
    PROTOCOL_TRAIN: "protocol-train.ids.txt",
    PROTOCOL_DEV: "protocol-dev.ids.txt",
}

DETERMINISM_RULE = (
    "The membership artifacts are byte-deterministic for a given input digest, "
    "code version and seed. No timestamp, run uuid, hostname or absolute path "
    "enters split-manifest.json or the id files; runtime evidence is written "
    "separately to runtime-environment.json, which is NOT part of the "
    "scientific artifact."
)

RAW_TEXT_RULE = (
    "Sample ids, digests and counts only. No corpus text is written to any "
    "artifact, and no exception raised by this module embeds corpus text -- "
    "UIT-VSFC is redistributed by nobody here, and an error message is a file "
    "like any other."
)


# ---------------------------------------------------------------------------
# Expected split, derived from committed aggregates alone
# ---------------------------------------------------------------------------
def expected_split_counts(
    label_counts: dict[str, int] | None = None,
) -> dict[str, dict[str, int]]:
    """Per-part class counts implied by the class totals and the allocation rule.

    Computed, not typed in. Because the derived pool has **zero** canonical
    duplicate groups (Audit 022), every canonical group is a singleton, so the
    per-class allocation is fully determined by the class totals and the
    allocation rule -- and can therefore be stated **before** any membership is
    observed. That is what makes the locked numbers a precommitment rather than
    a description.

    Defaults to the locked derived aggregates. The parameter exists so the same
    rule can be checked against any pool; the locked values are enforced
    separately, by digest, in `validate_assignment`.
    """
    label_counts = label_counts or dict(DERIVED_TRAIN_LABEL_COUNTS)
    names = sorted(
        INTERNAL_SPLIT_FRACTIONS,
        key=lambda name: (-INTERNAL_SPLIT_FRACTIONS[name], name),
    )
    expected: dict[str, dict[str, int]] = {name: {} for name in names}
    for label in sorted(label_counts):
        total = label_counts[label]
        start = 0
        for index, name in enumerate(names):
            take = (
                total - start
                if index == len(names) - 1
                else round(total * INTERNAL_SPLIT_FRACTIONS[name])
            )
            expected[name][label] = take
            start += take
    return expected


def expected_split_totals(label_counts: dict[str, int] | None = None) -> dict[str, int]:
    return {
        name: sum(counts.values())
        for name, counts in expected_split_counts(label_counts).items()
    }


LOCKED_POOL_EXPECTATION_RULE = (
    "When the input digest is the locked derived-train digest, the split MUST "
    "land on the precommitted aggregates -- protocol-train 9139 "
    "(4259/366/4514) and protocol-dev 2285 (1065/92/1128). Those numbers were "
    "derived from committed class totals and the allocation rule BEFORE any "
    "membership was observed, so failing them means the mechanism changed, not "
    "that the data surprised us."
)


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DerivedPool:
    """The verified derived train pool. Text is held only to group and digest."""

    records: tuple[tuple[str, str, Any], ...]
    source_sha256: str

    @property
    def label_counts(self) -> dict[str, int]:
        return dict(Counter(_label_name(label) for _, _, label in self.records))


def _label_name(label: Any) -> str:
    """Accept either the encoded id or the label name; return the name.

    The derived csv carries whatever the profiler wrote. Both encodings are
    accepted, and anything else fails rather than being coerced.
    """
    text = str(label).strip()
    if text in LABEL_MAPPING:
        return text
    inverse = {str(index): name for name, index in LABEL_MAPPING.items()}
    if text in inverse:
        return inverse[text]
    raise EvaluationContractViolation(
        f"unknown label {text!r}; expected one of "
        f"{sorted(LABEL_MAPPING)} or {sorted(inverse)}"
    )


def load_derived_pool(
    path: str | Path,
    text_column: str,
    label_column: str,
    id_column: str,
    *,
    expected_sha256: str = DERIVED_TRAIN_CSV_SHA256,
    expected_rows: int = DERIVED_TRAIN_SIZE,
    expected_label_counts: dict[str, int] | None = None,
) -> DerivedPool:
    """Read and **verify** the derived train csv. Refuses anything unexpected.

    The digest check is the load-bearing one: it is what ties this split to the
    exact corpus the researcher approved in Audit 022, exclusion already applied.
    A pool that merely has the right number of rows is not the same pool.
    """
    expected_label_counts = expected_label_counts or dict(DERIVED_TRAIN_LABEL_COUNTS)
    source = Path(path)
    if not source.is_file():
        raise EvaluationContractViolation(f"derived train csv not found: {source}")

    actual_sha = file_sha256(str(source))
    if actual_sha != expected_sha256:
        raise EvaluationContractViolation(
            "derived train csv digest mismatch -- this is not the approved pool. "
            f"expected {expected_sha256}, got {actual_sha}"
        )

    records: list[tuple[str, str, Any]] = []
    with source.open(encoding="utf-8", newline="") as handle:
        delimiter = "\t" if source.suffix.lower() in {".tsv", ".tab"} else ","
        reader = csv.DictReader(handle, delimiter=delimiter)
        for column in (text_column, label_column, id_column):
            if reader.fieldnames is None or column not in reader.fieldnames:
                raise EvaluationContractViolation(
                    f"column {column!r} missing; found {reader.fieldnames}"
                )
        for index, row in enumerate(reader):
            sample_id = (row.get(id_column) or "").strip()
            if not sample_id:
                raise EvaluationContractViolation(f"row {index} has an empty sample id")
            text = row.get(text_column) or ""
            if not text.strip():
                raise EvaluationContractViolation(f"sample {sample_id} has empty text")
            records.append((sample_id, text, row.get(label_column)))

    if len(records) != expected_rows:
        raise EvaluationContractViolation(
            f"derived pool must have {expected_rows} rows, got {len(records)}"
        )

    ids = Counter(sample_id for sample_id, _, _ in records)
    repeated = sorted(sample_id for sample_id, count in ids.items() if count > 1)
    if repeated:
        raise EvaluationContractViolation(
            f"derived pool has {len(repeated)} duplicate sample ids: {repeated[:10]}"
        )

    pool = DerivedPool(records=tuple(records), source_sha256=actual_sha)
    if pool.label_counts != expected_label_counts:
        raise EvaluationContractViolation(
            f"derived label counts must be {expected_label_counts}, got {pool.label_counts}"
        )
    return pool


def check_no_conflicting_groups(pool: DerivedPool) -> dict[str, int]:
    """Canonical duplicate / conflict analysis, **before** any assignment.

    Audit 022 established these are zero on the approved pool. Re-checking here
    is not redundant: it is what makes the guarantee a property of this run
    rather than a property of a document.
    """
    profile, index = profile_split("derived-train", pool.records, None)
    report = analyse_duplicates({"derived-train": index})
    conflicting = len(report.conflicting_label_groups)
    if conflicting:
        raise EvaluationContractViolation(
            f"{conflicting} conflicting-label canonical group(s) in the derived pool; "
            "the exclusion contract requires zero before splitting"
        )
    return {
        "canonical_duplicate_groups": profile.canonical_duplicate_texts,
        "conflicting_label_groups": conflicting,
        "cross_split_groups": len(report.cross_split_groups),
    }


# ---------------------------------------------------------------------------
# Materialisation
# ---------------------------------------------------------------------------
def _assignment_digest(parts: dict[str, list[str]]) -> str:
    """One digest over the whole membership. Order-independent by construction."""
    payload = "\n".join(
        f"{name}\t{sample_id}"
        for name in sorted(parts)
        for sample_id in sorted(parts[name])
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _id_file_body(ids: Sequence[str]) -> str:
    return "".join(f"{sample_id}\n" for sample_id in sorted(ids))


def validate_assignment(
    pool: DerivedPool,
    parts: dict[str, list[str]],
) -> dict[str, Any]:
    """Every post-split invariant. Raises rather than returning a warning.

    A split is either usable or it is not; there is no partial credit, and a
    PASS-looking artifact from a failed run is worse than no artifact.
    """
    if set(parts) != set(INTERNAL_SPLIT_FRACTIONS):
        raise EvaluationContractViolation(
            f"parts must be exactly {sorted(INTERNAL_SPLIT_FRACTIONS)}, got {sorted(parts)}"
        )

    all_input = [sample_id for sample_id, _, _ in pool.records]
    emitted = [sample_id for name in sorted(parts) for sample_id in parts[name]]

    repeated = sorted(s for s, c in Counter(emitted).items() if c > 1)
    if repeated:
        raise EvaluationContractViolation(f"duplicate emitted ids: {repeated[:10]}")

    train_set, dev_set = set(parts[PROTOCOL_TRAIN]), set(parts[PROTOCOL_DEV])
    overlap = sorted(train_set & dev_set)
    if overlap:
        raise EvaluationContractViolation(f"parts are not disjoint: {overlap[:10]}")
    if set(emitted) != set(all_input):
        missing = sorted(set(all_input) - set(emitted))
        extra = sorted(set(emitted) - set(all_input))
        raise EvaluationContractViolation(
            f"parts must cover every input id exactly once; missing={missing[:5]} extra={extra[:5]}"
        )
    if len(emitted) != len(all_input):
        raise EvaluationContractViolation(
            f"emitted {len(emitted)} ids for {len(all_input)} inputs"
        )

    # Canonical groups must not cross parts.
    from unmark.evaluation.profiling import text_digest
    from unmark.orthography import canon

    part_of = {sample_id: name for name in parts for sample_id in parts[name]}
    group_parts: dict[str, set[str]] = {}
    for sample_id, text, _ in pool.records:
        group_parts.setdefault(text_digest(canon(text)), set()).add(part_of[sample_id])
    straddling = sorted(d for d, names in group_parts.items() if len(names) > 1)
    if straddling:
        raise EvaluationContractViolation(
            f"{len(straddling)} canonical group(s) straddle parts: {straddling[:5]}"
        )

    label_of = {sample_id: _label_name(label) for sample_id, _, label in pool.records}
    # Every label present in the pool gets an explicit entry, including zero.
    # `Counter` omits absent keys, which would make an empty class look like a
    # count mismatch rather than the zero it is.
    pool_labels = sorted(pool.label_counts)
    observed = {
        name: {
            label: sum(1 for i in parts[name] if label_of[i] == label)
            for label in pool_labels
        }
        for name in sorted(parts)
    }
    totals = {name: len(parts[name]) for name in sorted(parts)}

    # The allocation rule must hold for whatever pool this is.
    expected = expected_split_counts(pool.label_counts)
    for name in sorted(expected):
        if observed[name] != expected[name]:
            raise EvaluationContractViolation(
                f"{name} class counts must be {expected[name]}, got {observed[name]}"
            )
    if totals != expected_split_totals(pool.label_counts):
        raise EvaluationContractViolation(
            f"part totals must be {expected_split_totals(pool.label_counts)}, got {totals}"
        )

    # ... and on the LOCKED corpus it must additionally hit the precommitted
    # aggregates. Keyed on the digest, so a synthetic fixture cannot silently
    # satisfy a check that only the approved pool is supposed to satisfy.
    if pool.source_sha256 == DERIVED_TRAIN_CSV_SHA256:
        locked_counts = expected_split_counts()
        locked_totals = expected_split_totals()
        if observed != locked_counts or totals != locked_totals:
            raise EvaluationContractViolation(
                "locked derived pool must reproduce the precommitted split: "
                f"expected totals {locked_totals} counts {locked_counts}, "
                f"got totals {totals} counts {observed}. "
                + LOCKED_POOL_EXPECTATION_RULE
            )
    return {"totals": totals, "label_counts": observed}


def build_manifest(
    pool: DerivedPool,
    parts: dict[str, list[str]],
    integrity: dict[str, int],
    validation: dict[str, Any],
    id_file_digests: dict[str, str],
    repository_head: str | None,
) -> dict[str, Any]:
    """The deterministic scientific membership manifest.

    Contains no timestamp, uuid, hostname or absolute path -- see
    `DETERMINISM_RULE`. `repository_head` is provenance the caller supplies; it
    is part of the scientific identity of the run, not runtime noise.
    """
    return {
        "schema_version": SPLIT_SCHEMA_VERSION,
        "dataset": PRIMARY_DATASET,
        "dataset_version": PRIMARY_DATASET_VERSION,
        "task": PRIMARY_TASK,
        "repository_head": repository_head,
        "input": {
            "derived_train_sha256": pool.source_sha256,
            "derived_train_rows": len(pool.records),
            "derived_train_label_counts": pool.label_counts,
            "exclusion_policy": CONFLICTING_GROUP_POLICY,
            "duplicate_id_count": 0,
            "canonical_duplicate_groups": integrity["canonical_duplicate_groups"],
            "canonical_conflicting_label_groups": integrity["conflicting_label_groups"],
        },
        "split": {
            "fractions": dict(INTERNAL_SPLIT_FRACTIONS),
            "seed_tag": SPLIT_SEED_TAG,
            "seed": SPLIT_SEED,
            "grouping_rule": SPLIT_GROUPING_RULE,
            "allocation_order_rule": SPLIT_ALLOCATION_ORDER_RULE,
            "stratification_rule": SPLIT_STRATIFICATION_RULE,
            "determinism_rule": DETERMINISM_RULE,
        },
        "result": {
            "totals": validation["totals"],
            "label_counts": validation["label_counts"],
            "id_file_sha256": id_file_digests,
            "assignment_digest": _assignment_digest(parts),
            "canonical_cross_part_leakage": 0,
        },
        "boundaries": {
            "raw_text_persisted": False,
            "raw_text_rule": RAW_TEXT_RULE,
            "official_validation_used": False,
            "official_test_used": False,
            "downstream_score_used": False,
            "head_trained": False,
        },
    }


def render_split_report(manifest: dict[str, Any]) -> str:
    lines: list[str] = []
    a = lines.append
    a(f"# Pre-G1 internal split — `{manifest['schema_version']}`")
    a("")
    a(f"`{manifest['dataset']}` v{manifest['dataset_version']}, {manifest['task']}.")
    a("")
    a("| | |")
    a("|---|---|")
    a(f"| derived input SHA-256 | `{manifest['input']['derived_train_sha256']}` |")
    a(f"| derived rows | {manifest['input']['derived_train_rows']} |")
    a(f"| seed tag | `{manifest['split']['seed_tag']}` |")
    a(f"| seed | {manifest['split']['seed']} |")
    a(f"| assignment digest | `{manifest['result']['assignment_digest']}` |")
    a("")
    a("| Part | fraction | total | " + " | ".join(sorted(LABEL_MAPPING)) + " |")
    a("|---|---|---|" + "---|" * len(LABEL_MAPPING))
    for name in sorted(manifest["result"]["totals"]):
        counts = manifest["result"]["label_counts"][name]
        a(
            f"| `{name}` | {manifest['split']['fractions'][name]} | "
            f"{manifest['result']['totals'][name]} | "
            + " | ".join(str(counts.get(label, 0)) for label in sorted(LABEL_MAPPING))
            + " |"
        )
    a("")
    a("Canonical cross-part leakage: "
      f"**{manifest['result']['canonical_cross_part_leakage']}**. "
      f"Conflicting-label canonical groups: "
      f"**{manifest['input']['canonical_conflicting_label_groups']}**.")
    a("")
    a("**No corpus text is stored in any artifact of this run.** Official "
      "validation and official test took no part in this split, and no "
      "downstream score exists.")
    a("")
    return "\n".join(lines)


def materialize_split(
    pool: DerivedPool,
    output_dir: str | Path,
    *,
    repository_head: str | None = None,
    runtime_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assign, validate, then write. Refuses to overwrite an existing directory.

    Writing is staged into a sibling `.partial` directory and moved into place
    only after every invariant passes, so an interrupted or failing run cannot
    leave something that reads as an authoritative membership.
    """
    destination = Path(output_dir)
    if destination.exists():
        raise EvaluationContractViolation(
            f"refusing to overwrite an existing split directory: {destination}. "
            "Membership artifacts are immutable; use a new directory."
        )

    integrity = check_no_conflicting_groups(pool)
    parts = stratified_group_split(
        pool.records, dict(INTERNAL_SPLIT_FRACTIONS), SPLIT_SEED
    )
    validation = validate_assignment(pool, parts)

    bodies = {name: _id_file_body(parts[name]) for name in parts}
    digests = {
        ID_FILE_NAMES[name]: hashlib.sha256(body.encode("utf-8")).hexdigest()
        for name, body in bodies.items()
    }
    manifest = build_manifest(
        pool, parts, integrity, validation, digests, repository_head
    )

    staging = destination.with_name(destination.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        for name, body in bodies.items():
            (staging / ID_FILE_NAMES[name]).write_text(body, encoding="utf-8")
        (staging / "split-manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (staging / "report.md").write_text(
            render_split_report(manifest), encoding="utf-8"
        )
        if runtime_evidence is not None:
            # Deliberately separate: runtime facts vary between machines and
            # must never enter the deterministic membership artifact.
            (staging / "runtime-environment.json").write_text(
                json.dumps(runtime_evidence, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        staging.replace(destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest
