#!/usr/bin/env python3
"""B3B-0: PhoBERT input-contract and token-grid feasibility probe.

**INTENDED FOR GOOGLE COLAB, NOT THE LOCAL `.venv`.** It needs `transformers`
(tokenizer only) and, for the segmentation paths, a Java runtime plus
`py_vncorenlp`. The local environment deliberately has none of these; running it
locally prints what is missing and exits.

What it is for
--------------
Proposal v1.3 §4.4 writes the token grid as `T(b(x))` -- the frozen tokenizer
applied straight to the stripped base text -- and propagates channel labels "by
tracking character offsets through tokenization". PhoBERT's published usage
contract expects **word-segmented** input, i.e. `T(S(b(x)))`. Those are different
pipelines, and the difference decides:

* what distribution the frozen encoder actually sees;
* whether the base token grid stays invariant under corruption (§4.5);
* whether a segmenter that reads diacritics smuggles in a restoration signal;
* whether §4.4's offset-tracking step is even implementable for this tokenizer.

This script measures all of that. **It does not choose a policy** -- the report
compares the paths and stops. See `docs/spec/decisions.md` D-B3B0-001.

It never loads model weights: tokenizer only, no `AutoModel`.

Usage (Colab, inside the cloned repository)
-------------------------------------------
    pip install "transformers==4.57.6"
    # optional, for the segmentation paths (needs a JVM; Colab provides one):
    #   pip install py_vncorenlp
    export HF_HOME="$PWD/.hf-cache"
    python scripts/b3b0_phobert_input_probe.py \
        --checkpoint vinai/phobert-base \
        --revision <FULL_SHA> \
        --vncorenlp-dir .vncorenlp \
        --vncorenlp-hashes configs/linguistics/vncorenlp_v1.2_hashes.json

Two things it will NOT do, both learned from the first real run:

* it never downloads VnCoreNLP -- `download_model()` is absent, the resource must
  be provisioned externally, and `pinned=true` is claimed only when this run
  verified every file against supplied SHA-256 hashes;
* it never writes outside the repository -- output paths are resolved absolutely
  before any dependency runs, because `py_vncorenlp.VnCoreNLP()` chdir()s into
  its resource directory and the first run's artifacts landed in
  `.vncorenlp/results/b3b0/` as a result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from unmark.alignment import (  # noqa: E402
    PROBE_CONDITIONS,
    REPO_LOCAL_HF_CACHE,
    AlignmentStatus,
    OffsetAvailability,
    PathAvailability,
    PathObservation,
    PreprocessingPath,
    SegmenterContract,
    TokenizerContract,
    TokenSpan,
    alignment_status,
    character_coverage,
    compare_paths,
    syllable_token_map,
    validate_offsets,
)
from unmark.corruption import CorruptionPurpose, corrupt  # noqa: E402
from unmark.linguistics import make_classifier, try_load_inventory  # noqa: E402
from unmark.orthography import canon, decompose  # noqa: E402

STATUS_OK = "B3B0_PROBE_COMPLETE"
STATUS_PARTIAL = "B3B0_PROBE_PARTIAL"

# The proposal names "PhoBERT-base" (§6.1) but pins no repository or revision.
# This default is a probe convenience, NOT a specification lock; see audit 006.
DEFAULT_CHECKPOINT = "vinai/phobert-base"

# Representative cases. Expected tokenizer output is deliberately NOT hard-coded:
# nothing here asserts what PhoBERT will produce.
CASES: tuple[tuple[str, str], ...] = (
    ("vi_research", "Tôi đang nghiên cứu xử lý ngôn ngữ tự nhiên."),
    ("vi_multisyllable", "Trường đại học công nghệ thông tin"),
    ("vi_city", "Thành phố Hồ Chí Minh rất đẹp"),
    ("vi_proper_names", "Nguyễn Việt Anh đang làm việc tại Hà Nội"),
    ("vi_all_tones", "ma má mà mả mã mạ"),
    ("vi_letter_diacritics", "đường ăn cân ơn ưu êm ôm Đại"),
    ("vi_uppercase", "ĐẠI HỌC KHOA HỌC TỰ NHIÊN"),
    ("mixed_en", "tôi dùng Python và PyTorch để train model"),
    ("mixed_ml", "Tôi đang học machine learning tại VNU-HCM"),
    ("ascii_ambiguous", "ban the com on in an la co"),
    ("punctuation", "Năm 2026, GDP tăng 6,5% (VAT 10%)!"),
    ("numbers_dates", "Cuộc họp lúc 14:30 ngày 19/08/2026."),
    ("url", "Xem tại https://example.edu.vn/tuyen-sinh?id=42&lang=vi"),
    ("email", "Liên hệ qua lien.he@example.com nhé"),
    ("emoji", "hôm nay tôi rất vui 😄🎉"),
    ("hyphenated", "Việt-Nam và VNU-HCM là tên riêng"),
    ("presegmented", "Tôi đang nghiên_cứu xử_lý ngôn_ngữ tự_nhiên"),
    ("long_sentence", (
        "Trường Đại học Khoa học Tự nhiên trực thuộc Đại học Quốc gia Thành phố Hồ Chí Minh "
        "là một trong những cơ sở đào tạo và nghiên cứu khoa học công nghệ hàng đầu của cả nước."
    )),
)

SEED = 20260819


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
FULL_SHA_LENGTH = 40
_SNAPSHOT_PATTERN = re.compile(r"[/\\]snapshots[/\\]([0-9a-f]{40})(?:[/\\]|$)")


def is_full_commit_sha(value: str | None) -> bool:
    """Whether `value` is a full immutable Git commit SHA.

    Branch names, tags and abbreviated SHAs are all mutable or ambiguous: `main`
    moves, a tag can be re-pointed, and a short SHA is a prefix. A scientific
    probe must name the exact commit.
    """
    # Lowercase only: the hub emits lowercase, and accepting mixed case here
    # would turn an argument-format problem into a confusing comparison failure
    # later.
    return (
        isinstance(value, str)
        and len(value) == FULL_SHA_LENGTH
        and all(c in "0123456789abcdef" for c in value)
    )


def extract_snapshot_revision(path: str) -> str | None:
    """Pull the commit SHA out of a Hugging Face cache path.

    The hub caches as `models--org--name/snapshots/<commit_sha>/<file>`, and the
    snapshot directory is always the *resolved commit* -- passing `main` still
    lands under the SHA it resolved to. Reading it back from a file the
    tokenizer actually loaded is therefore genuine post-load evidence, not a
    restatement of the request.
    """
    if not isinstance(path, str):
        return None
    match = _SNAPSHOT_PATTERN.search(path)
    return match.group(1) if match else None


def _candidate_resolved_paths(tokenizer) -> list[str]:
    """Paths of files the tokenizer actually loaded from.

    Drawn from documented attributes and `init_kwargs`, not from a guessed
    private field: anything that looks like an existing file path is considered,
    and paths that carry no snapshot component are simply ignored.
    """
    candidates: list[str] = []

    def consider(value: Any) -> None:
        if isinstance(value, str) and value and os.sep in value:
            candidates.append(value)

    for attribute in ("vocab_file", "merges_file", "tokenizer_file"):
        consider(getattr(tokenizer, attribute, None))
    init_kwargs = getattr(tokenizer, "init_kwargs", None)
    if isinstance(init_kwargs, dict):
        for value in init_kwargs.values():
            consider(value)
    for attribute in ("name_or_path", "_tokenizer_file"):
        consider(getattr(tokenizer, attribute, None))
    # Preserve order, drop duplicates.
    seen: list[str] = []
    for path in candidates:
        if path not in seen:
            seen.append(path)
    return seen


def observe_tokenizer_revision(tokenizer) -> tuple[str | None, tuple[str, ...], str]:
    """Read the resolved commit back off the loaded tokenizer.

    Returns `(revision, evidence_paths, source_description)`. `revision` is None
    when no snapshot path could be found, or when the paths disagree -- either
    way the probe must not claim verification.

    Deliberately does no network I/O and issues no second download: re-resolving
    the repository would either restate the request or introduce a floating
    lookup, and neither is evidence.
    """
    paths = _candidate_resolved_paths(tokenizer)
    found: dict[str, list[str]] = {}
    for path in paths:
        revision = extract_snapshot_revision(path)
        if revision:
            found.setdefault(revision, []).append(path)

    if not found:
        return (
            None,
            tuple(paths[:5]),
            "no Hugging Face snapshot path was found among the tokenizer's resolved files; "
            "the tokenizer may have been loaded from a local directory rather than the hub cache",
        )
    if len(found) > 1:
        return (
            None,
            tuple(p for group in found.values() for p in group)[:5],
            f"resolved files disagree about the snapshot revision: {sorted(found)}",
        )
    revision, evidence = next(iter(found.items()))
    return revision, tuple(evidence[:5]), "hugging face cache snapshot path of the loaded tokenizer files"


def load_tokenizer(checkpoint: str, revision: str | None, use_fast: bool):
    """Tokenizer only. `AutoModel` is never called anywhere in this script."""
    from transformers import AutoTokenizer

    kwargs: dict[str, Any] = {"use_fast": use_fast}
    if revision:
        kwargs["revision"] = revision
    return AutoTokenizer.from_pretrained(checkpoint, **kwargs)


def describe_tokenizer(tokenizer, checkpoint: str, revision: str | None) -> TokenizerContract:
    import transformers

    observed, evidence, source = observe_tokenizer_revision(tokenizer)
    verified = bool(revision) and observed is not None and observed == revision

    return TokenizerContract(
        checkpoint=checkpoint,
        revision_requested=revision,
        revision_observed=observed,
        revision_verified=verified,
        revision_evidence=evidence,
        revision_evidence_source=source,
        tokenizer_class=type(tokenizer).__name__,
        is_fast=bool(getattr(tokenizer, "is_fast", False)),
        vocab_size=getattr(tokenizer, "vocab_size", None),
        unk_token=getattr(tokenizer, "unk_token", None),
        special_tokens=tuple(getattr(tokenizer, "all_special_tokens", ()) or ()),
        # Recorded as a documented expectation, not inferred from behaviour.
        word_segmentation_expected=True,
        transformers_version=transformers.__version__,
        notes=(
            "PhoBERT's model card states that input should be Vietnamese word-segmented "
            "(VnCoreNLP/RDRSegmenter) as in pretraining. Verify against the card for this "
            "exact checkpoint before locking any policy."
        ),
    )


REQUIRED_SEGMENTER_FILES: tuple[str, ...] = (
    "models/wordsegmenter/vi-vocab",
    "models/wordsegmenter/wordsegmenter.rdr",
)
JAR_GLOB = "VnCoreNLP-*.jar"
DEFAULT_REQUIRED_JAR = "VnCoreNLP-1.2.jar"


def git_head_revision(checkout: Path) -> str | None:
    """`git rev-parse HEAD` for a checkout, or None when unavailable.

    A local subprocess; no network. Returns None rather than raising so a
    non-Git provisioning (an unpacked archive, say) still gets its resource
    hashes checked -- it simply cannot be marked pinned.
    """
    if not (checkout / ".git").exists():
        return None
    try:
        result = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


def git_tags_at_head(checkout: Path) -> tuple[str, ...]:
    """Tags pointing at HEAD. Diagnostic only -- never verification."""
    if not (checkout / ".git").exists():
        return ()
    try:
        result = subprocess.run(
            ["git", "-C", str(checkout), "tag", "--points-at", "HEAD"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return ()
    if result.returncode != 0:
        return ()
    return tuple(line.strip() for line in result.stdout.splitlines() if line.strip())


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


PLACEHOLDER_MARKER = "PENDING"
_SHA256_LENGTH = 64
DEFAULT_VNCORENLP_MANIFEST = "configs/linguistics/vncorenlp_v1.2.json"

REQUIRED_MANIFEST_KEYS = (
    "schema_version",
    "source",
    "source_repository",
    "release_tag",
    "revision",
    "required_jar",
    "files",
)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(c in "0123456789abcdef" for c in value.lower())
    )


class ManifestIncomplete(SystemExit):
    """Raised when the committed pin still carries unresolved placeholders."""


def load_vncorenlp_manifest(path: Path | None) -> dict[str, Any]:
    """Read and validate the committed VnCoreNLP pin.

    Fails closed on anything short of a complete pin: a manifest with a
    placeholder digest is worse than no manifest, because it *looks* like
    provenance.
    """
    if path is None:
        return {}
    if not path.is_file():
        raise SystemExit(f"VnCoreNLP manifest not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit(f"{path}: manifest must be a JSON object")

    missing = [key for key in REQUIRED_MANIFEST_KEYS if key not in data]
    if missing:
        raise SystemExit(f"{path}: manifest is missing required key(s): {', '.join(missing)}")

    files = data["files"]
    if not isinstance(files, dict) or not files:
        raise SystemExit(f"{path}: 'files' must be a non-empty object of relpath -> sha256")

    required_jar = data["required_jar"]
    if required_jar not in files:
        raise SystemExit(f"{path}: required_jar {required_jar!r} has no entry in 'files'")
    for rel in REQUIRED_SEGMENTER_FILES:
        if rel not in files:
            raise SystemExit(f"{path}: 'files' is missing the required resource {rel!r}")

    unresolved = [name for name, digest in files.items() if not _is_sha256(digest)]
    if not _is_sha256(data["revision"]) and len(str(data["revision"])) != 40:
        unresolved.append("revision")
    if unresolved:
        raise ManifestIncomplete(
            f"{path} is not a usable pin: {', '.join(sorted(unresolved))} "
            f"still carr{'ies' if len(unresolved) == 1 else 'y'} a placeholder.\n\n"
            "The exact VnCoreNLP v1.2 Git revision and the SHA-256 of\n"
            "  VnCoreNLP-1.2.jar\n"
            "  models/wordsegmenter/vi-vocab\n"
            "  models/wordsegmenter/wordsegmenter.rdr\n"
            "must be supplied from the Colab provisioning cells that fetched the pinned\n"
            "checkout. They cannot be derived here: .vncorenlp/ is a Colab-side runtime\n"
            "directory, and inventing a digest would defeat the purpose of the pin.\n\n"
            "Fill them into the manifest, then rerun. Until then the probe refuses to\n"
            "load the segmenter and no run can be marked scientifically usable."
        )
    return data


def load_expected_hashes(path: Path | None) -> dict[str, Any]:
    """Legacy `--vncorenlp-hashes` reader, retained for compatibility.

    The committed manifest is the canonical path; this exists so an ad-hoc
    hashes file still works, and so a conflict between the two can be detected.
    """
    if path is None:
        return {}
    if not path.is_file():
        raise SystemExit(f"--vncorenlp-hashes file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "files" not in data:
        raise SystemExit(f"{path}: expected an object with a 'files' mapping")
    return data


def reconcile_provenance(
    manifest: dict[str, Any],
    hashes: dict[str, Any],
    supplied_revision: str | None,
) -> dict[str, Any]:
    """Merge the provenance sources, refusing on any contradiction.

    The committed manifest wins nothing by default -- a disagreement is an error,
    not a precedence question. Silently preferring one source would let a stale
    CLI value quietly override the repository's pin.
    """
    if not manifest:
        return hashes

    revision = manifest["revision"]
    if supplied_revision and supplied_revision != revision:
        raise SystemExit(
            f"--vncorenlp-revision {supplied_revision!r} contradicts the committed manifest "
            f"revision {revision!r}. Resolve the disagreement rather than overriding: if the "
            "pin changed, update configs/linguistics/vncorenlp_v1.2.json and record the "
            "change in docs/spec/decisions.md."
        )
    if hashes:
        hash_revision = hashes.get("revision")
        if hash_revision and hash_revision != revision:
            raise SystemExit(
                f"--vncorenlp-hashes revision {hash_revision!r} contradicts the committed "
                f"manifest revision {revision!r}."
            )
        conflicts = [
            name
            for name, digest in (hashes.get("files") or {}).items()
            if name in manifest["files"] and manifest["files"][name] != digest
        ]
        if conflicts:
            raise SystemExit(
                "--vncorenlp-hashes contradicts the committed manifest for: "
                f"{', '.join(sorted(conflicts))}."
            )
    return manifest


def load_segmenter(
    resource_dir: Path | None,
    expected: dict[str, Any],
    supplied_revision: str | None,
) -> tuple[Any, SegmenterContract]:
    """Load VnCoreNLP from an ALREADY-PRESENT, externally provisioned directory.

    Never downloads. `py_vncorenlp.download_model()` is deliberately absent from
    this script: the first Colab run showed that an automatic downloader makes the
    segmentation model unpinned, which silently invalidates every number that
    depends on segmentation. The resource must be provisioned and verified
    outside the probe, and `pinned=True` is claimed only when this run actually
    checked the files against supplied hashes.
    """
    if resource_dir is None:
        return None, SegmenterContract(
            available=False,
            name="VnCoreNLP",
            package="py_vncorenlp",
            notes="no --vncorenlp-dir supplied; segmentation paths not probed",
        )

    if not resource_dir.is_dir():
        return None, SegmenterContract(
            available=False, name="VnCoreNLP", package="py_vncorenlp",
            model_resource=str(resource_dir),
            notes=(
                f"resource directory does not exist: {resource_dir}. The probe never "
                "downloads it; provision VnCoreNLP externally and pass --vncorenlp-dir."
            ),
        )

    # The jar is named by the pin, never discovered. Picking "the first matching
    # VnCoreNLP-*.jar" would silently choose a version when several are present
    # (audit 007 N2).
    required_jar = str(expected.get("required_jar") or DEFAULT_REQUIRED_JAR)
    jar = resource_dir / required_jar
    other_jars = sorted(p.name for p in resource_dir.glob(JAR_GLOB) if p.name != required_jar)

    missing = [rel for rel in REQUIRED_SEGMENTER_FILES if not (resource_dir / rel).is_file()]
    if not jar.is_file():
        missing.append(required_jar)
    if missing:
        note = f"missing required resource(s) under {resource_dir}: {', '.join(missing)}"
        if other_jars:
            note += (
                f". Other VnCoreNLP jars are present ({', '.join(other_jars)}) but are NOT "
                "substituted: the pin names exactly one."
            )
        return None, SegmenterContract(
            available=False, name="VnCoreNLP", package="py_vncorenlp",
            model_resource=str(resource_dir), required_jar=required_jar,
            other_jars_present=tuple(other_jars),
            notes=note,
        )
    observed: dict[str, str] = {required_jar: sha256_of(jar)}
    for rel in REQUIRED_SEGMENTER_FILES:
        observed[rel] = sha256_of(resource_dir / rel)

    expected_files = dict(expected.get("files") or {})
    mismatches = [
        name for name, digest in expected_files.items()
        if name in observed and observed[name] != digest
    ]
    unverified = [name for name in observed if name not in expected_files]
    # Hashes verified only when EVERY observed resource was checked against a
    # pinned digest and matched. Existence alone never counts.
    hashes_verified = bool(expected_files) and not mismatches and not unverified

    # Revision verification (audit 008 N1). The manifest pins a Git revision, so
    # content hashes alone are not sufficient for scientific usability.
    manifest_revision = expected.get("revision")
    observed_revision = git_head_revision(resource_dir)
    observed_tags = git_tags_at_head(resource_dir)
    revision_verified = bool(
        manifest_revision and observed_revision and observed_revision == manifest_revision
    )

    # pinned is the conjunction: both the checkout identity and its contents.
    verified = hashes_verified and revision_verified

    # Provenance describes the FILES ON DISK, so it is recorded on every return
    # path below -- including failures. Whether the library imported is a
    # separate fact from what the checkout contains.
    resolved_revision = expected.get("revision") or supplied_revision

    def contract(available: bool, note: str, package_version: str | None = None) -> SegmenterContract:
        return SegmenterContract(
            available=available,
            name="VnCoreNLP",
            package="py_vncorenlp",
            package_version=package_version,
            model_resource=str(resource_dir),
            model_version=resolved_revision,
            jar_name=required_jar if available else None,
            required_jar=required_jar,
            other_jars_present=tuple(other_jars),
            manifest_path=str(expected.get("_manifest_path") or "") or None,
            manifest_revision=manifest_revision,
            observed_revision=observed_revision,
            revision_verified=revision_verified,
            observed_tags_at_head=observed_tags,
            expected_hashes=expected_files,
            resource_hashes=observed,
            hashes_verified=hashes_verified,
            # `pinned` reflects verification of the files, so it stays true even
            # if the library then fails to load.
            pinned=verified,
            notes=note,
        )

    if mismatches:
        return None, contract(
            False,
            "REFUSING: resource SHA-256 mismatch against the pin for "
            f"{', '.join(sorted(mismatches))}. The checkout is not the pinned one.",
        )

    if manifest_revision and observed_revision and observed_revision != manifest_revision:
        return None, contract(
            False,
            f"REFUSING: checkout HEAD {observed_revision} != pinned revision "
            f"{manifest_revision}. Check out the pinned revision, or update "
            "configs/linguistics/vncorenlp_v1.2.json and record the dependency change "
            "in docs/spec/decisions.md.",
        )

    try:
        import py_vncorenlp
    except Exception as exc:  # noqa: BLE001
        return None, contract(False, f"import failed: {type(exc).__name__}: {exc}")

    try:
        # NOTE: this call chdir()s into save_dir. Every output path is resolved
        # to an absolute Path before we get here, so it cannot move artifacts.
        segmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=str(resource_dir))
    except Exception as exc:  # noqa: BLE001
        return None, contract(
            False,
            f"initialisation failed: {type(exc).__name__}: {exc}",
            getattr(py_vncorenlp, "__version__", None),
        )

    return segmenter, contract(
        True,
        (
            "resources externally provisioned and verified against supplied SHA-256"
            if verified
            else (
                "resources externally provisioned but NOT verified: "
                + (
                    "no --vncorenlp-hashes supplied"
                    if not expected_files
                    else f"no supplied hash for {', '.join(unverified)}"
                )
                + ". pinned=false; results depending on segmentation are not reproducible."
            )
        ),
        getattr(py_vncorenlp, "__version__", None),
    )


def segment(segmenter, text: str) -> str:
    if segmenter is None:
        raise RuntimeError("segmenter unavailable")
    output = segmenter.word_segment(text)
    return " ".join(output) if isinstance(output, list) else str(output)


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------
def build_spans(tokenizer, encoding, tokens: Sequence[str]) -> list[TokenSpan]:
    offsets = encoding.get("offset_mapping")
    ids = list(encoding.get("input_ids") or [])
    specials = set(getattr(tokenizer, "all_special_tokens", ()) or ())
    unk = getattr(tokenizer, "unk_token", None)

    spans: list[TokenSpan] = []
    for index, token in enumerate(tokens):
        start = end = None
        if offsets is not None and index < len(offsets):
            pair = offsets[index]
            if pair is not None and len(pair) == 2:
                start, end = int(pair[0]), int(pair[1])
        is_special = token in specials
        spans.append(
            TokenSpan(
                index=index,
                token=token,
                token_id=ids[index] if index < len(ids) else None,
                start=None if is_special else start,
                end=None if is_special else end,
                is_special=is_special,
                is_unknown=(unk is not None and token == unk),
            )
        )
    return spans


def probe_one(
    tokenizer,
    case_id: str,
    condition: str,
    path: PreprocessingPath,
    source_text: str,
    canonical_text: str,
    base_text: str,
    tokenizer_input: str,
    segmented_text: str | None,
    syllable_ranges: Sequence[tuple[int, int]],
    eligibility: Sequence[str],
    tones: Sequence[str],
) -> PathObservation:
    try:
        try:
            encoding = tokenizer(tokenizer_input, return_offsets_mapping=True)
        except (NotImplementedError, ValueError, TypeError):
            # Slow tokenizers reject return_offsets_mapping; that is a finding.
            encoding = tokenizer(tokenizer_input)
        encoding = dict(encoding)
        tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
        spans = build_spans(tokenizer, encoding, tokens)

        availability, offset_reason = validate_offsets(tokenizer_input, spans)
        coverage = character_coverage(tokenizer_input, spans)
        mapping = syllable_token_map(spans, list(syllable_ranges))
        status, status_reason = alignment_status(availability, coverage, mapping)

        return PathObservation(
            case_id=case_id,
            condition=condition,
            path=path,
            availability=PathAvailability.OK,
            source_text=source_text,
            canonical_text=canonical_text,
            base_text=base_text,
            segmented_text=segmented_text,
            tokenizer_input=tokenizer_input,
            tokens=tuple(tokens),
            token_ids=tuple(encoding["input_ids"]),
            spans=tuple(spans),
            offset_availability=availability,
            offset_reason=offset_reason,
            alignment=status,
            alignment_reason=status_reason,
            coverage=coverage,
            syllable_map=mapping,
            eligibility=tuple(eligibility),
            observed_tones=tuple(tones),
        )
    except Exception as exc:  # noqa: BLE001 - one bad case must not abort the probe
        return PathObservation(
            case_id=case_id,
            condition=condition,
            path=path,
            availability=PathAvailability.ERROR,
            source_text=source_text,
            canonical_text=canonical_text,
            base_text=base_text,
            tokenizer_input=tokenizer_input,
            error=f"{type(exc).__name__}: {exc}",
        )


def run_probe(tokenizer, segmenter, segmenter_contract: SegmenterContract) -> list[PathObservation]:
    inventory = try_load_inventory()
    classifier = make_classifier(inventory) if inventory else None
    observations: list[PathObservation] = []

    for case_id, clean_text in CASES:
        for condition in PROBE_CONDITIONS:
            # With the pinned inventory present this is a proper SCIENTIFIC call.
            # Without it, B2's guard would refuse, so the probe drops to
            # SELF_CHECK and the artifacts record that it did.
            result = corrupt(
                clean_text,
                condition,
                seed=SEED,
                sample_id=case_id,
                purpose=CorruptionPurpose.SCIENTIFIC if inventory else CorruptionPurpose.SELF_CHECK,
            )
            observed = result.corrupted_text
            base_text = result.corrupted_decomposition.base_text
            decomposed = decompose(base_text, eligibility_classifier=classifier)
            ranges = [(s.base_start, s.base_end) for s in decomposed.syllables]
            eligibility = [s.eligibility.value for s in decomposed.syllables]
            tones = [d.corrupted_observed_tone.value for d in result.decisions]

            # PATH RAW: T(b(x)) exactly as the proposal writes it.
            observations.append(
                probe_one(
                    tokenizer, case_id, condition, PreprocessingPath.RAW_BASE,
                    clean_text, result.canonical_clean_text, base_text,
                    base_text, None, ranges, eligibility, tones,
                )
            )

            # PATH BASE-THEN-SEG: strip to the invariant base, then segment it.
            if segmenter_contract.available:
                try:
                    segmented = segment(segmenter, base_text)
                    seg_decomposed = decompose(segmented, eligibility_classifier=classifier)
                    seg_ranges = [(s.base_start, s.base_end) for s in seg_decomposed.syllables]
                    observations.append(
                        probe_one(
                            tokenizer, case_id, condition, PreprocessingPath.BASE_THEN_SEGMENT,
                            clean_text, result.canonical_clean_text, base_text,
                            segmented, segmented, seg_ranges, eligibility, tones,
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    observations.append(
                        PathObservation(
                            case_id=case_id, condition=condition,
                            path=PreprocessingPath.BASE_THEN_SEGMENT,
                            availability=PathAvailability.ERROR, base_text=base_text,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    )
            else:
                observations.append(
                    PathObservation(
                        case_id=case_id, condition=condition,
                        path=PreprocessingPath.BASE_THEN_SEGMENT,
                        availability=PathAvailability.UNAVAILABLE_SEGMENTER,
                        base_text=base_text,
                    )
                )

            # PATH CLEAN-SEG-THEN-BASE: segment the CLEAN text, then strip.
            # Deployability red flag by construction: needs clean text at
            # inference. Measured anyway, because it is the path that best
            # matches PhoBERT's pretraining distribution.
            if segmenter_contract.available:
                try:
                    segmented_clean = segment(segmenter, canon(clean_text))
                    stripped_seg = decompose(segmented_clean, eligibility_classifier=classifier)
                    stripped_text = stripped_seg.base_text
                    ranges_cs = [(s.base_start, s.base_end) for s in stripped_seg.syllables]
                    elig_cs = [s.eligibility.value for s in stripped_seg.syllables]
                    observations.append(
                        probe_one(
                            tokenizer, case_id, condition,
                            PreprocessingPath.CLEAN_SEGMENT_THEN_BASE,
                            clean_text, result.canonical_clean_text, stripped_text,
                            stripped_text, segmented_clean, ranges_cs, elig_cs, tones,
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    observations.append(
                        PathObservation(
                            case_id=case_id, condition=condition,
                            path=PreprocessingPath.CLEAN_SEGMENT_THEN_BASE,
                            availability=PathAvailability.ERROR, base_text=base_text,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    )
            else:
                observations.append(
                    PathObservation(
                        case_id=case_id, condition=condition,
                        path=PreprocessingPath.CLEAN_SEGMENT_THEN_BASE,
                        availability=PathAvailability.UNAVAILABLE_SEGMENTER,
                        base_text=base_text,
                    )
                )

            # PATH OBSERVED-SEG-THEN-BASE: segment what is actually observed.
            if segmenter_contract.available:
                try:
                    segmented_obs = segment(segmenter, observed)
                    obs_decomposed = decompose(segmented_obs, eligibility_classifier=classifier)
                    obs_text = obs_decomposed.base_text
                    obs_ranges = [(s.base_start, s.base_end) for s in obs_decomposed.syllables]
                    obs_elig = [s.eligibility.value for s in obs_decomposed.syllables]
                    observations.append(
                        probe_one(
                            tokenizer, case_id, condition,
                            PreprocessingPath.OBSERVED_SEGMENT_THEN_BASE,
                            clean_text, result.canonical_clean_text, obs_text,
                            obs_text, segmented_obs, obs_ranges, obs_elig, tones,
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    observations.append(
                        PathObservation(
                            case_id=case_id, condition=condition,
                            path=PreprocessingPath.OBSERVED_SEGMENT_THEN_BASE,
                            availability=PathAvailability.ERROR, base_text=base_text,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    )
            else:
                observations.append(
                    PathObservation(
                        case_id=case_id, condition=condition,
                        path=PreprocessingPath.OBSERVED_SEGMENT_THEN_BASE,
                        availability=PathAvailability.UNAVAILABLE_SEGMENTER,
                        base_text=base_text,
                    )
                )
    return observations


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def render_report(config: dict[str, Any], comparison: dict[str, Any], observations: Sequence[PathObservation]) -> str:
    lines: list[str] = []
    a = lines.append
    a("# B3B-0 PhoBERT input-contract probe")
    a("")
    a(f"Run id: `{config['run_id']}`  ")
    a(f"Timestamp (UTC): `{config['timestamp_utc']}`")
    a("")
    a("> **This report does not choose a preprocessing policy.** It measures the")
    a("> candidate pipelines so the researcher can choose with evidence. No model")
    a("> weights were loaded; tokenizer only. See `docs/spec/decisions.md` D-B3B0-001.")
    a("")
    a("## Environment")
    a("")
    a("| Field | Value |")
    a("|---|---|")
    for key, value in config["tokenizer"].items():
        a(f"| tokenizer.{key} | {value} |")
    for key, value in config["segmenter"].items():
        a(f"| segmenter.{key} | {value} |")
    a(f"| python | {config['python_version']} |")
    a("")
    a("## Reproducibility and paths")
    a("")
    a("| Field | Value |")
    a("|---|---|")
    for key in (
        "tokenizer_revision_requested", "tokenizer_revision_observed",
        "tokenizer_revision_verified", "scientifically_usable",
        "repository_root", "resolved_output_root",
        "resolved_vncorenlp_dir", "cwd_at_start", "cwd_after_segmenter_initialization",
        "cwd_changed_by_dependency",
    ):
        a(f"| `{key}` | {config.get(key)} |")
    a("")
    phobert = config.get("phobert_provenance") or {}
    if phobert:
        a("### PhoBERT tokenizer provenance")
        a("")
        a("| Field | Value |")
        a("|---|---|")
        for key in (
            "checkpoint", "revision_requested", "revision_observed", "revision_verified",
            "revision_evidence_source", "tokenizer_class", "is_fast",
        ):
            a(f"| `{key}` | {phobert.get(key)} |")
        for path in phobert.get("revision_evidence") or []:
            a(f"| evidence | `{path}` |")
        a("")
        if not phobert.get("revision_verified"):
            a("> The tokenizer revision was **not verified**: supplying `--revision` is an")
            a("> argument, not a verification. This run is not scientifically usable.")
            a("")

    provenance = config.get("vncorenlp_provenance") or {}
    if provenance:
        a("### VnCoreNLP provenance")
        a("")
        a("| Field | Value |")
        a("|---|---|")
        for key in (
            "manifest_path", "manifest_revision", "observed_revision", "revision_verified",
            "observed_tags_at_head", "required_jar", "jar_name", "other_jars_present",
            "hashes_verified", "pinned",
        ):
            a(f"| `{key}` | {provenance.get(key)} |")
        a("")
        expected_hashes = provenance.get("expected_hashes") or {}
        observed_hashes = provenance.get("resource_hashes") or {}
        if expected_hashes or observed_hashes:
            a("| Resource | Expected | Observed | Match |")
            a("|---|---|---|---|")
            for name in sorted(set(expected_hashes) | set(observed_hashes)):
                want, got = expected_hashes.get(name), observed_hashes.get(name)
                a(f"| `{name}` | `{want}` | `{got}` | {want == got if want and got else 'n/a'} |")
            a("")
    if config.get("cwd_changed_by_dependency"):
        a("> A dependency changed the working directory during setup. Every output path was")
        a("> resolved absolutely beforehand, so artifacts are unaffected -- this row exists")
        a("> because the first Colab run wrote into `.vncorenlp/results/b3b0/` for exactly")
        a("> this reason.")
        a("")
    if not config.get("scientifically_usable"):
        a("> **NOT SCIENTIFICALLY USABLE.** Either the tokenizer revision was not pinned or")
        a("> the segmenter resources were not verified against supplied SHA-256 hashes. The")
        a("> measurements below are exploratory; do not base a preprocessing decision on")
        a("> them.")
        a("")
    if not config["segmenter"].get("available"):
        a("> **Segmentation paths UNAVAILABLE.** VnCoreNLP was not usable in this run, so")
        a("> the three segmentation pathways are reported as `UNAVAILABLE_SEGMENTER`")
        a("> rather than faked. Only `RAW_BASE` carries measurements.")
        a("")
    elif not config["segmenter"].get("pinned"):
        a("> **Segmenter NOT verified.** The resources were used as found but not checked")
        a("> against supplied SHA-256 hashes, so `pinned=false`. Pass `--vncorenlp-hashes`")
        a("> before any result depends on segmentation.")
        a("")
    else:
        a("> Segmenter resources verified against supplied SHA-256 hashes.")
        a("")

    a("## Preprocessing paths compared")
    a("")
    a("| Path | Observations | Usable | Aligned | Mean fragmentation | Unknown tokens | Grid invariant |")
    a("|---|---:|---:|---:|---:|---:|---|")
    for name, summary in comparison["path_summaries"].items():
        invariance = comparison["grid_invariance"].get(name, {})
        frag = summary["mean_fragmentation"]
        a(
            f"| `{name}` | {summary['observations']} | {summary['usable']} | {summary['aligned']} | "
            f"{'n/a' if frag is None else f'{frag:.3f}'} | {summary['total_unknown_tokens']} | "
            f"{invariance.get('all_cases_invariant', 'n/a')} |"
        )
    a("")
    a("`Grid invariant` is the load-bearing column: proposal §4.5 requires the base token")
    a("grid to be identical across every corruption condition. A path that fails it cannot")
    a("be used for UNMARK regardless of how well it matches PhoBERT's training distribution.")
    a("")

    a("## Offset availability")
    a("")
    a("| Path | Offset availability observed |")
    a("|---|---|")
    for name, summary in comparison["path_summaries"].items():
        a(f"| `{name}` | {', '.join(summary['offset_availability']) or 'n/a'} |")
    a("")
    a("§4.4 propagates channel labels \"by tracking character offsets through")
    a("tokenization\". If this reads `ABSENT` or `NATIVE_MALFORMED`, that step is not")
    a("implementable as written for this tokenizer and a deterministic manual alignment")
    a("strategy will be needed -- B3B's problem, not B3B-0's.")
    a("")

    a("## Grid invariance detail")
    a("")
    for name, invariance in comparison["grid_invariance"].items():
        a(f"### `{name}`")
        a("")
        a(f"Cases comparable: {invariance['cases_comparable']}; "
          f"satisfying grid invariance: {invariance['cases_satisfying_grid_invariance']}")
        a("")
        broken = {
            case: result
            for case, result in invariance["per_case"].items()
            if result.get("comparable") and not result.get("satisfies_base_grid_invariance")
        }
        if broken:
            a("| Case | base text | tokenizer input | token ids |")
            a("|---|---|---|---|")
            for case, result in list(broken.items())[:15]:
                a(
                    f"| `{case}` | {result['base_text_invariant']} | "
                    f"{result['tokenizer_input_invariant']} | {result['token_ids_invariant']} |"
                )
        else:
            a("No invariance breaks observed.")
        a("")

    a("## Decision")
    a("")
    a(f"**{comparison['decision']}** — {comparison['decision_note']}")
    a("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="B3B-0 PhoBERT input-contract probe (Colab).")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--revision",
        default=None,
        help="REQUIRED: full commit SHA of the tokenizer revision to probe",
    )
    parser.add_argument("--use-fast", action="store_true", default=True)
    parser.add_argument("--no-fast", dest="use_fast", action="store_false")
    parser.add_argument(
        "--output-root",
        default="results/b3b0",
        help="relative values resolve against the repository root, never the cwd",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--vncorenlp-dir",
        default=None,
        help="directory holding an ALREADY-PROVISIONED VnCoreNLP checkout; never downloaded",
    )
    parser.add_argument(
        "--vncorenlp-manifest",
        default=DEFAULT_VNCORENLP_MANIFEST,
        help="committed VnCoreNLP pin; the canonical scientific provenance source",
    )
    parser.add_argument("--vncorenlp-revision", default=None, help="legacy: externally supplied revision")
    parser.add_argument(
        "--vncorenlp-hashes",
        default=None,
        help='JSON {"revision": ..., "files": {relpath: sha256}}; pinned=true requires it',
    )
    parser.add_argument(
        "--allow-floating-revision",
        action="store_true",
        help="run without --revision; marks the run NOT scientifically usable",
    )
    parser.add_argument("--skip-segmenter", action="store_true")
    args = parser.parse_args(argv)

    # Resolve every path to an absolute Path BEFORE any third-party code runs.
    # py_vncorenlp.VnCoreNLP() chdir()s into its resource directory, which is how
    # the first Colab run wrote its artifacts into .vncorenlp/results/b3b0/.
    cwd_at_start = Path.cwd()
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (REPO_ROOT / output_root).resolve()
    vncorenlp_dir = None
    if args.vncorenlp_dir and not args.skip_segmenter:
        vncorenlp_dir = Path(args.vncorenlp_dir)
        if not vncorenlp_dir.is_absolute():
            vncorenlp_dir = (REPO_ROOT / vncorenlp_dir).resolve()
    hashes_path = None
    if args.vncorenlp_hashes:
        hashes_path = Path(args.vncorenlp_hashes)
        if not hashes_path.is_absolute():
            hashes_path = (REPO_ROOT / hashes_path).resolve()
    manifest_path = None
    if args.vncorenlp_manifest:
        manifest_path = Path(args.vncorenlp_manifest)
        if not manifest_path.is_absolute():
            manifest_path = (REPO_ROOT / manifest_path).resolve()

    # A scientific probe must name the tokenizer revision it used, as a full
    # immutable commit SHA.
    if args.revision and not args.allow_floating_revision and not is_full_commit_sha(args.revision):
        print(
            f"--revision {args.revision!r} is not a full immutable commit SHA.\n\n"
            "Branch names (`main`, `master`), tags and abbreviated SHAs are mutable or\n"
            "ambiguous: a branch moves, a tag can be re-pointed, and a short SHA is a\n"
            f"prefix. Supply the full {FULL_SHA_LENGTH}-character lowercase commit hash of\n"
            "the tokenizer revision to probe.\n\n"
            "To run anyway for exploration only, pass --allow-floating-revision; the\n"
            "artifacts will record revision_verified=false and scientifically_usable=false.\n",
            file=sys.stderr,
        )
        return 2

    if not args.revision and not args.allow_floating_revision:
        print(
            "--revision is required.\n\n"
            "A floating checkpoint revision makes the probe unreproducible: the tokenizer\n"
            "could change between runs and the result would not be attributable. Supply the\n"
            "full commit SHA of the tokenizer you intend to probe:\n\n"
            f"    --checkpoint {args.checkpoint} --revision <FULL_SHA>\n\n"
            "Resolve it from the model's Hugging Face page or the API. To run anyway for\n"
            "exploration only, pass --allow-floating-revision; the artifacts will record\n"
            "revision_pinned=false and must not be used for a scientific decision.\n",
            file=sys.stderr,
        )
        return 2

    try:
        import transformers  # noqa: F401
    except ImportError:
        print(
            "transformers is not installed.\n\n"
            "This probe is intended for Google Colab, not the local .venv, which is "
            "deliberately ML-free. In Colab:\n\n"
            '    pip install "transformers==4.57.6"\n'
            "    pip install py_vncorenlp        # optional, needs a JVM\n"
            f'    export HF_HOME="$PWD/{REPO_LOCAL_HF_CACHE}"\n'
            f"    python scripts/b3b0_phobert_input_probe.py --checkpoint {args.checkpoint}\n",
            file=sys.stderr,
        )
        return 2

    print(f"Loading tokenizer only (no model weights): {args.checkpoint}")
    tokenizer = load_tokenizer(args.checkpoint, args.revision, args.use_fast)
    tokenizer_contract = describe_tokenizer(tokenizer, args.checkpoint, args.revision)
    print(f"  class={tokenizer_contract.tokenizer_class} fast={tokenizer_contract.is_fast}")
    print(
        f"  revision requested={tokenizer_contract.revision_requested} "
        f"observed={tokenizer_contract.revision_observed} "
        f"verified={tokenizer_contract.revision_verified}"
    )
    # A resolved revision that disagrees with the request is a hard stop: the
    # measurements would be attributed to the wrong tokenizer.
    if (
        args.revision
        and tokenizer_contract.revision_observed
        and tokenizer_contract.revision_observed != args.revision
    ):
        print(
            "REFUSING: the tokenizer that loaded is not the one requested.\n"
            f"  requested : {args.revision}\n"
            f"  observed  : {tokenizer_contract.revision_observed}\n"
            f"  evidence  : {', '.join(tokenizer_contract.revision_evidence) or 'n/a'}\n\n"
            "Clear the Hugging Face cache or correct --revision, then rerun.",
            file=sys.stderr,
        )
        return 3
    if args.revision and tokenizer_contract.revision_observed is None:
        print(
            "WARNING: the resolved tokenizer revision could not be determined "
            f"({tokenizer_contract.revision_evidence_source}). "
            "revision_verified=false, so this run is NOT scientifically usable.",
            file=sys.stderr,
        )

    if args.skip_segmenter:
        segmenter, segmenter_contract = None, SegmenterContract(
            available=False, name="VnCoreNLP", notes="skipped by --skip-segmenter"
        )
    else:
        manifest = load_vncorenlp_manifest(manifest_path if vncorenlp_dir else None)
        hashes = load_expected_hashes(hashes_path)
        expected = reconcile_provenance(manifest, hashes, args.vncorenlp_revision)
        if manifest and manifest_path:
            expected = {**expected, "_manifest_path": str(manifest_path)}
        segmenter, segmenter_contract = load_segmenter(
            vncorenlp_dir, expected, args.vncorenlp_revision
        )
    cwd_after_segmenter = Path.cwd()
    print(
        f"  segmenter available: {segmenter_contract.available} "
        f"(pinned={segmenter_contract.pinned}, revision_verified={segmenter_contract.revision_verified}, "
        f"hashes_verified={segmenter_contract.hashes_verified})"
    )
    if segmenter_contract.observed_tags_at_head:
        print(f"  tags at HEAD       : {', '.join(segmenter_contract.observed_tags_at_head)}")
    if cwd_after_segmenter != cwd_at_start:
        print(
            f"  NOTE: a dependency changed the working directory "
            f"({cwd_at_start} -> {cwd_after_segmenter}); output paths were resolved "
            "absolutely beforehand and are unaffected."
        )

    observations = run_probe(tokenizer, segmenter, segmenter_contract)
    by_path: dict[str, list[PathObservation]] = {}
    for observation in observations:
        by_path.setdefault(observation.path.value, []).append(observation)
    comparison = compare_paths(by_path)

    status = STATUS_OK if (segmenter_contract.available and segmenter_contract.pinned) else STATUS_PARTIAL
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / run_id
    suffix = 1
    while run_dir.exists():
        run_dir = output_root / f"{run_id}-{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "run_id": run_dir.name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "tokenizer": tokenizer_contract.to_dict(),
        "segmenter": segmenter_contract.to_dict(),
        "conditions": list(PROBE_CONDITIONS),
        "cases": [case_id for case_id, _ in CASES],
        "seed": SEED,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "status": status,
        "model_weights_loaded": False,
        # scientifically_usable requires EVERY provenance check to have actually
        # passed -- not merely to have been requested:
        #   tokenizer: revision supplied AND resolved AND equal
        #   segmenter: checkout revision verified AND every digest verified
        "tokenizer_revision_requested": tokenizer_contract.revision_requested,
        "tokenizer_revision_observed": tokenizer_contract.revision_observed,
        "tokenizer_revision_verified": tokenizer_contract.revision_verified,
        "scientifically_usable": tokenizer_contract.revision_verified and segmenter_contract.pinned,
        "phobert_provenance": tokenizer_contract.to_dict(),
        "vncorenlp_provenance": segmenter_contract.to_dict(),
        # Path and cwd diagnostics: the first Colab run wrote its artifacts into
        # .vncorenlp/results/b3b0/ because a dependency changed the cwd.
        "repository_root": str(REPO_ROOT),
        "resolved_output_root": str(output_root),
        "resolved_vncorenlp_dir": str(vncorenlp_dir) if vncorenlp_dir else None,
        "cwd_at_start": str(cwd_at_start),
        "cwd_after_segmenter_initialization": str(cwd_after_segmenter),
        "cwd_changed_by_dependency": cwd_at_start != cwd_after_segmenter,
        "note": (
            "Feasibility probe only. Compares preprocessing pathways; does not choose one. "
            "No model weights were loaded and no policy is locked by this run."
        ),
    }
    environment = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "tokenizer": tokenizer_contract.to_dict(),
        "segmenter": segmenter_contract.to_dict(),
        "repository_root": str(REPO_ROOT),
        "resolved_output_root": str(output_root),
        "resolved_vncorenlp_dir": str(vncorenlp_dir) if vncorenlp_dir else None,
        "cwd_at_start": str(cwd_at_start),
        "cwd_after_segmenter_initialization": str(cwd_after_segmenter),
        "cwd_changed_by_dependency": cwd_at_start != cwd_after_segmenter,
        "hf_home": os.environ.get("HF_HOME"),
    }

    for name, payload in (("config.json", config), ("environment.json", environment), ("summary.json", comparison)):
        with (run_dir / name).open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2, default=str)
            fh.write("\n")
    with (run_dir / "cases.jsonl").open("w", encoding="utf-8") as fh:
        for observation in observations:
            fh.write(json.dumps(observation.to_dict(), ensure_ascii=False) + "\n")
    (run_dir / "report.md").write_text(render_report(config, comparison, observations), encoding="utf-8")

    print()
    for name, summary in comparison["path_summaries"].items():
        invariance = comparison["grid_invariance"].get(name, {})
        print(
            f"  {name:28} usable={summary['usable']:4} aligned={summary['aligned']:4} "
            f"offsets={','.join(summary['offset_availability']) or '-':16} "
            f"grid_invariant={invariance.get('all_cases_invariant')}"
        )
    print()
    print(f"  output root : {output_root}")
    if cwd_after_segmenter != cwd_at_start:
        print(f"  cwd changed : {cwd_at_start} -> {cwd_after_segmenter} (artifacts unaffected)")
    print()
    print(f"Status: {status}")
    if not (tokenizer_contract.revision_verified and segmenter_contract.pinned):
        reasons = []
        if not tokenizer_contract.revision_verified:
            reasons.append("tokenizer revision not verified")
        if not segmenter_contract.revision_verified:
            reasons.append("VnCoreNLP checkout revision not verified")
        if not segmenter_contract.hashes_verified:
            reasons.append("VnCoreNLP resource hashes not verified")
        print(f"NOT scientifically usable: {'; '.join(reasons)}.")
    print("No preprocessing policy was chosen. See docs/spec/decisions.md D-B3B0-001.")
    print()
    print(f"Results: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
