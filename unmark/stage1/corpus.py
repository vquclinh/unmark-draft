"""Stage-1 corpus preparation: pinned UVW bytes -> verified prepared corpus.

**Torch-free.** `pyarrow` is imported lazily inside the reader, so this module
stays importable in the ML-free local environment and every contract below is
testable against synthetic fixtures.

The pipeline order is load-bearing and enforced by construction (D-S1B-002)::

    1. verify the pin        (filename, byte size, sha256 -- all three)
    2. read + concatenate    (train.parquet -> validation.parquet -> test.parquet)
    3. schema check          (id, content; unique ids; no nulls)
    4. contamination screen  (exact canon() duplicates, opened material only)
    5. document-level split  (exactly 5 000 dev documents)
    6. chunking              (happens AFTER 5, in unmark.stage1.chunking)

Steps 5 and 6 are in that order and cannot be reversed: `partition_documents`
returns a partition keyed by *document* id, and the chunker takes that partition
as an input rather than assigning one. Chunking before splitting would let two
chunks of one article land on opposite sides -- near-duplicate leakage into the
very held-out signal that selects `r` and the learning rate.

**Official UIT-VSFC TEST is unreachable from this module.** There is no
parameter, path or code route to it; the screen accepts only the two openable
sources named in `CONTAMINATION_SCREEN_INPUTS`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from unmark.orthography import canon
from unmark.stage1.contracts import Stage1ContractViolation
from unmark.stage1.protocol import (
    CONTAMINATION_METHOD,
    CONTAMINATION_SCREEN_INPUTS,
    CORPUS_DATASET,
    CORPUS_REVISION,
    CORPUS_SHARD_ORDER,
    DEV_DOCUMENTS,
    REQUIRED_CORPUS_COLUMNS,
    SPLIT_SEED,
    SPLIT_SEED_TAG,
)

CORPUS_SCHEMA_VERSION = "stage1-corpus-v1"
PIN_MANIFEST_PATH = Path(__file__).resolve().parents[2] / "configs" / "data" / "uvw_2026.json"


class CorpusContractViolation(Stage1ContractViolation):
    """Raised when the corpus does not match its pin or its schema. Fail closed."""


# ---------------------------------------------------------------------------
# 1. The pin
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ShardPin:
    name: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class CorpusPin:
    """The committed corpus pin. Every field is verified, none is trusted."""

    dataset: str
    revision: str
    files: tuple[ShardPin, ...]
    concatenation_order: tuple[str, ...]
    schema_version: str

    def __post_init__(self) -> None:
        if len(self.revision) != 40 or not all(c in "0123456789abcdef" for c in self.revision):
            raise CorpusContractViolation(
                f"revision must be a full 40-character sha, got {self.revision!r}. "
                "A moving `main` is not a pin."
            )
        names = tuple(f.name for f in self.files)
        if names != self.concatenation_order:
            raise CorpusContractViolation(
                f"file order {names} does not match concatenation_order "
                f"{self.concatenation_order}; the order is part of the pin"
            )

    def shard(self, name: str) -> ShardPin:
        for f in self.files:
            if f.name == name:
                return f
        raise CorpusContractViolation(f"{name} is not in the pin")


def load_pin(path: Path | None = None) -> CorpusPin:
    """Read the committed pin manifest."""
    path = Path(path) if path is not None else PIN_MANIFEST_PATH
    if not path.is_file():
        raise CorpusContractViolation(f"corpus pin manifest not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    pin = CorpusPin(
        dataset=raw["dataset"],
        revision=raw["revision"],
        files=tuple(ShardPin(f["name"], int(f["bytes"]), f["sha256"]) for f in raw["files"]),
        concatenation_order=tuple(raw["concatenation_order"]),
        schema_version=raw["schema_version"],
    )
    if pin.dataset != CORPUS_DATASET or pin.revision != CORPUS_REVISION:
        raise CorpusContractViolation(
            f"pin manifest ({pin.dataset} @ {pin.revision}) disagrees with the locked "
            f"protocol ({CORPUS_DATASET} @ {CORPUS_REVISION})"
        )
    if pin.concatenation_order != CORPUS_SHARD_ORDER:
        raise CorpusContractViolation(
            f"pin order {pin.concatenation_order} != protocol order {CORPUS_SHARD_ORDER}"
        )
    return pin


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_corpus_root(root: Path, pin: CorpusPin | None = None) -> dict[str, Any]:
    """Fail closed unless **all three** files match name, size **and** digest.

    Verified before a single row is read: reading first and checking later would
    mean a wrong revision had already influenced the run.
    """
    pin = pin or load_pin()
    root = Path(root)
    verified = []
    for name in pin.concatenation_order:
        expected = pin.shard(name)
        path = root / name
        if not path.is_file():
            raise CorpusContractViolation(
                f"missing corpus shard {name} under {root}. All three of "
                f"{list(pin.concatenation_order)} are required."
            )
        size = path.stat().st_size
        if size != expected.bytes:
            raise CorpusContractViolation(
                f"{name}: expected {expected.bytes} bytes, found {size}. This is not "
                f"revision {pin.revision}."
            )
        digest = file_sha256(path)
        if digest != expected.sha256:
            raise CorpusContractViolation(
                f"{name}: sha256 mismatch.\n  expected {expected.sha256}\n  found    {digest}\n"
                "The corpus is not the pinned revision; refusing to proceed."
            )
        verified.append({"name": name, "bytes": size, "sha256": digest})
    return {
        "dataset": pin.dataset,
        "revision": pin.revision,
        "concatenation_order": list(pin.concatenation_order),
        "files": verified,
        "shard_labels_are_a_split": False,
    }


# ---------------------------------------------------------------------------
# 2-3. Read, concatenate in the fixed order, check the schema
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CorpusDocument:
    """One source document. `source_shard` is provenance, never a partition."""

    document_id: str
    content: str
    source_shard: str
    source_row: int


def read_shard(path: Path, shard_name: str) -> list[CorpusDocument]:
    """Read one parquet shard. `pyarrow` is imported lazily."""
    import pyarrow.parquet as pq  # noqa: PLC0415 - lazy by design

    table = pq.read_table(path, columns=list(REQUIRED_CORPUS_COLUMNS))
    return _documents_from_columns(
        table.column("id").to_pylist(), table.column("content").to_pylist(), shard_name
    )


def _documents_from_columns(
    ids: Sequence[Any], contents: Sequence[Any], shard_name: str
) -> list[CorpusDocument]:
    """Shared by the real reader and the synthetic fixtures. Fails closed."""
    if len(ids) != len(contents):
        raise CorpusContractViolation(
            f"{shard_name}: id/content length mismatch ({len(ids)} vs {len(contents)})"
        )
    out: list[CorpusDocument] = []
    for row, (doc_id, content) in enumerate(zip(ids, contents)):
        if doc_id is None or (isinstance(doc_id, str) and not doc_id.strip()):
            raise CorpusContractViolation(
                f"{shard_name} row {row}: null/empty document id. A stable identity is "
                "required -- row order is not an identity, and corruption is keyed on it."
            )
        if content is None or not isinstance(content, str) or not content.strip():
            raise CorpusContractViolation(
                f"{shard_name} row {row} (id {doc_id!r}): null/empty content"
            )
        out.append(CorpusDocument(str(doc_id), content, shard_name, row))
    return out


def check_schema(columns: Iterable[str], shard_name: str) -> None:
    """The two columns scientific correctness depends on. Optional metadata is
    neither required nor used."""
    present = set(columns)
    missing = [c for c in REQUIRED_CORPUS_COLUMNS if c not in present]
    if missing:
        raise CorpusContractViolation(
            f"{shard_name}: missing required column(s) {missing}; found {sorted(present)}"
        )


def concatenate(shards: dict[str, list[CorpusDocument]]) -> list[CorpusDocument]:
    """Concatenate in the locked order and refuse duplicate ids.

    Duplicates **fail** and are reported. Silently renaming or de-duplicating
    them would change document identity, and `sample_id` keys the corruption
    draw -- so a rename is a different corruption stream.
    """
    missing = [n for n in CORPUS_SHARD_ORDER if n not in shards]
    if missing:
        raise CorpusContractViolation(f"missing shard(s) {missing}; all three are required")
    documents: list[CorpusDocument] = []
    for name in CORPUS_SHARD_ORDER:
        documents.extend(shards[name])
    seen: dict[str, CorpusDocument] = {}
    duplicates: list[tuple[str, str, str]] = []
    for doc in documents:
        if doc.document_id in seen:
            first = seen[doc.document_id]
            duplicates.append((doc.document_id, first.source_shard, doc.source_shard))
        else:
            seen[doc.document_id] = doc
    if duplicates:
        sample = duplicates[:10]
        raise CorpusContractViolation(
            f"{len(duplicates)} duplicate document id(s) across the concatenated shards; "
            f"first {len(sample)}: {sample}. Refusing to rename or de-duplicate: document "
            "identity keys the corruption stream."
        )
    return documents


# ---------------------------------------------------------------------------
# 4. Contamination screen -- exact/canonical only, opened material only
# ---------------------------------------------------------------------------
def canonical_digest(text: str) -> str:
    """sha256 of `canon(text)`. The single comparison key for the screen."""
    return hashlib.sha256(canon(text).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ContaminationReport:
    """What the screen found. **Carries no UIT-VSFC text**, only digests."""

    screened_against: tuple[str, ...]
    reference_digest_count: int
    excluded_document_ids: tuple[str, ...]
    excluded_digests: tuple[str, ...]
    method: str = CONTAMINATION_METHOD

    @property
    def excluded_count(self) -> int:
        return len(self.excluded_document_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "screened_against": list(self.screened_against),
            "reference_digest_count": self.reference_digest_count,
            "excluded_count": self.excluded_count,
            "excluded_document_ids": list(self.excluded_document_ids),
            "excluded_digests": list(self.excluded_digests),
            "official_test_screened": False,
            "claim": (
                "no EXACT canonical overlap against the UIT-VSFC material already "
                "legitimately opened by the pre-G1 protocol. This is NOT a claim of "
                "zero overlap with official TEST, which remains SEALED and unopened."
            ),
        }


def screen_contamination(
    documents: Sequence[CorpusDocument], reference_texts: dict[str, Sequence[str]]
) -> tuple[list[CorpusDocument], ContaminationReport]:
    """Exclude UVW documents whose `canon()` equals an opened UIT-VSFC text.

    Args:
        reference_texts: keyed by source name. Only the names in
            `CONTAMINATION_SCREEN_INPUTS` are accepted -- any other key, and in
            particular anything naming official TEST, raises.
    """
    unknown = sorted(set(reference_texts) - set(CONTAMINATION_SCREEN_INPUTS))
    if unknown:
        raise CorpusContractViolation(
            f"contamination screen refuses source(s) {unknown}. Only "
            f"{list(CONTAMINATION_SCREEN_INPUTS)} may be read -- both already opened by "
            "the pre-G1 protocol. Official UIT-VSFC TEST is SEALED and there is no "
            "route to it."
        )
    reference: set[str] = set()
    for texts in reference_texts.values():
        reference.update(canonical_digest(t) for t in texts)

    kept: list[CorpusDocument] = []
    excluded_ids: list[str] = []
    excluded_digests: list[str] = []
    for doc in documents:
        digest = canonical_digest(doc.content)
        if digest in reference:
            excluded_ids.append(doc.document_id)
            excluded_digests.append(digest)
        else:
            kept.append(doc)
    return kept, ContaminationReport(
        screened_against=tuple(sorted(reference_texts)),
        reference_digest_count=len(reference),
        excluded_document_ids=tuple(excluded_ids),
        excluded_digests=tuple(excluded_digests),
    )


# ---------------------------------------------------------------------------
# 5. Document-level partition -- BEFORE any chunking
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DocumentPartition:
    """Which documents are train and which are dev. Keyed by document id."""

    train: tuple[str, ...]
    dev: tuple[str, ...]
    seed: int
    seed_tag: str = SPLIT_SEED_TAG

    def __post_init__(self) -> None:
        overlap = set(self.train) & set(self.dev)
        if overlap:
            raise CorpusContractViolation(
                f"{len(overlap)} document(s) in both partitions: {sorted(overlap)[:5]}"
            )

    @property
    def assignment(self) -> dict[str, str]:
        return {**{d: "train" for d in self.train}, **{d: "dev" for d in self.dev}}

    @property
    def membership_digest(self) -> str:
        """Order-insensitive digest of the assignment."""
        payload = "\n".join(f"{d}\t{p}" for d, p in sorted(self.assignment.items()))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed_tag": self.seed_tag,
            "seed": self.seed,
            "train_documents": len(self.train),
            "dev_documents": len(self.dev),
            "membership_digest": self.membership_digest,
        }


def _rank_key(document_id: str, seed: int) -> tuple[str, str]:
    """Stable hash rank for one document. No global RNG anywhere.

    The document id is the tie-break, so the ordering is a total order on
    *content identity* and cannot depend on the order documents were iterated.
    """
    payload = f"{SPLIT_SEED_TAG}|{seed}|{document_id}".encode("utf-8")
    return (hashlib.blake2b(payload, digest_size=16).hexdigest(), document_id)


def partition_documents(
    document_ids: Sequence[str], *, dev_documents: int = DEV_DOCUMENTS, seed: int = SPLIT_SEED
) -> DocumentPartition:
    """Exactly `dev_documents` documents to dev, the rest to train.

    Deterministic stable hash-ranking rather than `stratified_group_split`: that
    audited helper is **fraction**-based and **label**-stratified, and the
    Stage-1 corpus is unlabeled and needs an exact count. Reusing it would have
    meant passing a fake label and a fraction that only approximates 5 000.

    Order-independent by construction: the rank of a document depends on its id
    and the seed alone, so shuffling the input permutes nothing.
    """
    unique = list(dict.fromkeys(document_ids))
    if len(unique) != len(document_ids):
        raise CorpusContractViolation("duplicate document ids supplied to the partition")
    if dev_documents < 0:
        raise CorpusContractViolation(f"dev_documents must be non-negative, got {dev_documents}")
    if len(unique) <= dev_documents:
        raise CorpusContractViolation(
            f"corpus has {len(unique)} eligible documents; cannot take {dev_documents} "
            "for dev and leave a training set"
        )
    ordered = sorted(unique, key=lambda d: _rank_key(d, seed))
    dev = tuple(sorted(ordered[:dev_documents]))
    train = tuple(sorted(ordered[dev_documents:]))
    return DocumentPartition(train=train, dev=dev, seed=seed)
