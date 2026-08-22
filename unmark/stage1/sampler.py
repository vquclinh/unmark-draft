"""Deterministic training order with exact mid-pass resume. **Torch-free.**

Load-bearing because the corruption draw is keyed on `(seed, sample_id, visit)`:
if a resume restarted the pass or advanced `visit` early, the corruption stream
would silently change and the resumed run would not be the run it claims to
continue.

The order is a **stable hash ranking** per pass, not a shuffled list and not a
global RNG. Two consequences that matter:

* the order for pass `v` is reconstructible from `(seed, visit=v)` alone, so a
  checkpoint stores a cursor rather than an opaque RNG blob;
* nothing about the order depends on Python's hash randomisation or on the order
  chunks were read from disk.

Within one pass every training chunk is consumed exactly once, and `visit`
increments only at the pass boundary.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Sequence

from unmark.stage1.contracts import Stage1ContractViolation

SAMPLER_SCHEMA_VERSION = "stage1-sampler-v1"
ORDER_NAMESPACE = "stage1-order"
"""Domain-separated from the corruption namespaces, so the training order and
the corruption draws cannot share a stream."""


class SamplerStateViolation(Stage1ContractViolation):
    """Raised when sampler state is incomplete or inconsistent with its corpus."""


def pass_order(chunk_ids: Sequence[str], seed: int, visit: int) -> list[str]:
    """The consumption order for one pass. Deterministic and reconstructible."""

    def rank(chunk_id: str) -> tuple[str, str]:
        payload = f"{ORDER_NAMESPACE}|{seed}|{visit}|{chunk_id}".encode("utf-8")
        return (hashlib.blake2b(payload, digest_size=16).hexdigest(), chunk_id)

    return sorted(chunk_ids, key=rank)


@dataclass
class DeterministicSampler:
    """Cursor over an infinite sequence of passes. Mutable by design.

    State is `(visit, position)`, both small integers -- that is the whole
    resume payload, because `pass_order` is a pure function of
    `(chunk_ids, seed, visit)`.
    """

    chunk_ids: tuple[str, ...]
    seed: int
    visit: int = 0
    position: int = 0

    def __post_init__(self) -> None:
        if not self.chunk_ids:
            raise SamplerStateViolation("sampler needs at least one chunk")
        if len(set(self.chunk_ids)) != len(self.chunk_ids):
            raise SamplerStateViolation("duplicate chunk ids in the sampler")
        if self.visit < 0 or self.position < 0:
            raise SamplerStateViolation(f"invalid cursor visit={self.visit} pos={self.position}")
        if self.position >= len(self.chunk_ids):
            raise SamplerStateViolation(
                f"position {self.position} is past the end of a pass of "
                f"{len(self.chunk_ids)} chunks; the pass boundary advances `visit`"
            )
        self._order = pass_order(self.chunk_ids, self.seed, self.visit)

    @property
    def corpus_digest(self) -> str:
        """Binds sampler state to the exact chunk set it was built for."""
        payload = "\n".join(sorted(self.chunk_ids)).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def next_batch(self, batch_size: int) -> list[tuple[str, int]]:
        """`(chunk_id, visit)` pairs. The `visit` travels with each example.

        Each example carries the `visit` it was drawn under, so a batch that
        straddles a pass boundary corrupts each half under its own pass -- the
        alternative would silently re-serve one pass's corruption for the other.
        """
        if batch_size <= 0:
            raise SamplerStateViolation(f"batch_size must be positive, got {batch_size}")
        out: list[tuple[str, int]] = []
        while len(out) < batch_size:
            remaining = len(self._order) - self.position
            take = min(batch_size - len(out), remaining)
            out.extend((cid, self.visit) for cid in self._order[self.position : self.position + take])
            self.position += take
            if self.position >= len(self._order):
                # pass boundary: every chunk consumed exactly once, THEN advance
                self.visit += 1
                self.position = 0
                self._order = pass_order(self.chunk_ids, self.seed, self.visit)
        return out

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SAMPLER_SCHEMA_VERSION,
            "seed": self.seed,
            "visit": self.visit,
            "position": self.position,
            "corpus_digest": self.corpus_digest,
            "chunk_count": len(self.chunk_ids),
        }

    @classmethod
    def from_state(cls, chunk_ids: Sequence[str], state: dict[str, Any]) -> DeterministicSampler:
        """Rebuild an exact cursor. Fails closed on any provenance mismatch."""
        required = {"schema_version", "seed", "visit", "position", "corpus_digest"}
        missing = sorted(required - set(state))
        if missing:
            raise SamplerStateViolation(f"sampler state is missing {missing}")
        if state["schema_version"] != SAMPLER_SCHEMA_VERSION:
            raise SamplerStateViolation(
                f"sampler state schema {state['schema_version']!r} != {SAMPLER_SCHEMA_VERSION!r}"
            )
        sampler = cls(
            chunk_ids=tuple(chunk_ids),
            seed=int(state["seed"]),
            visit=int(state["visit"]),
            position=int(state["position"]),
        )
        if sampler.corpus_digest != state["corpus_digest"]:
            raise SamplerStateViolation(
                "sampler state was produced for a different chunk set; resuming would "
                "change both the training order and the corruption stream"
            )
        return sampler
