"""Stable per-unit corruption decisions.

Proposal v1.3 section 5.3 requires corruption to be a deterministic function::

    C(x, p, s) -> x̃

"of example, rate, and seed: the same triple must always produce the same
corrupted string. If corruption is nondeterministic, `RESTORE`, `ALIGN`, and
UNMARK are silently evaluated on different noise, and the whole results table
becomes meaningless without any error being raised."

The algorithm
-------------
Each orthographic unit gets an independent score in [0, 1), derived only from
values that are stable across processes and machines::

    payload = schema_version | seed | sample_id | text_identity | unit_index
    digest  = BLAKE2b(payload, digest_size=8)
    score   = int.from_bytes(digest, "big") / 2**64
    selected = score < probability

where `text_identity` is the SHA-256 hex digest of the *canonical clean text*,
so two spellings that canonicalise to the same string get identical decisions.

Why it is built this way, point by point:

* **No `random` module, no global RNG.** Nothing here reads or mutates process
  state, so corruption cannot depend on what ran before it.
* **No Python `hash()`.** It is randomised per process by PYTHONHASHSEED, so a
  corrupted corpus would not reproduce tomorrow.
* **Per-unit, not sequential.** A sequential RNG would make unit *k*'s decision
  depend on units 0..k-1, so inserting one syllable would change every later
  decision. Here each unit is independent of its neighbours: the score for unit
  7 is the same whatever happens at unit 6.
* **Keyed by `sample_id`, not row order.** Reordering a dataset must not change
  any sample's corruption. The proposal says "example"; `sample_id` is what
  makes that identity explicit (D-B2-001).
* **`schema_version` is in the payload**, so a future change to the algorithm
  produces different, non-silently-comparable decisions.

`score < probability` gives the intended endpoints exactly: `p = 0.0` selects
nothing (no score is < 0) and `p = 1.0` selects everything (every score is
< 1.0, since the largest possible value is (2**64 - 1) / 2**64).
"""

from __future__ import annotations

import hashlib

CORRUPTION_SCHEMA_VERSION = "b2-v1"

# Byte that cannot occur in the textual fields, so the payload is unambiguous:
# ("a", "bc") and ("ab", "c") must not produce the same key.
_FIELD_SEPARATOR = "\x1f"

_DIGEST_BYTES = 8
_SCORE_DENOMINATOR = float(1 << (8 * _DIGEST_BYTES))


def text_identity(canonical_text: str) -> str:
    """Stable identity of a canonical clean string: its SHA-256 hex digest.

    Hashed rather than embedded so the payload stays bounded for long inputs.
    SHA-256 is used from the standard library; nothing here is security-
    sensitive, the requirement is only cross-process stability.
    """
    return hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()


def unit_score(
    *,
    seed: int,
    sample_id: str,
    identity: str,
    unit_index: int,
    schema_version: str = CORRUPTION_SCHEMA_VERSION,
) -> float:
    """Deterministic score in [0, 1) for one orthographic unit."""
    payload = _FIELD_SEPARATOR.join(
        (schema_version, str(seed), str(sample_id), identity, str(unit_index))
    )
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=_DIGEST_BYTES).digest()
    return int.from_bytes(digest, "big") / _SCORE_DENOMINATOR


def is_selected(
    *,
    probability: float,
    seed: int,
    sample_id: str,
    identity: str,
    unit_index: int,
    schema_version: str = CORRUPTION_SCHEMA_VERSION,
) -> tuple[bool, float]:
    """Whether a unit is corrupted, and the score that decided it."""
    score = unit_score(
        seed=seed,
        sample_id=sample_id,
        identity=identity,
        unit_index=unit_index,
        schema_version=schema_version,
    )
    return score < probability, score
