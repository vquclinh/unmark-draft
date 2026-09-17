"""Corruption protocol and cross-pathway input identity. **Torch-free.**

Two rules govern the inputs every branch sees.

**One scientific corruption seed, stated per protocol.** `CorruptionProtocol`
has no default seed. A dataset's protocol must freeze its own seed, or
explicitly freeze the reuse of an earlier one; nothing inherits a seed silently.
`FULL` is the no-corruption condition. The corruption API still takes an integer
for it, and `FULL_CONDITION_API_SEED` is that placeholder. It is not a
scientific seed and cannot change any output.

**Cross-pathway input identity.** ViUnMark-Gate and ViUnMark-Scale receive the
SAME corrupted realization for the same dataset, role, sample, condition and
scientific seed. Corruption is drawn once and shared, never redrawn per
pathway. `require_cross_pathway_input_identity` refuses realizations that break
this.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from unmark.viunmark.config import SIX_CONDITIONS, Pathway, ViUnMarkContractError

FULL_CONDITION = "FULL"
FULL_CONDITION_API_SEED = 0
"""API placeholder for `FULL`. **Not a scientific corruption seed**: FULL removes
nothing, so the value has no effect on any output."""
FULL_CONDITION_API_SEED_IS_SCIENTIFIC = False

CROSS_PATHWAY_INPUT_IDENTITY_REQUIRED = True
BASE_GRID_INVARIANCE_REQUIRED = True
ADAPTED_PATHWAYS: tuple[Pathway, ...] = (Pathway.VIUNMARK_GATE, Pathway.VIUNMARK_SCALE)


def _seed(value: object, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ViUnMarkContractError(f"{what} must be an int, got {value!r}")
    return value


@dataclass(frozen=True)
class CorruptionProtocol:
    """The corruption identity of one evaluation protocol.

    `scientific_corruption_seed` is required and has no default.
    """

    scientific_corruption_seed: int
    conditions: tuple[str, ...] = SIX_CONDITIONS

    def __post_init__(self) -> None:
        _seed(self.scientific_corruption_seed, "scientific_corruption_seed")
        if tuple(self.conditions) != SIX_CONDITIONS:
            raise ViUnMarkContractError(
                f"ViUnMark uses exactly {list(SIX_CONDITIONS)} in that order, got "
                f"{list(self.conditions)}"
            )

    def api_seed_for(self, condition: str) -> int:
        """The integer passed to the corruption API for `condition`."""
        if condition not in SIX_CONDITIONS:
            raise ViUnMarkContractError(f"unknown condition {condition!r}")
        if condition == FULL_CONDITION:
            return FULL_CONDITION_API_SEED
        return self.scientific_corruption_seed


@dataclass(frozen=True)
class CorruptionRealizationKey:
    """What one corrupted text realization is the realization OF."""

    dataset: str
    role: str
    sample_id: str
    condition: str
    scientific_corruption_seed: int

    def __post_init__(self) -> None:
        for name in ("dataset", "role", "sample_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ViUnMarkContractError(f"{name} must be a non-empty string")
        if self.condition not in SIX_CONDITIONS:
            raise ViUnMarkContractError(f"unknown condition {self.condition!r}")
        _seed(self.scientific_corruption_seed, "scientific_corruption_seed")


def require_cross_pathway_input_identity(
    realizations: Mapping[Pathway, Mapping[CorruptionRealizationKey, str]],
) -> None:
    """Refuse unless every supplied pathway saw the same corrupted text per key.

    ViUnMark-Gate and ViUnMark-Scale must both be supplied. They must cover the
    same keys, and for every key their corrupted texts must be identical.
    """
    missing = [p.value for p in ADAPTED_PATHWAYS if p not in realizations]
    if missing:
        raise ViUnMarkContractError(
            f"cross-pathway input identity needs realizations for {missing}"
        )
    pathways = list(realizations)
    reference_pathway = pathways[0]
    reference = realizations[reference_pathway]
    if not reference:
        raise ViUnMarkContractError("no realizations supplied")
    for pathway in pathways[1:]:
        other = realizations[pathway]
        if set(other) != set(reference):
            raise ViUnMarkContractError(
                f"{pathway.value} and {reference_pathway.value} cover different "
                "dataset/role/sample/condition/seed keys"
            )
        for key, text in reference.items():
            if other[key] != text:
                raise ViUnMarkContractError(
                    f"{pathway.value} received a different corrupted text than "
                    f"{reference_pathway.value} for sample {key.sample_id!r} under "
                    f"{key.condition}: corruption must be drawn once and shared"
                )


__all__ = [
    "ADAPTED_PATHWAYS",
    "BASE_GRID_INVARIANCE_REQUIRED",
    "CROSS_PATHWAY_INPUT_IDENTITY_REQUIRED",
    "CorruptionProtocol",
    "CorruptionRealizationKey",
    "FULL_CONDITION_API_SEED",
    "FULL_CONDITION_API_SEED_IS_SCIENTIFIC",
    "require_cross_pathway_input_identity",
]
