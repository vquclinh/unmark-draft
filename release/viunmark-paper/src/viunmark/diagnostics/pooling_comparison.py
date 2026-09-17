"""Matched-head pooling-comparison plan validator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from viunmark.config import Pathway, ReadoutKind, ViUnMarkContractError

CHECKPOINT_SELECTION_WITHIN_EACH_HEAD = "within_each_head"


@dataclass(frozen=True)
class MatchedPoolingComparisonPlan:
    pathway: Pathway
    seeds_by_readout: Mapping[ReadoutKind, tuple[int, ...]]
    schedule_by_readout: Mapping[ReadoutKind, Mapping[str, Any]]
    initial_state_digest_by_readout: Mapping[ReadoutKind, Mapping[int, str]]
    best_seed_selection: bool = False
    checkpoint_selection_scope: str = CHECKPOINT_SELECTION_WITHIN_EACH_HEAD

    COMPARED: ClassVar[tuple[ReadoutKind, ReadoutKind]] = (ReadoutKind.FIRST_TOKEN, ReadoutKind.MASKED_MEAN)

    def __post_init__(self) -> None:
        if not isinstance(self.pathway, Pathway):
            raise ViUnMarkContractError("pathway must be a Pathway")
        compared = set(self.COMPARED)
        for name in ("seeds_by_readout", "schedule_by_readout", "initial_state_digest_by_readout"):
            if set(getattr(self, name)) != compared:
                raise ViUnMarkContractError(f"{name} must cover FIRST_TOKEN and MASKED_MEAN")
        first, mean = self.COMPARED
        seeds = self.seeds_by_readout[first]
        if not isinstance(seeds, tuple) or not seeds or len(set(seeds)) != len(seeds):
            raise ViUnMarkContractError("seeds must be a non-empty unique tuple")
        if self.seeds_by_readout[mean] != seeds:
            raise ViUnMarkContractError("both readouts must use the same seeds in the same order")
        if dict(self.schedule_by_readout[first]) != dict(self.schedule_by_readout[mean]):
            raise ViUnMarkContractError("both readouts must use one identical training schedule")
        for seed in seeds:
            digests = [self.initial_state_digest_by_readout[k].get(seed) for k in self.COMPARED]
            if any(not isinstance(d, str) or not d for d in digests) or digests[0] != digests[1]:
                raise ViUnMarkContractError("paired heads must start from identical parameters")
        if self.best_seed_selection is not False:
            raise ViUnMarkContractError("a matched pooling comparison selects no best seed")
        if self.checkpoint_selection_scope != CHECKPOINT_SELECTION_WITHIN_EACH_HEAD:
            raise ViUnMarkContractError("checkpoint selection happens within each head")

    @property
    def paired_seeds(self) -> tuple[int, ...]:
        return self.seeds_by_readout[ReadoutKind.FIRST_TOKEN]
