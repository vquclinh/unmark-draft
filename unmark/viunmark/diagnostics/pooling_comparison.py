"""Matched-Head Pooling Comparison: the plan contract. **Torch-free.**

A controlled learned-head comparison of FIRST_TOKEN against MASKED_MEAN is only
a pooling comparison if everything else is held fixed. `MatchedPoolingComparisonPlan`
refuses a plan that breaks any of these:

* both heads read the same pathway;
* both use exactly the same seeds, in the same order, so seeds pair up;
* for each paired seed, both heads start from identical parameters (the two
  features have the same width, so this is always achievable);
* both use one identical training schedule;
* checkpoints are selected within each head only;
* no seed is selected over another.

This module validates a plan. It trains nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

from unmark.viunmark.config import Pathway, ReadoutKind, ViUnMarkContractError

CHECKPOINT_SELECTION_WITHIN_EACH_HEAD = "within_each_head"


@dataclass(frozen=True)
class MatchedPoolingComparisonPlan:
    pathway: Pathway
    seeds_by_readout: Mapping[ReadoutKind, tuple[int, ...]]
    schedule_by_readout: Mapping[ReadoutKind, Mapping[str, Any]]
    initial_state_digest_by_readout: Mapping[ReadoutKind, Mapping[int, str]]
    best_seed_selection: bool = False
    checkpoint_selection_scope: str = CHECKPOINT_SELECTION_WITHIN_EACH_HEAD

    COMPARED: ClassVar[tuple[ReadoutKind, ReadoutKind]] = (
        ReadoutKind.FIRST_TOKEN,
        ReadoutKind.MASKED_MEAN,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.pathway, Pathway):
            raise ViUnMarkContractError("pathway must be a Pathway")
        compared = set(self.COMPARED)
        for name in ("seeds_by_readout", "schedule_by_readout", "initial_state_digest_by_readout"):
            keys = set(getattr(self, name))
            if keys != compared:
                raise ViUnMarkContractError(
                    f"{name} must cover exactly FIRST_TOKEN and MASKED_MEAN, got "
                    f"{sorted(k.value for k in keys)}"
                )
        first, mean = self.COMPARED

        seeds = self.seeds_by_readout[first]
        if not isinstance(seeds, tuple) or not seeds or len(set(seeds)) != len(seeds):
            raise ViUnMarkContractError("seeds must be a non-empty tuple of unique seeds")
        if self.seeds_by_readout[mean] != seeds:
            raise ViUnMarkContractError(
                "both readouts must use the same seeds in the same order, so each seed pairs "
                f"one FIRST_TOKEN head with one MASKED_MEAN head: "
                f"{seeds} vs {self.seeds_by_readout[mean]}"
            )

        if dict(self.schedule_by_readout[first]) != dict(self.schedule_by_readout[mean]):
            raise ViUnMarkContractError("both readouts must use one identical training schedule")

        for seed in seeds:
            digests = [self.initial_state_digest_by_readout[kind].get(seed) for kind in self.COMPARED]
            if any(not isinstance(d, str) or not d for d in digests):
                raise ViUnMarkContractError(f"seed {seed} lacks an initial-state digest")
            if digests[0] != digests[1]:
                raise ViUnMarkContractError(
                    f"seed {seed}: the paired heads start from different parameters"
                )

        if self.best_seed_selection is not False:
            raise ViUnMarkContractError("a matched pooling comparison selects no best seed")
        if self.checkpoint_selection_scope != CHECKPOINT_SELECTION_WITHIN_EACH_HEAD:
            raise ViUnMarkContractError(
                "checkpoint selection happens within each head only, got "
                f"{self.checkpoint_selection_scope!r}"
            )

    @property
    def paired_seeds(self) -> tuple[int, ...]:
        return self.seeds_by_readout[ReadoutKind.FIRST_TOKEN]


__all__ = ["CHECKPOINT_SELECTION_WITHIN_EACH_HEAD", "MatchedPoolingComparisonPlan"]
