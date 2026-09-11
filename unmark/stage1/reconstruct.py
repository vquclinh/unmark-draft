"""Rebuild a Stage-1 adapter from the provenance its checkpoint records.

**Imports torch lazily.** Nothing here trains, evaluates, selects or scores.

Why this module exists. A Stage-1 checkpoint stores the ADAPTER's own
`state_dict` -- eight tensors, the same names and the same shapes for every
candidate, because a candidate fusion adds no parameter. That is what makes the
comparison clean, and it is also what makes a silent mistake possible: a C1
adapter's weights load into the historical mixture rule with `strict=True` and no
error, producing a model that is numerically wrong and structurally plausible.

The architecture therefore has to come from the checkpoint's recorded identity,
not from whichever constructor the caller happened to reach for:

    provenance.fusion  ->  AdapterConfig(fusion_id=...)  ->  OrthographyInputAdapter

`adapter_for_provenance` is that path, and it is the API a later Stage-2 loader
calls. Stage-2 itself is NOT implemented here and no downstream experiment is
run, scored or planned by this module -- it exists so that the C1 checkpoints
written now remain correctly reconstructable later.
"""

from __future__ import annotations

from typing import Any, Mapping

from unmark.modeling.contracts import FUSION_IDS, HISTORICAL_FUSION_ID
from unmark.stage1.candidates import candidate_for_identity
from unmark.stage1.contracts import Stage1ContractViolation

ADAPTER_STATE_KEY = "adapter_state"
PROVENANCE_KEY = "provenance"


def recorded_fusion_id(provenance: Mapping[str, Any]) -> str:
    """The fusion a recorded provenance describes. **Fails closed.**

    A provenance written before the field existed carries no `fusion` block, and
    that absence is unambiguous: only the historical adapter could have produced
    it, because every architecture added since records its own identity. It is
    therefore READ as `historical-fusion-v1` -- which is what keeps the frozen
    UNMARK-A/B checkpoints loadable -- and never as "unknown, assume the caller
    knows best".
    """
    if not isinstance(provenance, Mapping):
        raise Stage1ContractViolation(
            "checkpoint carries no provenance mapping, so the architecture it was "
            "trained under cannot be determined. It is refused rather than assumed."
        )
    block = provenance.get("fusion")
    if block is None:
        return HISTORICAL_FUSION_ID
    if not isinstance(block, Mapping) or "fusion_id" not in block:
        raise Stage1ContractViolation(
            f"provenance carries a malformed fusion block {block!r}; it was not "
            "written by this repository's serializer and cannot be trusted to "
            "describe the architecture it was trained under."
        )
    fusion_id = block["fusion_id"]
    if fusion_id not in FUSION_IDS:
        raise Stage1ContractViolation(
            f"provenance records fusion {fusion_id!r}, which this repository cannot "
            f"build; the closed set is {list(FUSION_IDS)}. Reconstructing it as some "
            "other architecture would silently change the model."
        )
    return fusion_id


def recorded_identity(payload: Mapping[str, Any]) -> tuple[str, str]:
    """`(objective_id, fusion_id)` of a checkpoint payload. Fails closed."""
    provenance = payload.get(PROVENANCE_KEY)
    if not isinstance(provenance, Mapping):
        raise Stage1ContractViolation(
            "checkpoint carries no provenance mapping; its identity cannot be read"
        )
    objective = provenance.get("objective") or {}
    objective_id = objective.get("objective_id") if isinstance(objective, Mapping) else None
    if objective_id is None:
        # Same rule as the fusion: absence is the historical claim, because every
        # objective added since the field existed records its own identity.
        from unmark.stage1.protocol import HISTORICAL_OBJECTIVE_ID

        objective_id = HISTORICAL_OBJECTIVE_ID
    return (str(objective_id), recorded_fusion_id(provenance))


def adapter_for_provenance(
    provenance: Mapping[str, Any], hidden_size: int, *, init_seed: int = 0
) -> Any:
    """An adapter built for the architecture `provenance` records. **Lazy torch.**

    The weights are meant to be overwritten by `load_adapter_state`, so the
    initialisation seed is irrelevant to the result and defaults to `0`; it is
    exposed only so a caller that wants a deterministic pre-load state can ask
    for one.
    """
    from unmark.stage1.initialisation import fresh_adapter

    return fresh_adapter(hidden_size, init_seed, recorded_fusion_id(provenance))


def load_adapter_state(adapter: Any, payload: Mapping[str, Any]) -> Any:
    """Load a checkpoint's adapter tensors **strictly**. Returns the adapter.

    `strict=True`, always: a key mismatch is a different architecture, and the
    v1 checkpoint defect (Audit 030 §AC.9) was precisely a `strict=False` load
    that restored nothing and trained on from fresh weights.
    """
    if ADAPTER_STATE_KEY not in payload:
        raise Stage1ContractViolation(
            f"checkpoint has no {ADAPTER_STATE_KEY!r}; there is nothing to load"
        )
    adapter.load_state_dict(payload[ADAPTER_STATE_KEY], strict=True)
    return adapter


def reconstruct_adapter(payload: Mapping[str, Any], hidden_size: int) -> Any:
    """**THE** candidate-checkpoint construction API. Build, then load strictly.

    One call, so a later Stage-2 loader cannot accidentally pair one candidate's
    architecture with another's weights:

        payload -> recorded fusion -> matching adapter -> strict state load

    Returns the adapter together with nothing else: this module deliberately
    knows nothing about encoders, objectives, task heads or downstream data.
    """
    provenance = payload.get(PROVENANCE_KEY)
    if not isinstance(provenance, Mapping):
        raise Stage1ContractViolation(
            "checkpoint carries no provenance mapping, so the architecture it was "
            "trained under cannot be determined"
        )
    adapter = adapter_for_provenance(provenance, hidden_size)
    return load_adapter_state(adapter, payload)


def require_loadable_as(payload: Mapping[str, Any], fusion_id: str) -> None:
    """Refuse unless this checkpoint really is the named architecture.

    The guard a loader that expects ONE architecture calls before touching a
    payload. Without it, a C1 checkpoint loads into the historical adapter
    silently -- same eight keys, same shapes, different equation -- and the error
    surfaces only as a downstream number nobody can explain.
    """
    recorded = recorded_fusion_id(payload.get(PROVENANCE_KEY, {}))
    if recorded != fusion_id:
        raise Stage1ContractViolation(
            f"this checkpoint records fusion {recorded!r}, but the loader expects "
            f"{fusion_id!r}. The adapter tensors are shape-compatible across fusions, "
            "so loading it anyway would produce a model that is wrong and looks fine."
        )


def candidate_for_payload(payload: Mapping[str, Any]) -> Any:
    """The registered `StageCandidate` a checkpoint belongs to. Fails closed."""
    objective_id, fusion_id = recorded_identity(payload)
    return candidate_for_identity(objective_id, fusion_id)
