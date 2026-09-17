"""Orthographic adapter modules for ViUnMark-Gate and ViUnMark-Scale."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from viunmark.config import (
    GATE_INIT_BIAS,
    GATE_INIT_WEIGHT,
    HIDDEN_SIZE,
    LETTER_TABLE_ROWS,
    SCALE_EPSILON,
    TONE_TABLE_ROWS,
    ViUnMarkContractError,
)

TONE_LABELS = ("SAC", "HUYEN", "HOI", "NGA", "NANG", "UNMARKED", "UNUSED")
LETTER_LABELS = ("NONE", "BREVE", "CIRCUMFLEX", "HORN", "STROKE")
TONE_NA_SENTINEL = -1
LETTER_NA_SENTINEL = -1


@dataclass(frozen=True)
class AdapterShape:
    hidden_size: int = HIDDEN_SIZE
    tone_rows: int = TONE_TABLE_ROWS
    letter_rows: int = LETTER_TABLE_ROWS


def scale_calibrated_fusion(fused: Any, base: Any) -> Any:
    e_norm = base.norm(dim=-1, keepdim=True)
    f_norm = fused.norm(dim=-1, keepdim=True)
    return fused * (e_norm / f_norm.clamp(min=SCALE_EPSILON))


def convex_combination(gate: Any, fused: Any, base: Any) -> Any:
    return gate * fused + (1.0 - gate) * base


class OrthographicAdapter:
    """Torch module wrapper built lazily so importing viunmark stays stdlib-only."""

    def __new__(cls, *, scale_calibrated: bool = False, hidden_size: int = HIDDEN_SIZE):
        import torch
        from torch import nn

        if hidden_size != HIDDEN_SIZE:
            raise ViUnMarkContractError(f"hidden_size must be {HIDDEN_SIZE}")

        class _Adapter(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.hidden_size = hidden_size
                self.scale_calibrated = scale_calibrated
                self.tone_embedding = nn.Embedding(TONE_TABLE_ROWS, hidden_size)
                self.letter_embedding = nn.Embedding(LETTER_TABLE_ROWS, hidden_size)
                self.fusion = nn.Linear(3 * hidden_size, hidden_size)
                self.layer_norm = nn.LayerNorm(hidden_size)
                self.gate = nn.Linear(3 * hidden_size, hidden_size)
                self.reset_gate_parameters()

            def reset_gate_parameters(self) -> None:
                with torch.no_grad():
                    self.gate.weight.fill_(GATE_INIT_WEIGHT)
                    self.gate.bias.fill_(GATE_INIT_BIAS)

            def _embed_channel(self, embedding: Any, ids: Any, mask: Any, rows: int, name: str) -> Any:
                mask = mask.bool()
                if ids.shape != mask.shape:
                    raise ViUnMarkContractError(f"{name} ids and mask shapes differ")
                if mask.any():
                    live = ids[mask]
                    if int(live.min()) < 0 or int(live.max()) >= rows:
                        raise ViUnMarkContractError(f"{name} ids contain unmasked values outside table")
                safe = torch.where(mask, ids, torch.zeros_like(ids))
                return embedding(safe) * mask.unsqueeze(-1).to(dtype=embedding.weight.dtype)

            def tone_channel(self, tone_ids: Any, tone_mask: Any) -> Any:
                return self._embed_channel(self.tone_embedding, tone_ids, tone_mask, TONE_TABLE_ROWS, "tone")

            def letter_channel(self, letter_ids: Any, letter_mask: Any) -> Any:
                mask = letter_mask.bool()
                if letter_ids.shape != mask.shape:
                    raise ViUnMarkContractError("letter ids and mask shapes differ")
                if mask.any():
                    live = letter_ids[mask]
                    if int(live.min()) < 0 or int(live.max()) >= LETTER_TABLE_ROWS:
                        raise ViUnMarkContractError("letter ids contain unmasked values outside table")
                safe = torch.where(mask, letter_ids, torch.zeros_like(letter_ids))
                embedded = self.letter_embedding(safe)
                weights = mask.unsqueeze(-1).to(embedded.dtype)
                numerator = (embedded * weights).sum(dim=-2)
                count = weights.sum(dim=-2)
                return (numerator / count.clamp(min=1.0)) * (count > 0).to(embedded.dtype)

            def forward(self, base_embeddings: Any, tone_ids: Any, tone_mask: Any, letter_ids: Any, letter_mask: Any) -> Any:
                tone = self.tone_channel(tone_ids, tone_mask)
                letter = self.letter_channel(letter_ids, letter_mask)
                q = torch.cat([base_embeddings, tone, letter], dim=-1)
                fused = self.layer_norm(self.fusion(q))
                if self.scale_calibrated:
                    fused = scale_calibrated_fusion(fused, base_embeddings)
                gate = torch.sigmoid(self.gate(q))
                return convex_combination(gate, fused, base_embeddings)

        return _Adapter()


def build_gate_adapter() -> Any:
    return OrthographicAdapter(scale_calibrated=False)


def build_scale_adapter() -> Any:
    return OrthographicAdapter(scale_calibrated=True)
