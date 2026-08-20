#!/usr/bin/env python3
"""B4B — real-PhoBERT adapter integration probe (Colab only).

Loads the real pinned checkpoint **weights** and validates the B4B adapter
against them. **Do not run this locally**: the local `.venv` is deliberately
ML-free.

**Nothing is trained.** There is no optimizer, no `optimizer.step()`, no
parameter update, no dataset and no checkpoint saving. A single forward + scalar
diagnostic + backward runs solely to validate gradient routing. The diagnostic
scalar is not a scientific objective; Stage-1's cosine loss belongs to a later
phase.

What it answers, in order:

1. **Provenance.** Requested vs resolved revision for both tokenizer and model.
2. **The position-id question left open by Audit 014.** PhoBERT is
   RoBERTa-family, whose position ids are derived from `input_ids` through a
   padding-aware offset; passing only `inputs_embeds` takes a different code
   path. This probe **instruments the real position-embedding module** and
   reports the actual index tensors, rather than inferring them from source.
3. **The frozen-model control.** `input_ids` vs `inputs_embeds` on the *same*
   frozen encoder, with no adapter, to floating-point precision.
4. **The adapter contract** on real channel tensors built by the real B3B
   pipeline: tone `NA` exactly zero, empty letter channel exactly zero, initial
   gate `0.01`, shapes, parameter count `6d^2 + 16d`, zero trainable encoder
   parameters, the forced `g := 0` wiring identity, and that the initialised
   adapter is **not** falsely reported as identity.
5. **Gradient routing, through the real encoder.** One backward pass from a
   scalar derived from the encoder's **final hidden state** -- not from `z`. A
   loss on `z` alone would still pass if the integration path contained
   `z.detach()` or ran the encoder under `no_grad`, which would leave Stage-1
   unable to train `A_phi` through the encoder while this probe reported
   success.
6. **Module-mode invariants.** `requires_grad=False` freezes weights; `eval()`
   disables dropout. `nn.Module.train()` recurses into children, so the frozen
   encoder must be proven to stay in eval across every wrapper mode transition.

Colab::

    pip install "transformers==4.57.6" torch
    export HF_HOME="$PWD/.hf-cache"
    python scripts/fetch_vietnamese_syllable_inventory.py
    python scripts/b4b_phobert_adapter_probe.py --revision <40-char-sha>
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from unmark.alignment import (  # noqa: E402
    OrthographicRegion,
    align_chunk,
    character_letter_labels,
    overlay_orthography,
    project_piece,
    whitespace_chunks,
)
from unmark.alignment.contracts import REPO_LOCAL_HF_CACHE  # noqa: E402
from unmark.linguistics import load_inventory, make_classifier  # noqa: E402
from unmark.modeling.config import AdapterConfig  # noqa: E402
from unmark.modeling.contracts import GATE_INIT_TARGET  # noqa: E402
from unmark.orthography import Eligibility, canon, decompose  # noqa: E402

DEFAULT_CHECKPOINT = "vinai/phobert-base"
B4B_PROBE_REVISION = "01daacda68afe13d83023d16ec647239e344a1e6"
FULL_SHA_LENGTH = 40
_SNAPSHOT_PATTERN = re.compile(r"snapshots[/\\]([0-9a-f]{40})[/\\]")

# Exercises: marked tone, UNMARKED, tone NA (digits/punctuation), letter NONE,
# non-NONE letter labels, multi-piece tokens, punctuation, special tokens; the
# second, shorter sentence forces padding.
PROBE_SENTENCES: tuple[str, ...] = (
    "Tôi đang học nghiên cứu tại Đại học Quốc gia 2026.",
    "Chào bạn.",
)


def is_full_commit_sha(value: str | None) -> bool:
    return (
        isinstance(value, str)
        and len(value) == FULL_SHA_LENGTH
        and all(c in "0123456789abcdef" for c in value)
    )


def extract_snapshot_revision(path: str) -> str | None:
    if not isinstance(path, str):
        return None
    match = _SNAPSHOT_PATTERN.search(path)
    return match.group(1) if match else None


def observe_revision(obj: Any, attributes: Sequence[str]) -> tuple[str | None, tuple[str, ...], str]:
    """Read the resolved commit back off a loaded tokenizer or model (audit 010).

    Passing `revision=` does not prove the post-load state; the resolved cache
    snapshot path does.
    """
    import os as _os

    candidates: list[str] = []

    def consider(value: Any) -> None:
        if isinstance(value, str) and value and _os.sep in value and value not in candidates:
            candidates.append(value)

    for attribute in attributes:
        consider(getattr(obj, attribute, None))
    init_kwargs = getattr(obj, "init_kwargs", None)
    if isinstance(init_kwargs, dict):
        for value in init_kwargs.values():
            consider(value)
    config = getattr(obj, "config", None)
    if config is not None:
        consider(getattr(config, "_name_or_path", None))

    found: dict[str, list[str]] = {}
    for path in candidates:
        revision = extract_snapshot_revision(path)
        if revision:
            found.setdefault(revision, []).append(path)
    if not found:
        return None, tuple(candidates[:5]), "no Hugging Face snapshot path among the resolved files"
    if len(found) > 1:
        return None, tuple(candidates[:5]), f"resolved files disagree: {sorted(found)}"
    revision, evidence = next(iter(found.items()))
    return revision, tuple(evidence[:5]), "hugging face cache snapshot path"


# ---------------------------------------------------------------------------
# Channel tensors from the real deterministic pipeline
# ---------------------------------------------------------------------------
def raw_tokenize(tokenizer, text: str) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Raw pieces and ids. Never `convert_ids_to_tokens`: that round trip
    destroys the surface of any OOV piece (the `khut` finding, D-B3B1B-002)."""
    tokens = tuple(tokenizer.tokenize(text))
    return tokens, tuple(tokenizer.convert_tokens_to_ids(list(tokens)))


def build_regions(base_text: str, parts) -> list[OrthographicRegion]:
    regions: list[OrthographicRegion] = []
    cursor = 0
    for span in parts.syllables:
        if span.base_start > cursor:
            regions.append(
                OrthographicRegion(
                    len(regions), base_text[cursor : span.base_start], cursor, span.base_start,
                    Eligibility.NOT_APPLICABLE, is_syllable=False,
                )
            )
        regions.append(
            OrthographicRegion(
                len(regions), span.base_text, span.base_start, span.base_end, span.eligibility
            )
        )
        cursor = span.base_end
    if cursor < len(base_text):
        regions.append(
            OrthographicRegion(
                len(regions), base_text[cursor:], cursor, len(base_text),
                Eligibility.NOT_APPLICABLE, is_syllable=False,
            )
        )
    return regions


def project_sentence(tokenizer, text: str, classifier, unk_id: int | None):
    """Run the real B1A/B3A/B3B pipeline and return (content_ids, projections)."""
    parts = decompose(canon(text), eligibility_classifier=classifier)
    base_text = parts.base_text
    labels = character_letter_labels(parts)
    regions = build_regions(base_text, parts)
    tones = {
        region.index: span.observed_tone
        for region in regions
        if region.is_syllable
        for span in parts.syllables
        if span.base_start == region.start
    }

    projections = []
    content_ids: list[int] = []
    for chunk in whitespace_chunks(base_text):
        tokens, ids = raw_tokenize(tokenizer, chunk.text)
        alignment = align_chunk(chunk, tokens, ids, unk_token_id=unk_id)
        overlays = overlay_orthography(alignment.pieces, regions)
        for piece, overlay in zip(alignment.pieces, overlays):
            projections.append(
                project_piece(
                    len(projections), piece, overlay, base_text, labels, regions, tones
                )
            )
            content_ids.append(piece.token_id)
    return base_text, content_ids, projections


# ---------------------------------------------------------------------------
# Position-id instrumentation
# ---------------------------------------------------------------------------
def find_position_embedding(model):
    """Locate the real position-embedding module, by inspection not assumption."""
    import torch.nn as nn

    for name, module in model.named_modules():
        if name.endswith("position_embeddings") and isinstance(module, nn.Embedding):
            return name, module
    return None, None


class PositionCapture:
    """Records the actual index tensor the position-embedding module receives."""

    def __init__(self, module) -> None:
        self.module = module
        self.captured: list[list[list[int]]] = []
        self._handle = None

    def __enter__(self) -> PositionCapture:
        def hook(_module, inputs, _output):
            if inputs and hasattr(inputs[0], "tolist"):
                self.captured.append(inputs[0].detach().cpu().tolist())

        self._handle = self.module.register_forward_hook(hook)
        return self

    def __exit__(self, *_exc) -> None:
        if self._handle is not None:
            self._handle.remove()

    @property
    def last(self) -> list[list[int]] | None:
        return self.captured[-1] if self.captured else None


def compare_paths(model, batch, position_module) -> dict[str, Any]:
    """Path A (`input_ids`) vs path B (`inputs_embeds`), then C if they differ."""
    import torch

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    with torch.no_grad():
        with PositionCapture(position_module) as capture_a:
            out_a = model(input_ids=input_ids, attention_mask=attention_mask)
        positions_a = capture_a.last

        embeddings = model.get_input_embeddings()(input_ids)
        with PositionCapture(position_module) as capture_b:
            out_b = model(inputs_embeds=embeddings, attention_mask=attention_mask)
        positions_b = capture_b.last

    identical = positions_a == positions_b
    result: dict[str, Any] = {
        "position_ids_input_ids_path": positions_a,
        "position_ids_inputs_embeds_path": positions_b,
        "position_ids_identical": identical,
        "hidden_diff_b": tensor_diff(out_a.last_hidden_state, out_b.last_hidden_state, attention_mask),
    }

    if not identical and positions_a is not None:
        explicit = torch.tensor(positions_a, dtype=torch.long, device=input_ids.device)
        with torch.no_grad():
            with PositionCapture(position_module) as capture_c:
                out_c = model(
                    inputs_embeds=embeddings,
                    attention_mask=attention_mask,
                    position_ids=explicit,
                )
            positions_c = capture_c.last
        result["position_ids_explicit_path"] = positions_c
        result["explicit_matches_authoritative"] = positions_c == positions_a
        result["hidden_diff_c"] = tensor_diff(
            out_a.last_hidden_state, out_c.last_hidden_state, attention_mask
        )
    return result


def tensor_diff(a, b, attention_mask=None) -> dict[str, Any]:
    """Absolute differences, overall and restricted to attended positions.

    The split matters: if the two paths disagree only at padding positions, the
    overall figure hides a result that the content figure states plainly.
    """
    import torch

    delta = (a - b).abs()
    out = {
        "max_abs_diff": float(delta.max()),
        "mean_abs_diff": float(delta.mean()),
        "shape": list(a.shape),
    }
    if attention_mask is not None:
        keep = attention_mask.bool().unsqueeze(-1).expand_as(delta)
        content = delta[keep]
        if content.numel():
            out["max_abs_diff_content"] = float(content.max())
            out["mean_abs_diff_content"] = float(content.mean())
        padding = delta[~keep]
        if padding.numel():
            out["max_abs_diff_padding"] = float(padding.max())
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def render_report(config: dict[str, Any], summary: dict[str, Any], detail: dict[str, Any]) -> str:
    lines: list[str] = []
    a = lines.append
    a("# B4B — PhoBERT adapter integration probe")
    a("")
    a(f"Run `{config['run_id']}` — {config['checkpoint']}@{config['revision'][:12]}")
    a("")
    a("Real model weights **were** loaded. **Nothing was trained**: no optimizer, no")
    a("`optimizer.step()`, no parameter update, no dataset, no checkpoint saved. One")
    a("backward pass ran solely to validate gradient routing.")
    a("")
    a(f"Checks computed: **{len(summary['checks'])}**.")
    a("")
    a("## Provenance")
    a("")
    a("| | |")
    a("|---|---|")
    for key in (
        "checkpoint", "revision", "resolved_tokenizer_revision", "resolved_model_revision",
        "tokenizer_class", "model_class", "transformers_version", "torch_version",
        "hidden_size", "vocab_size", "pad_token_id", "model_dtype", "device", "python",
    ):
        a(f"| **{key}** | `{config.get(key)}` |")
    a("")
    a(f"Special tokens: `{config.get('special_tokens')}`")
    a("")
    a("## Verdict checks")
    a("")
    a("| # | Check | Result |")
    a("|---|---|---|")
    for index, (name, ok) in enumerate(summary["checks"].items(), start=1):
        a(f"| {index} | {name} | {'**PASS**' if ok else '**FAIL**'} |")
    a("")
    a("## Position ids — the question Audit 014 left open")
    a("")
    a("Captured from the real position-embedding module by forward hook, not inferred")
    a("from source.")
    a("")
    a("| Case | A == B | explicit C matches A | max abs diff (content) |")
    a("|---|---|---|---|")
    for case, row in detail.get("position_cases", {}).items():
        diff = row.get("hidden_diff_b", {})
        a(
            f"| `{case}` | {row.get('position_ids_identical')} |"
            f" {row.get('explicit_matches_authoritative', '—')} |"
            f" {diff.get('max_abs_diff_content', diff.get('max_abs_diff'))} |"
        )
    a("")
    a(f"**Explicit `position_ids` required: {summary['explicit_position_ids_required']}**")
    a("")
    a("## Frozen-model control")
    a("")
    a("`input_ids` vs base `inputs_embeds`, same frozen encoder, no adapter,")
    a("`model.eval()` under `torch.no_grad()` so dropout is disabled.")
    a("")
    a("```json")
    a(json.dumps(detail.get("baseline_equivalence", {}), indent=2))
    a("```")
    a("")
    a("## Adapter")
    a("")
    a("```json")
    a(json.dumps(summary.get("adapter", {}), indent=2))
    a("```")
    a("")
    a("## Gradient routing")
    a("")
    a("The diagnostic scalar is derived from the encoder's **final hidden state**,")
    a("so the backward pass traverses the frozen encoder into `A_phi`. A loss taken")
    a("directly from `z` would not have tested that path.")
    a("")
    a("```json")
    a(json.dumps(summary.get("gradients", {}), indent=2))
    a("```")
    a("")
    a("## Module modes")
    a("")
    a("`requires_grad=False` freezes weights; `eval()` disables dropout. They are")
    a("different contracts, and `nn.Module.train()` recurses into children.")
    a("")
    a("| Step | wrapper | encoder | adapter |")
    a("|---|---|---|---|")
    for step in summary.get("module_modes", []):
        a(
            f"| `{step['step']}` | {step['wrapper_training']} |"
            f" **{step['encoder_training']}** | {step['adapter_training']} |"
        )
    a("")
    a("The encoder column must be `False` on every row.")
    a("")
    a("## Scope")
    a("")
    a("The revision above is a **probe** revision, not the final backbone decision.")
    a("D-B3B0-002 remains OPEN.")
    a("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="B4B PhoBERT adapter integration probe (Colab).")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--revision", default=B4B_PROBE_REVISION, help="full 40-char commit SHA")
    parser.add_argument("--output-root", default="results/b4b")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--seed", type=int, default=20260820)
    args = parser.parse_args(argv)

    # Resolve before anything can change the working directory.
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (REPO_ROOT / output_root).resolve()

    if not is_full_commit_sha(args.revision):
        print(f"--revision {args.revision!r} is not a full 40-character lowercase commit SHA.", file=sys.stderr)
        return 2

    try:
        import torch
        import transformers
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        print(
            "torch/transformers are not installed. This probe is for Colab, not the\n"
            "ML-free local .venv. In Colab:\n\n"
            '    pip install "transformers==4.57.6" torch\n'
            f'    export HF_HOME="$PWD/{REPO_LOCAL_HF_CACHE}"\n'
            "    python scripts/fetch_vietnamese_syllable_inventory.py\n"
            f"    python scripts/b4b_phobert_adapter_probe.py --revision {args.revision}\n",
            file=sys.stderr,
        )
        return 2

    from unmark.modeling.adapter import (
        OrthographyInputAdapter,
        UnmarkEncoder,
        convex_combination,
        trainable_parameters,
    )
    from unmark.modeling.collate import build_example, collate_examples
    from unmark.modeling.pooling import masked_mean_non_special

    torch.manual_seed(args.seed)
    inventory = load_inventory()
    classifier = make_classifier(inventory)

    print(f"Loading tokenizer and MODEL WEIGHTS: {args.checkpoint}@{args.revision[:12]}")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, revision=args.revision, use_fast=False)
    model = AutoModel.from_pretrained(args.checkpoint, revision=args.revision)
    model.eval()

    tok_rev, tok_evidence, tok_source = observe_revision(
        tokenizer, ("vocab_file", "merges_file", "tokenizer_file", "name_or_path")
    )
    model_rev, model_evidence, model_source = observe_revision(model, ("name_or_path",))
    for label, observed in (("tokenizer", tok_rev), ("model", model_rev)):
        if observed is not None and observed != args.revision:
            print(f"REFUSING: {label} resolved to {observed}, not {args.revision}", file=sys.stderr)
            return 3
    revision_verified = tok_rev == args.revision and model_rev == args.revision

    hidden_size = int(model.config.hidden_size)
    pad_token_id = tokenizer.pad_token_id
    unk_id = tokenizer.unk_token_id
    device = next(model.parameters()).device

    # -- channel tensors from the real deterministic pipeline ---------------
    examples, sentence_detail = [], []
    for text in PROBE_SENTENCES:
        base_text, content_ids, projections = project_sentence(tokenizer, text, classifier, unk_id)
        full_ids = tokenizer.build_inputs_with_special_tokens(list(content_ids))
        special_mask = tokenizer.get_special_tokens_mask(
            list(content_ids), already_has_special_tokens=False
        )
        example = build_example(full_ids, special_mask, projections)
        examples.append(example)
        sentence_detail.append({
            "text": text,
            "base_text": base_text,
            "content_tokens": len(content_ids),
            "sequence_length": example.length,
            "tone_labels": [p.tone.label.value for p in projections],
            "letter_labels": [[c.value for c in p.letter.applicable_labels] for p in projections],
        })

    batch = collate_examples(examples, pad_token_id=pad_token_id)
    batch = {key: value.to(device) for key, value in batch.items()}

    # -- position ids -------------------------------------------------------
    position_name, position_module = find_position_embedding(model)
    position_cases: dict[str, Any] = {}
    if position_module is None:
        position_cases["error"] = "no position-embedding module found; cannot instrument"
    else:
        single = {k: v[:1, ...] for k, v in batch.items()}
        # CASE 1 is genuinely unpadded only after trimming the pad columns.
        keep = int(single["attention_mask"][0].sum())
        case1 = {k: v[:, :keep, ...] for k, v in single.items()}
        case3_inputs = collate_examples(
            [examples[0], examples[1], examples[1]], pad_token_id=pad_token_id
        )
        case3 = {k: v.to(device) for k, v in case3_inputs.items()}

        position_cases["case1_single_no_padding"] = compare_paths(model, case1, position_module)
        position_cases["case2_batch_right_padding"] = compare_paths(model, batch, position_module)
        position_cases["case3_unequal_lengths"] = compare_paths(model, case3, position_module)
        position_cases["case4_real_special_tokens"] = {
            "special_tokens_present": bool(batch["special_tokens_mask"].sum() > 0),
            "special_token_ids": sorted(
                {int(i) for i, s in zip(batch["input_ids"].flatten(), batch["special_tokens_mask"].flatten()) if s}
            ),
            **compare_paths(model, batch, position_module),
        }

    differing = [
        name for name, row in position_cases.items()
        if isinstance(row, dict) and row.get("position_ids_identical") is False
    ]
    explicit_required = bool(differing)
    explicit_recovers = all(
        position_cases[name].get("explicit_matches_authoritative") for name in differing
    ) if differing else True

    def authoritative_positions(case_batch):
        if not explicit_required or position_module is None:
            return None
        with torch.no_grad(), PositionCapture(position_module) as capture:
            model(input_ids=case_batch["input_ids"], attention_mask=case_batch["attention_mask"])
        return torch.tensor(capture.last, dtype=torch.long, device=device)

    # -- frozen-model control ----------------------------------------------
    positions = authoritative_positions(batch)
    with torch.no_grad():
        out_a = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        e_base = model.get_input_embeddings()(batch["input_ids"])
        kwargs = {"position_ids": positions} if positions is not None else {}
        out_b = model(inputs_embeds=e_base, attention_mask=batch["attention_mask"], **kwargs)
    baseline = tensor_diff(out_a.last_hidden_state, out_b.last_hidden_state, batch["attention_mask"])
    baseline_ok = baseline.get("max_abs_diff_content", baseline["max_abs_diff"]) < 1e-4

    # -- adapter ------------------------------------------------------------
    config = AdapterConfig(hidden_size=hidden_size)
    adapter = OrthographyInputAdapter(config).to(device)
    wrapped = UnmarkEncoder(model, adapter)

    with torch.no_grad():
        tone = adapter.tone_channel(batch["tone_ids"], batch["tone_mask"])
        letter = adapter.letter_channel(batch["letter_ids"], batch["letter_mask"])
        na_tone_zero = bool((tone[~batch["tone_mask"]].abs().max() == 0).item()) if (~batch["tone_mask"]).any() else True
        empty_rows = ~batch["letter_mask"].any(dim=-1)
        empty_letter_zero = bool((letter[empty_rows].abs().max() == 0).item()) if empty_rows.any() else True

        gate = adapter.gate_values(
            e_base, batch["tone_ids"], batch["tone_mask"], batch["letter_ids"], batch["letter_mask"]
        )
        gate_min, gate_max = float(gate.min()), float(gate.max())
        gate_ok = abs(gate_min - GATE_INIT_TARGET) < 1e-6 and abs(gate_max - GATE_INIT_TARGET) < 1e-6

        z = adapter(
            e_base, batch["tone_ids"], batch["tone_mask"], batch["letter_ids"], batch["letter_mask"]
        )
        shapes_ok = tuple(z.shape) == tuple(e_base.shape)
        z_differs = float((z - e_base).abs().max()) > 0

        # Forced g := 0 -- wiring test on the primitive, never a public flag.
        fused = adapter.layer_norm(adapter.fusion(torch.cat([e_base, tone, letter], dim=-1)))
        wired = convex_combination(torch.zeros_like(fused), fused, e_base)
        wiring_ok = bool(torch.equal(wired, e_base))

        pooled = masked_mean_non_special(
            out_a.last_hidden_state, batch["attention_mask"], batch["special_tokens_mask"]
        )

    expected_params = 6 * hidden_size**2 + 16 * hidden_size
    adapter_params = trainable_parameters(adapter)
    encoder_trainable = trainable_parameters(model)

    # -- module-mode invariants --------------------------------------------
    # requires_grad=False freezes weights; eval() disables dropout. They are
    # different contracts, and nn.Module.train() recurses into children -- so a
    # plain wrapper.train() would silently reactivate the frozen encoder's
    # dropout. Exercise the full transition sequence.
    def mode_snapshot(label: str) -> dict[str, Any]:
        return {
            "step": label,
            "wrapper_training": bool(wrapped.training),
            "encoder_training": bool(model.training),
            "adapter_training": bool(adapter.training),
            "encoder_requires_grad_any": any(p.requires_grad for p in model.parameters()),
        }

    mode_steps = [mode_snapshot("constructed")]
    wrapped.train()
    mode_steps.append(mode_snapshot("wrapper.train()"))
    wrapped.eval()
    mode_steps.append(mode_snapshot("wrapper.eval()"))
    wrapped.train()
    mode_steps.append(mode_snapshot("wrapper.train() again"))

    encoder_always_eval = all(not step["encoder_training"] for step in mode_steps)
    adapter_follows_mode = (
        mode_steps[1]["adapter_training"] is True
        and mode_steps[2]["adapter_training"] is False
        and mode_steps[3]["adapter_training"] is True
    )
    encoder_stays_frozen = all(not step["encoder_requires_grad_any"] for step in mode_steps)

    # -- gradient routing: ONE backward pass, no optimizer, no update -------
    # The loss MUST come from the real encoder's output, not from z. A loss on z
    # alone would still pass if the integration path contained z.detach() or ran
    # the encoder under no_grad -- which would make Stage-1 unable to train
    # A_phi through the encoder while this probe reported success.
    #
    # Deliberately NOT inside torch.no_grad(): this is the path future Stage-1
    # will use. The equivalence control above is inference-only and correctly
    # uses no_grad; this is a different path.
    wrapped.train()  # adapter -> train, frozen encoder -> forced back to eval
    model.zero_grad(set_to_none=True)
    adapter.zero_grad(set_to_none=True)

    grad_outputs = wrapped(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        tone_ids=batch["tone_ids"],
        tone_mask=batch["tone_mask"],
        letter_ids=batch["letter_ids"],
        letter_mask=batch["letter_mask"],
        position_ids=positions,
    )
    grad_hidden = grad_outputs.last_hidden_state
    hidden_requires_grad = bool(grad_hidden.requires_grad)
    # A finite diagnostic scalar over attended, non-special final hidden states.
    # NOT a scientific objective: Stage-1's cosine loss is a later phase.
    diagnostic_loss = masked_mean_non_special(
        grad_hidden, batch["attention_mask"], batch["special_tokens_mask"]
    ).sum()
    diagnostic_loss.backward()

    encoder_grads = [
        n for n, p in model.named_parameters()
        if p.grad is not None and float(p.grad.abs().sum()) != 0
    ]
    encoder_grad_tensors = [n for n, p in model.named_parameters() if p.grad is not None]
    adapter_grads = {
        name: (None if p.grad is None else bool(torch.isfinite(p.grad).all()))
        for name, p in adapter.named_parameters()
    }
    required_components = {
        "gate.weight": adapter.gate.weight.grad is not None,
        "gate.bias": adapter.gate.bias.grad is not None,
        "fusion.weight": adapter.fusion.weight.grad is not None,
        "fusion.bias": adapter.fusion.bias.grad is not None,
        "layer_norm.weight": adapter.layer_norm.weight.grad is not None,
        "layer_norm.bias": adapter.layer_norm.bias.grad is not None,
        "tone_embedding.weight": adapter.tone_embedding.weight.grad is not None,
        "letter_embedding.weight": adapter.letter_embedding.weight.grad is not None,
    }
    # Connectivity, not magnitude: at least one adapter gradient must be nonzero,
    # otherwise the graph could be connected but severed numerically.
    adapter_grad_nonzero = any(
        p.grad is not None and float(p.grad.abs().sum()) != 0 for p in adapter.parameters()
    )
    gradients_reach_adapter = all(required_components.values()) and adapter_grad_nonzero

    adapter.zero_grad(set_to_none=True)
    model.zero_grad(set_to_none=True)
    wrapped.eval()

    checks = {
        "revision verified (tokenizer and model)": revision_verified,
        "hidden size read from model": hidden_size > 0,
        "special token ids recorded": bool(batch["special_tokens_mask"].sum() > 0),
        "position-id behaviour determined": position_module is not None,
        "explicit position_ids recover the authoritative path when needed": explicit_recovers,
        "input_ids vs inputs_embeds control equivalent": baseline_ok,
        "tone NA is exactly zero": na_tone_zero,
        "empty letter channel is exactly zero": empty_letter_zero,
        "initial gate is 0.01": gate_ok,
        "adapter output shape correct": shapes_ok,
        "parameter count equals 6d^2 + 16d": adapter_params == expected_params,
        "encoder has zero trainable parameters": encoder_trainable == 0,
        "forced g=0 wiring identity holds": wiring_ok,
        "initialised adapter is not identity": z_differs,
        "no encoder gradients": not encoder_grads,
        "gradient loss derived from real encoder output": hidden_requires_grad,
        "gradient graph reaches the adapter through the frozen encoder": gradients_reach_adapter,
        "adapter gradients finite": all(v is not False for v in adapter_grads.values()),
        "encoder stays eval across every mode transition": encoder_always_eval,
        "adapter follows the wrapper train/eval mode": adapter_follows_mode,
        "encoder stays frozen across mode transitions": encoder_stays_frozen,
        "stage-1 pooling returns [B, d]": tuple(pooled.shape) == (batch["input_ids"].shape[0], hidden_size),
    }
    complete = all(checks.values())

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    config_record = {
        "run_id": run_id,
        "checkpoint": args.checkpoint,
        "revision": args.revision,
        "resolved_tokenizer_revision": tok_rev,
        "resolved_model_revision": model_rev,
        "revision_evidence": {"tokenizer": list(tok_evidence), "model": list(model_evidence)},
        "revision_evidence_source": {"tokenizer": tok_source, "model": model_source},
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_is_fast": bool(getattr(tokenizer, "is_fast", False)),
        "model_class": type(model).__name__,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "hidden_size": hidden_size,
        "vocab_size": int(model.config.vocab_size),
        "pad_token_id": pad_token_id,
        "unk_token_id": unk_id,
        "special_tokens": tokenizer.all_special_tokens,
        "special_token_ids": tokenizer.all_special_ids,
        "model_dtype": str(next(model.parameters()).dtype),
        "device": str(device),
        "position_embedding_module": position_name,
        "seed": args.seed,
        "python": platform.python_version(),
        "model_weights_loaded": True,
        "training_performed": False,
        "output_root": str(output_root),
    }
    summary = {
        "checks": checks,
        "explicit_position_ids_required": explicit_required,
        "position_cases_differing": differing,
        "adapter": {
            "hidden_size": hidden_size,
            "trainable_parameters": adapter_params,
            "expected_parameters": expected_params,
            "encoder_trainable_parameters": encoder_trainable,
            "wrapped_trainable_parameters": wrapped.trainable_parameter_count(),
            "initial_gate_min": gate_min,
            "initial_gate_max": gate_max,
            "z_max_abs_diff_from_base": float((z - e_base).abs().max()),
        },
        "gradients": {
            "gradient_loss_source": "encoder_final_hidden_state",
            "gradient_path_includes_encoder": True,
            "encoder_output_requires_grad": hidden_requires_grad,
            "encoder_parameters_with_nonzero_grad": encoder_grads[:10],
            "encoder_grad_count": len(encoder_grads),
            "encoder_parameters_with_grad_tensor": len(encoder_grad_tensors),
            "required_adapter_components_with_grad": required_components,
            "adapter_grad_nonzero_somewhere": adapter_grad_nonzero,
            "adapter_parameters_with_finite_grad": sum(1 for v in adapter_grads.values() if v),
            "adapter_parameters_without_grad": [n for n, v in adapter_grads.items() if v is None],
        },
        "module_modes": mode_steps,
        "status": "B4B_PHOBERT_ADAPTER_INTEGRATION_COMPLETE" if complete
        else "B4B_PHOBERT_ADAPTER_INTEGRATION_INCOMPLETE",
        "failed_checks": [name for name, ok in checks.items() if not ok],
    }
    detail = {
        "position_cases": position_cases,
        "baseline_equivalence": baseline,
        "sentences": sentence_detail,
    }

    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config_record, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "position_ids.json").write_text(json.dumps(position_cases, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "equivalence.json").write_text(json.dumps(baseline, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "gradients.json").write_text(json.dumps(summary["gradients"], indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "module_modes.json").write_text(json.dumps(mode_steps, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "channels.json").write_text(json.dumps(sentence_detail, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "report.md").write_text(render_report(config_record, summary, detail), encoding="utf-8")

    print(f"\nWrote {run_dir}")
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"\n  explicit position_ids required : {explicit_required}")
    print(f"  adapter trainable parameters   : {adapter_params} (expected {expected_params})")
    print(f"  encoder trainable parameters   : {encoder_trainable}")
    print(f"  gradient loss source           : encoder_final_hidden_state")
    print(f"  encoder eval across transitions: {encoder_always_eval}")
    print("\n" + summary["status"])
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
