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
    # optional, for the segmentation paths:
    #   pip install py_vncorenlp     (needs a JVM; Colab provides one)
    export HF_HOME="$PWD/.hf-cache"
    python scripts/b3b0_phobert_input_probe.py --checkpoint vinai/phobert-base
"""

from __future__ import annotations

import argparse
import json
import platform
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
def load_tokenizer(checkpoint: str, revision: str | None, use_fast: bool):
    """Tokenizer only. `AutoModel` is never called anywhere in this script."""
    from transformers import AutoTokenizer

    kwargs: dict[str, Any] = {"use_fast": use_fast}
    if revision:
        kwargs["revision"] = revision
    return AutoTokenizer.from_pretrained(checkpoint, **kwargs)


def describe_tokenizer(tokenizer, checkpoint: str, revision: str | None) -> TokenizerContract:
    import transformers

    return TokenizerContract(
        checkpoint=checkpoint,
        revision=revision,
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


def load_segmenter(save_dir: Path) -> tuple[Any, SegmenterContract]:
    """Try to bring up VnCoreNLP. Never fakes segmentation when unavailable."""
    try:
        import py_vncorenlp
    except Exception as exc:  # noqa: BLE001
        return None, SegmenterContract(
            available=False,
            name="VnCoreNLP",
            package="py_vncorenlp",
            notes=f"import failed: {type(exc).__name__}: {exc}",
        )
    try:
        version = getattr(py_vncorenlp, "__version__", None)
        save_dir.mkdir(parents=True, exist_ok=True)
        py_vncorenlp.download_model(save_dir=str(save_dir))
        segmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=str(save_dir))
        jars = sorted(p.name for p in save_dir.rglob("*.jar"))
        return segmenter, SegmenterContract(
            available=True,
            name="VnCoreNLP",
            package="py_vncorenlp",
            package_version=version,
            model_resource=", ".join(jars) or str(save_dir),
            model_version=None,
            # py_vncorenlp.download_model fetches whatever is current upstream.
            pinned=False,
            notes=(
                "REPRODUCIBILITY RISK: download_model() is not revision-pinned, so the "
                "segmentation model can change between runs. Record the jar checksums and "
                "pin them before any result depends on segmentation."
            ),
        )
    except Exception as exc:  # noqa: BLE001
        return None, SegmenterContract(
            available=False,
            name="VnCoreNLP",
            package="py_vncorenlp",
            notes=f"initialisation failed: {type(exc).__name__}: {exc}",
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
    if not config["segmenter"].get("available"):
        a("> **Segmentation paths UNAVAILABLE.** VnCoreNLP was not usable in this run, so")
        a("> the three segmentation pathways are reported as `UNAVAILABLE_SEGMENTER`")
        a("> rather than faked. Only `RAW_BASE` carries measurements.")
        a("")
    elif not config["segmenter"].get("pinned"):
        a("> **Reproducibility risk.** The segmentation model was not revision-pinned, so")
        a("> these numbers may not reproduce. Pin it before any result depends on it.")
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
    parser.add_argument("--revision", default=None, help="pin the tokenizer revision if known")
    parser.add_argument("--use-fast", action="store_true", default=True)
    parser.add_argument("--no-fast", dest="use_fast", action="store_false")
    parser.add_argument("--output-root", default="results/b3b0")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--skip-segmenter", action="store_true")
    args = parser.parse_args(argv)

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

    if args.skip_segmenter:
        segmenter, segmenter_contract = None, SegmenterContract(
            available=False, name="VnCoreNLP", notes="skipped by --skip-segmenter"
        )
    else:
        segmenter, segmenter_contract = load_segmenter(REPO_ROOT / ".vncorenlp")
    print(f"  segmenter available: {segmenter_contract.available}")

    observations = run_probe(tokenizer, segmenter, segmenter_contract)
    by_path: dict[str, list[PathObservation]] = {}
    for observation in observations:
        by_path.setdefault(observation.path.value, []).append(observation)
    comparison = compare_paths(by_path)

    status = STATUS_OK if segmenter_contract.available else STATUS_PARTIAL
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.output_root) / run_id
    suffix = 1
    while run_dir.exists():
        run_dir = Path(args.output_root) / f"{run_id}-{suffix}"
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
    print(f"Status: {status}")
    print("No preprocessing policy was chosen. See docs/spec/decisions.md D-B3B0-001.")
    print()
    print(f"Results: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
