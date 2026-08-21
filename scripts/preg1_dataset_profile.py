#!/usr/bin/env python3
"""Pre-G1 dataset profiler — data-only, with Colab-only tokenizer profiling.

Two phases, deliberately separable:

1. **Data-only** (`--data-only`): provenance, label balance, orthographic
   observables, duplicates and leakage. Pure Python — runnable anywhere, and
   what the local test suite exercises.
2. **Token-length** (default): additionally loads the pinned **tokenizer only**
   to profile Vanilla and Base-only sequence lengths on TRAIN, and applies the
   precommitted `max_length` rule. **Colab only** — the local `.venv` is
   deliberately ML-free.

**Nothing here trains.** No head, no optimizer, no model weights (the tokenizer
is not the model), no downstream score, and the official TEST split is never
consulted for any protocol decision.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from unmark.evaluation.preg1_protocol import (  # noqa: E402
    LABEL_MAPPING,
    MEASUREMENT_SEEDS,
    PRIMARY_DATASET,
    PRIMARY_DATASET_VERSION,
    PRIMARY_TASK,
    SPLIT_SEED,
    TUNING_SEEDS,
    Preg1Protocol,
)
from unmark.evaluation.profiling import (  # noqa: E402
    FIXED_MAX_LENGTH,
    PROFILE_SCHEMA_VERSION,
    DatasetAccess,
    DatasetProvenance,
    FileProvenance,
    analyse_duplicates,
    distribution,
    file_sha256,
    length_coverage,
    profile_split,
)
from unmark.orthography import canon, decompose  # noqa: E402

PHOBERT_CHECKPOINT = "vinai/phobert-base"
PHOBERT_REVISION = "01daacda68afe13d83023d16ec647239e344a1e6"


def read_split(path: Path, text_column: str, label_column: str, id_column: str | None):
    """Read one delimited split. Returns `(sample_id, text, label)` triples."""
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        delimiter = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
        for index, row in enumerate(csv.DictReader(handle, delimiter=delimiter)):
            sample_id = row.get(id_column) if id_column else None
            rows.append(
                (
                    str(sample_id) if sample_id else f"{path.stem}-{index:06d}",
                    row.get(text_column, ""),
                    row.get(label_column, ""),
                )
            )
    return rows


def tokenize_lengths(tokenizer, texts: Sequence[str]) -> dict[str, Any]:
    """Vanilla and Base-only token lengths and UNK counts, kept **separate**.

    Lengths use `build_inputs_with_special_tokens`, so they match the convention
    the future evaluator uses rather than a bare piece count.

    **UNK counts are per pathway.** An earlier version accumulated one counter
    across both, which made the reported total unattributable: a single number
    could not say whether stripping to `b(x)` introduced unknown pieces, which is
    exactly the question a two-pathway profile is asked. Tokenization itself is
    unchanged.
    """
    vanilla, base = [], []
    vanilla_unk = base_unk = 0
    unk_id = getattr(tokenizer, "unk_token_id", None)
    for text in texts:
        canonical = canon(text)
        base_text = decompose(canonical).base_text
        for surface, sink, is_vanilla in (
            (canonical, vanilla, True),
            (base_text, base, False),
        ):
            pieces = tokenizer.tokenize(surface)
            ids = tokenizer.convert_tokens_to_ids(pieces)
            sink.append(len(tokenizer.build_inputs_with_special_tokens(ids)))
            if unk_id is not None:
                count = sum(1 for i in ids if i == unk_id)
                if is_vanilla:
                    vanilla_unk += count
                else:
                    base_unk += count
    return {
        "vanilla_lengths": vanilla,
        "base_only_lengths": base,
        "vanilla_unk_token_count": vanilla_unk,
        "base_only_unk_token_count": base_unk,
        "total_unk_token_count": vanilla_unk + base_unk,
    }


def render_report(config: dict[str, Any], summary: dict[str, Any]) -> str:
    lines: list[str] = []
    a = lines.append
    a(f"# Pre-G1 dataset profile — {config['dataset']}")
    a("")
    a(f"Run `{config['run_id']}` · schema `{config['schema_version']}`")
    a("")
    a("**Data-only profile. No head training, no optimizer, no downstream score.**")
    a("")
    a("## Provenance")
    a("")
    a("```json")
    a(json.dumps(summary.get("provenance", {}), indent=2, ensure_ascii=False))
    a("```")
    a("")
    prov = summary.get("provenance", {})
    if not prov.get("usable_for_scientific_run"):
        a(f"> **Access is `{prov.get('access')}`, not an official distribution.**")
        a("> Usable for profiling; a scientific pre-G1 run needs an official copy.")
        a("")
    if not prov.get("license_established"):
        a("> **No explicit license metadata was established.** Official public")
        a("> distribution and an identified license are different facts.")
        a("")
    a("## Splits")
    a("")
    a("| Split | N | base-equivalent | canon changed | canonical dups | conflicting-label groups |")
    a("|---|---|---|---|---|---|")
    for name, profile in summary.get("splits", {}).items():
        a(
            f"| `{name}` | {profile['examples']} | "
            f"{profile['base_equivalent']} ({profile['base_equivalent_rate']:.1%}) | "
            f"{profile['canon_changed']} | {profile['canonical_duplicate_texts']} | "
            f"{profile['conflicting_label_groups']} |"
        )
    a("")
    a("### Unit-level channel densities (§4.3 granularity)")
    a("")
    a("| Split | observed tone syllables / eligible | density | observed letter units / applicable | density |")
    a("|---|---|---|---|---|")
    for name, profile in summary.get("splits", {}).items():
        tone_density = profile.get("observed_tone_unit_density")
        letter_density = profile.get("observed_letter_unit_density")
        a(
            f"| `{name}` | {profile['tone_observed_syllables']} / "
            f"{profile['tone_eligible_syllables']} | "
            f"{'unresolved' if tone_density is None else f'{tone_density:.4f}'} | "
            f"{profile['letter_observed_units']} / {profile['letter_eligible_units']} | "
            f"{'n/a' if letter_density is None else f'{letter_density:.4f}'} |"
        )
    a("")
    a("Tone denominator = syllables with Eligibility `VIETNAMESE_CANDIDATE`.")
    a("Letter denominator = character units whose `LetterDiacritic` is **not NA**;")
    a("`NONE` is included, `NA` is not.")
    a("")
    a("`base-equivalent` = **no observed mark**. It is *not* a missing-diacritic")
    a("rate: unmarked Vietnamese is observationally ambiguous (§4.3).")
    a("")
    a("## Duplicates and leakage")
    a("")
    a("```json")
    a(json.dumps(summary.get("duplicates", {}), indent=2, ensure_ascii=False))
    a("```")
    a("")
    if "token_lengths" in summary:
        a("## Token lengths (TRAIN only)")
        a("")
        a("```json")
        a(json.dumps(summary["token_lengths"], indent=2, ensure_ascii=False))
        a("```")
        a("")
        a(f"**max_length: fixed at `{summary['max_length']['value']}`** — not selected from data.")
        a("")
        a(f"Overflow at {summary['max_length']['value']}: "
          f"vanilla {summary['max_length']['vanilla_overflow_rate']:.3%}, "
          f"base-only {summary['max_length']['base_only_overflow_rate']:.3%}")
        a("")
    else:
        a("## Token lengths")
        a("")
        a("Not profiled in this run (`--data-only`). `max_length` remains fixed at 256.")
        a("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pre-G1 data-only dataset profiler.")
    parser.add_argument("--dataset", default=PRIMARY_DATASET)
    parser.add_argument("--dataset-version", default=PRIMARY_DATASET_VERSION)
    parser.add_argument("--task", default=PRIMARY_TASK)
    parser.add_argument("--validation", default=None,
                        help="official validation = measurement-dev; integrity profiling only")
    parser.add_argument("--train", required=True, help="path to the official train split")
    parser.add_argument("--test", default=None, help="optional; profiled for integrity ONLY")
    parser.add_argument("--text-column", required=True)
    parser.add_argument("--label-column", required=True)
    parser.add_argument("--id-column", default=None)
    parser.add_argument(
        "--access", required=True, choices=[a.value for a in DatasetAccess],
        help="how the copy was obtained; no default",
    )
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--source-revision", default=None)
    parser.add_argument("--source-url", default=None)
    parser.add_argument("--license-status", default=None,
                        help="only if an authoritative source states one; no license is invented")
    parser.add_argument("--data-only", action="store_true", help="skip tokenizer profiling")
    parser.add_argument("--revision", default=PHOBERT_REVISION)
    parser.add_argument("--output-root", default="results/preg1")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = (REPO_ROOT / output_root).resolve()

    splits = {"train": Path(args.train)}
    if args.validation:
        splits["validation"] = Path(args.validation)
    if args.test:
        splits["test"] = Path(args.test)

    records = {
        name: read_split(path, args.text_column, args.label_column, args.id_column)
        for name, path in splits.items()
    }
    # The B3A inventory resolves syllable eligibility, which is the DENOMINATOR
    # of the tone-channel density (§4.3: tone is a syllable property). Without it
    # every syllable is UNDECIDED and the density is reported unresolved rather
    # than silently computed.
    classifier = None
    eligibility_resolved = False
    try:
        from unmark.linguistics import make_classifier, try_load_inventory

        inventory = try_load_inventory()
        if inventory is not None:
            classifier = make_classifier(inventory)
            eligibility_resolved = True
    except Exception:  # pragma: no cover - inventory absence is reported, not fatal
        classifier = None

    profiles, indexes = {}, {}
    for name, rows in records.items():
        profile, index = profile_split(name, rows, classifier)
        profiles[name] = profile
        indexes[name] = index
    duplicates = analyse_duplicates(indexes)

    labels = sorted({str(label) for rows in records.values() for _, _, label in rows})
    from unmark.evaluation.profiling import LICENSE_NOT_ESTABLISHED

    provenance = DatasetProvenance(
        dataset_name=args.dataset,
        dataset_version=args.dataset_version,
        task=args.task,
        access=DatasetAccess(args.access),
        source_name=args.source_name,
        source_revision=args.source_revision,
        source_url=args.source_url,
        label_mapping=dict(LABEL_MAPPING) if set(labels) <= set(LABEL_MAPPING)
        else {label: index for index, label in enumerate(labels)},
        columns=(args.text_column, args.label_column),
        license_status=args.license_status or LICENSE_NOT_ESTABLISHED,
        files=tuple(
            FileProvenance(
                path=str(path.name),
                sha256=file_sha256(str(path)),
                size_bytes=path.stat().st_size,
                rows=len(records[name]),
            )
            for name, path in splits.items()
        ),
    )

    summary: dict[str, Any] = {
        # The profile contract must be readable from the TOP level. A consumer
        # should not have to reach into nested provenance to learn which schema
        # it is holding -- that is how config.json and provenance.json drifted
        # apart in the first place.
        "schema_version": PROFILE_SCHEMA_VERSION,
        "provenance": provenance.to_dict(),
        "splits": {name: p.to_dict() for name, p in profiles.items()},
        "duplicates": duplicates.to_dict(),
        "eligibility_resolved": eligibility_resolved,
        "eligibility_note": (
            "tone-density denominator requires the B3A syllable inventory; "
            "unresolved densities are reported as null, never as zero"
        ),
        "official_test_sealed": True,
        "official_test_used_for_protocol_decisions": False,
        "head_training_performed": False,
        "optimizer_created": False,
        "downstream_score_computed": False,
    }

    if not args.data_only:
        try:
            from transformers import AutoTokenizer
        except ImportError:
            print(
                "transformers is not installed. Token-length profiling is Colab-only;\n"
                "the local .venv is deliberately ML-free. Use --data-only locally, or in Colab:\n\n"
                '    pip install "transformers==4.57.6"\n'
                f"    python scripts/preg1_dataset_profile.py --revision {args.revision} ...\n",
                file=sys.stderr,
            )
            return 2
        tokenizer = AutoTokenizer.from_pretrained(
            PHOBERT_CHECKPOINT, revision=args.revision, use_fast=False
        )
        # TRAIN ONLY. The official test split is never tokenized for a protocol
        # decision. Token lengths are CHARACTERISED on train; they do not select
        # anything -- max_length is FIXED at 256 by D-PREG1-008b, decided before
        # this profiling existed, and coverage evidence must not reopen it.
        texts = [text for _, text, _ in records["train"]]
        measured = tokenize_lengths(tokenizer, texts)
        vanilla = measured["vanilla_lengths"]
        base = measured["base_only_lengths"]
        deltas = [b - v for v, b in zip(vanilla, base)]
        coverage = length_coverage(vanilla, base)
        at_fixed = next(c for c in coverage if c.threshold == FIXED_MAX_LENGTH)
        summary["token_lengths"] = {
            "tokenizer_revision": args.revision,
            "special_token_convention": "build_inputs_with_special_tokens",
            "vanilla": distribution(vanilla),
            "base_only": distribution(base),
            "base_minus_vanilla_delta": distribution(deltas),
            "length_changed_count": sum(1 for d in deltas if d != 0),
            "length_changed_fraction": sum(1 for d in deltas if d != 0) / len(deltas),
            "vanilla_unk_token_count": measured["vanilla_unk_token_count"],
            "base_only_unk_token_count": measured["base_only_unk_token_count"],
            "total_unk_token_count": measured["total_unk_token_count"],
            "unk_count_is_per_pathway": True,
            "coverage": [c.to_dict() for c in coverage],
        }
        # max_length is FIXED at 256 (D-PREG1-008b). The statistics above are
        # descriptive: they characterise the corpus and quantify truncation, and
        # they do NOT select the value.
        summary["max_length"] = {
            "value": FIXED_MAX_LENGTH,
            "fixed": True,
            "selected_from_data": False,
            "vanilla_overflow_rate": at_fixed.vanilla_overflow,
            "base_only_overflow_rate": at_fixed.base_only_overflow,
        }

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    config = {
        "run_id": run_id,
        "dataset": args.dataset,
        "schema_version": PROFILE_SCHEMA_VERSION,
        "data_only": args.data_only,
        "python": platform.python_version(),
        "protocol": Preg1Protocol().to_dict(),
        "split_seed": SPLIT_SEED,
        "tuning_seeds": list(TUNING_SEEDS),
        "measurement_seeds": list(MEASUREMENT_SEEDS),
    }

    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write = lambda name, payload: (run_dir / name).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write("config.json", config)
    write("provenance.json", summary["provenance"])
    write("dataset_profile.json", summary["splits"])
    write("orthography_profile.json", {n: p["base_equivalent_rate"] for n, p in summary["splits"].items()})
    write("duplicates.json", summary["duplicates"])
    if "token_lengths" in summary:
        write("token_length_profile.json", summary["token_lengths"])
    write("summary.json", summary)
    (run_dir / "report.md").write_text(render_report(config, summary), encoding="utf-8")

    print(f"\nWrote {run_dir}")
    for name, profile in summary["splits"].items():
        print(f"  {name:6} N={profile['examples']:6}  base-equivalent={profile['base_equivalent_rate']:.1%}")
    print(f"  conflicting-label groups : {len(duplicates.conflicting_label_groups)}")
    print(f"  cross-split groups       : {len(duplicates.cross_split_groups)}")
    print(f"  max_length               : {summary.get('max_length', {}).get('value', FIXED_MAX_LENGTH)} (fixed)")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
