#!/usr/bin/env python3
"""G-1 smoke test for the candidate RESTORE baseline - runtime entry point.

Gate G-1 (Phase 0) asks one narrow question: is the pinned off-the-shelf
Vietnamese diacritic restorer a *trustworthy candidate* for the ``RESTORE``
system of the UNMARK study? It does NOT ask whether the restorer is accurate,
and it does NOT compare anything to UNMARK.

Concretely this script checks that

  1. the pinned checkpoint downloads and loads reproducibly,
  2. inference runs at all,
  3. greedy decoding is deterministic across repeated calls,
  4. the model *restores marks* rather than *rewriting the lexical base*,
  5. it survives fully stripped, partially stripped and already-clean input,
     short ambiguous strings, proper names, mixed script, punctuation,
     numbers, URLs, e-mail addresses and emoji.

The script emits an ENGINEERING status only. The scientific G-1 decision is
deliberately left to the researcher; see the ``G-1 Assessment`` section of the
generated report.

WHERE THIS RUNS
---------------
Model inference is a **Colab-only** operation in this project. torch and
transformers are imported lazily, inside the model path, so that everything
this file does *without* the model -- listing the suite, validating the config
-- works in the lightweight local ``.venv``. Running the real smoke test
locally fails with an explicit message; nothing is ever auto-installed.

Usage
-----
    # Colab GPU runtime, inside the cloned repository
    .venv-colab/bin/python scripts/g_minus1_restore_smoke.py

    .venv-colab/bin/python scripts/g_minus1_restore_smoke.py \
        --config configs/restore/nrl_vit5_base.yaml \
        --output-root results/g_minus1

    # anywhere, no model needed
    python scripts/g_minus1_restore_smoke.py --list-cases
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

# Make the repository root importable when the script is invoked by path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from unmark.gates.g_minus1 import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT_ROOT,
    EXPERIMENT_DEPS_MESSAGE,
    SCRIPT_VERSION,
    SMOKE_CASES,
    STATUS_PASS,
    SmokeCase,
    build_record,
    engineering_settings,
    engineering_status,
    format_console_table,
    load_config,
    make_run_id,
    render_report,
    select_cases,
    summarize,
    unique_run_dir,
    validate_suite,
    write_artifacts,
)

# The revision was serialised by this transformers version; 5.x cannot load its
# tokenizer.json. See requirements/experiment.txt.
PINNED_TRANSFORMERS = "4.57.6"


class ExperimentDependenciesMissing(RuntimeError):
    """Raised when the heavy ML stack is absent - i.e. on the local machine."""


def require_experiment_dependencies() -> None:
    """Fail fast, and explain where the real run belongs. Never installs anything."""
    missing: list[str] = []
    for module in ("torch", "transformers"):
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    if missing:
        raise ExperimentDependenciesMissing(f"{EXPERIMENT_DEPS_MESSAGE}\nMissing module(s): {', '.join(missing)}")

    import transformers

    installed = transformers.__version__
    if installed.split(".")[0] != PINNED_TRANSFORMERS.split(".")[0]:
        print(
            f"[G-1] WARNING    : transformers {installed} is installed, but this checkpoint "
            f"requires {PINNED_TRANSFORMERS}.\n"
            f"[G-1]              transformers 5.x fails to load its tokenizer.json with\n"
            f"[G-1]              \"TypeError: argument 'vocab': 'dict' object cannot be "
            f"converted to 'Sequence'\".\n"
            f"[G-1]              Install requirements/experiment.txt to get the pinned version.",
            file=sys.stderr,
        )


def select_device(preference: str = "auto") -> str:
    """Resolve the device string; ``auto`` means CUDA when available, else CPU."""
    import torch

    if preference not in ("auto", "cpu", "cuda"):
        raise ValueError(f"unsupported device preference: {preference!r}")
    if preference == "cpu":
        return "cpu"
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device 'cuda' requested but torch.cuda.is_available() is False")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


class Restorer:
    """Thin, frozen wrapper around the pinned seq2seq diacritic restorer."""

    def __init__(self, cfg: dict[str, Any], device: str) -> None:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._torch = torch
        self.device = device
        self.model_id = cfg["model_id"]
        self.revision = cfg["revision"]
        self.generation_kwargs = dict(cfg["generation"])
        tok_cfg = cfg.get("tokenizer") or {}
        self.max_input_tokens = int(tok_cfg.get("max_input_tokens", 256))
        self.truncation = bool(tok_cfg.get("truncation", True))

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        # The checkpoint is stored in float32; G-1 keeps it there on purpose so
        # that CPU and GPU runs stay numerically comparable (see the config).
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id, revision=self.revision)
        self.model.to(device)
        self.model.eval()

    def metadata(self) -> dict[str, Any]:
        cfg = self.model.config
        return {
            "architecture": cfg.architectures[0] if getattr(cfg, "architectures", None) else type(self.model).__name__,
            "model_type": getattr(cfg, "model_type", None),
            "is_encoder_decoder": bool(getattr(cfg, "is_encoder_decoder", False)),
            "num_parameters": int(sum(p.numel() for p in self.model.parameters())),
            "parameter_dtype": str(next(self.model.parameters()).dtype),
            "tokenizer_class": type(self.tokenizer).__name__,
            "vocab_size": int(getattr(cfg, "vocab_size", 0)) or None,
            # Verifies the pin actually took effect.
            "resolved_commit_hash": getattr(cfg, "_commit_hash", None),
        }

    def encode(self, text: str):
        return self.tokenizer(
            text,
            return_tensors="pt",
            truncation=self.truncation,
            max_length=self.max_input_tokens,
        )

    def n_input_tokens(self, text: str) -> tuple[int, bool]:
        """Token count after truncation, and whether truncation actually bit."""
        untruncated = len(self.tokenizer(text)["input_ids"])
        kept = int(self.encode(text)["input_ids"].shape[-1])
        return kept, untruncated > kept

    def restore(self, text: str) -> str:
        torch = self._torch
        batch = {k: v.to(self.device) for k, v in self.encode(text).items()}
        with torch.inference_mode():
            generated = self.model.generate(**batch, **self.generation_kwargs)
        return self.tokenizer.decode(generated[0], skip_special_tokens=True)

    def synchronize(self) -> None:
        if self.device.startswith("cuda"):
            self._torch.cuda.synchronize()


def run_case(case: SmokeCase, restorer: Restorer, repeats: int) -> dict[str, Any]:
    """Run one case ``repeats`` times and build its record."""
    outputs: list[str] = []
    latencies: list[float] = []
    error: str | None = None
    n_tokens: int | None = None
    truncated: bool | None = None
    try:
        n_tokens, truncated = restorer.n_input_tokens(case.text)
        for _ in range(repeats):
            restorer.synchronize()
            start = time.perf_counter()
            out = restorer.restore(case.text)
            restorer.synchronize()
            latencies.append((time.perf_counter() - start) * 1000.0)
            outputs.append(out)
    except Exception as exc:  # noqa: BLE001 - a failed case must not abort the run
        error = f"{type(exc).__name__}: {exc}"
    return build_record(case, outputs, latencies, error, n_tokens, truncated)


def environment_info(device: str | None) -> dict[str, Any]:
    info: dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "device": device,
        "torch_version": None,
        "transformers_version": None,
        "device_name": None,
        "cuda_version": None,
    }
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_version"] = torch.version.cuda
        if device == "cuda" and torch.cuda.is_available():
            info["device_name"] = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001 - environment capture must never abort the run
        pass
    try:
        import transformers

        info["transformers_version"] = transformers.__version__
    except Exception:  # noqa: BLE001
        pass
    return info


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="G-1 smoke test for the pinned Vietnamese diacritic-restoration baseline (Colab GPU).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="path to the locked RESTORE YAML config")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="root directory for run artifacts")
    parser.add_argument("--run-id", default=None, help="override the timestamped run id")
    parser.add_argument("--device", default=None, choices=("auto", "cpu", "cuda"), help="override the config device")
    parser.add_argument("--repeats", type=int, default=None, help="override the determinism repeat count")
    parser.add_argument("--categories", default=None, help="comma-separated subset of categories to run")
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="print the built-in suite and exit (needs no model, runs anywhere)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.list_cases:
        validate_suite()
        for case in SMOKE_CASES:
            print(f"{case.id}\t{case.category}\t{case.text}")
        print(f"\n{len(SMOKE_CASES)} cases")
        return 0

    cfg = load_config(args.config)
    eng = engineering_settings(cfg)
    cases = select_cases(args.categories)
    repeats = args.repeats if args.repeats is not None else int(cfg["repeats"])
    if repeats < 2:
        raise SystemExit("--repeats must be >= 2 for the determinism check to mean anything")

    # Refuse before creating a run directory: a machine without the experiment
    # stack must not leave a misleading all-errors run behind.
    try:
        require_experiment_dependencies()
    except ExperimentDependenciesMissing as exc:
        print(str(exc), file=sys.stderr)
        return 2

    run_id = args.run_id or make_run_id()
    run_dir = unique_run_dir(Path(args.output_root), run_id)

    print(f"[G-1] config     : {args.config}")
    print(f"[G-1] model      : {cfg['model_id']}")
    print(f"[G-1] revision   : {cfg['revision']}")
    print(f"[G-1] cases      : {len(cases)}  repeats: {repeats}")

    device: str | None = None
    restorer: Restorer | None = None
    load_error: str | None = None
    dtype_matches: bool | None = None
    try:
        device = select_device(args.device or cfg.get("device", "auto"))
        print(f"[G-1] device     : {device}")
        import torch

        torch.manual_seed(int(cfg.get("seed", 42)))
        print("[G-1] loading checkpoint (first run downloads ~1 GB into HF_HOME) ...")
        restorer = Restorer(cfg, device)
        meta = restorer.metadata()
        print(f"[G-1] loaded     : {meta['architecture']}, {meta['num_parameters']:,} params, {meta['parameter_dtype']}")
        resolved = meta.get("resolved_commit_hash")
        if resolved and resolved != cfg["revision"]:
            print(f"[G-1] WARNING    : resolved commit {resolved} != pinned revision {cfg['revision']}")
        # expected_dtype is a diagnostic expectation, never a runtime control: the
        # checkpoint is loaded exactly as stored. Report a mismatch, do not cast.
        expected_dtype = cfg.get("expected_dtype")
        if expected_dtype is not None:
            observed = meta["parameter_dtype"].removeprefix("torch.")
            dtype_matches = observed == str(expected_dtype)
            if not dtype_matches:
                print(
                    f"[G-1] WARNING    : expected_dtype {expected_dtype} != observed {observed} "
                    f"(the checkpoint is loaded as stored; nothing was cast)",
                    file=sys.stderr,
                )
        # Warm up so the first timed case is not dominated by kernel autotuning.
        restorer.restore("xin chao")
        restorer.synchronize()
    except Exception as exc:  # noqa: BLE001 - a load failure is a G-1 result, not a crash
        load_error = f"{type(exc).__name__}: {exc}"
        print(f"[G-1] LOAD FAILED: {load_error}", file=sys.stderr)

    records: list[dict[str, Any]] = []
    if restorer is not None:
        for i, case in enumerate(cases, start=1):
            record = run_case(case, restorer, repeats)
            records.append(record)
            if record["error"]:
                flag = "ERR "
            elif record["rewrite_preserved"] is False:
                flag = "REWR"  # lexical rewrite: this is the one that matters
            elif record["formatting_only_difference"]:
                flag = "fmt "  # capitalisation / final stop only; not a failure
            else:
                flag = "ok  "
            print(f"[{i:>3}/{len(cases)}] {flag} {case.id:<6} {case.category:<20} {record['final_output']}")
    else:
        for case in cases:
            records.append(build_record(case, [], [], load_error or "model not loaded"))

    summary = summarize(records, cfg["model_id"], cfg["revision"])
    status, checks = engineering_status(summary, eng, model_loaded=restorer is not None)

    run_config = {
        "run_id": run_dir.name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script_version": SCRIPT_VERSION,
        "config_path": str(args.config),
        "model": {
            "model_id": cfg["model_id"],
            "revision": cfg["revision"],
            "metadata": restorer.metadata() if restorer is not None else None,
            "load_error": load_error,
        },
        "generation": dict(cfg["generation"]),
        "repeats": repeats,
        "max_input_tokens": (cfg.get("tokenizer") or {}).get("max_input_tokens"),
        "expected_dtype": cfg.get("expected_dtype"),
        "observed_parameter_dtype": (restorer.metadata()["parameter_dtype"] if restorer is not None else None),
        "dtype_matches_expectation": dtype_matches,
        "seed": cfg.get("seed"),
        "environment": environment_info(device),
        "categories_run": sorted({c.category for c in cases}),
        "engineering_thresholds": eng,
        "engineering_status": status,
        "engineering_checks": checks,
    }
    summary["engineering_status"] = status
    summary["engineering_checks"] = checks

    report = render_report(run_config, summary, records, status, checks, eng)
    write_artifacts(run_dir, run_config, records, summary, report)

    print()
    print(format_console_table(summary))
    print()
    print(f"Engineering status: {status}")
    print("(The scientific G-1 decision is the researcher's; see the report's 'G-1 Assessment'.)")
    print()
    print(f"Results: {run_dir}")
    return 0 if status == STATUS_PASS else 1


if __name__ == "__main__":
    raise SystemExit(main())
