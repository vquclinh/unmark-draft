#!/usr/bin/env python3
"""Measurements the PRE-TRAIN no-update smoke must take. **Changes no protocol.**

Audit 030 §S. An external review raised a plausible but **unmeasured** concern
about validation cost. This script produces the evidence that concern needs,
and deliberately changes nothing: the eval cadence stays 500, the dev set stays
whole, the four conditions stay, and the selection metric stays. Nothing here
decides anything — it measures, so that a later decision can be made on numbers
instead of on intuition.

Two modes:

    --profile      descriptive profile of the prepared corpus. **No model.**
    --validation   one full four-condition validation pass through the
                   **authoritative** `validation.evaluate` -- real encoder and
                   adapter forwards, all 4 locked conditions, batch 128, the
                   whole dev set, `no_grad`, CUDA-synchronised timing, and
                   **zero parameter updates**, proven by hashing every parameter
                   before and after. **This loads the real model.**

Clean-reference h(x) timing is reported by `--validation`: `evaluate` calls
`reference_representation` once per batch per condition, so the
candidate-invariant work a frozen cache would remove is measured by observing
that call rather than by a second forward implementation. **No cache is
implemented** (Audit 030 §S.6).

**No optimizer is constructed and no parameter is updated by any path here** --
AST-asserted, and enforced at runtime by parameter hashing that fails closed.

    python scripts/stage1_pretrain_measurements.py --profile \\
        --prepared-corpus /content/unmark-stage1-prepared-aa49785eadcb \\
        --completion-dir  /content/drive/.../stage1-checkpoints/aa49785eadcb
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from unmark.stage1.checkpoint import verify_prepared_corpus  # noqa: E402
from unmark.stage1.manifest import CHUNKS_NAME  # noqa: E402
from unmark.stage1.protocol import (  # noqa: E402
    BATCH_SIZE,
    ENCODER_CHECKPOINT,
    ENCODER_REVISION,
    EVAL_EVERY_UPDATES,
    MAX_LENGTH,
    PRECISION,
    VALIDATION_CONDITIONS,
    VALIDATION_CORRUPTION_SEED,
)

PERCENTILES = (25, 50, 75, 90, 95, 99)

SAMPLING_SCHEME_VERSION = "partition-local-stride-v2"
"""Partition-aware deterministic sampling (Audit 030 §U.3).

**v1 was reproducible but weak.** It advanced one counter shared across both
partitions and kept every 97th line, so dev -- ~0.43 % of the payload -- yielded
about **117 of 11 443** while train filled its cap. Reproducible, but thin
evidence for a descriptive profile.

**v2 counts within each partition.** Each partition gets its own index, and the
stride is computed from that partition's own population and the requested count:

    stride = max(1, population // requested)

so a partition smaller than the cap is taken **whole**. On the real corpus dev
is 11 443 < 20 000, so **every dev chunk is profiled**; train takes exactly
20 000, spread evenly across its full fixed order rather than clustered.

Still deterministic and still data-independent: the choice depends only on
partition-local position, never on token length, text, corruption outcome,
observed statistics or labels. No RNG, no seed, and no `hash()`. The population
is known only after one pass, so selection runs as a **second** streaming pass --
memory stays bounded either way.
"""


def percentile(values: list[int], q: int) -> int:
    """Nearest-rank, on an already-sorted list. No numpy dependency."""
    if not values:
        return 0
    rank = max(1, (q * len(values) + 99) // 100)
    return values[min(rank, len(values)) - 1]


def describe(values: list[int]) -> dict:
    values = sorted(values)
    total = len(values)
    return {
        "count": total,
        **{f"p{q}": percentile(values, q) for q in PERCENTILES},
        "max": values[-1] if values else 0,
        "mean": round(sum(values) / total, 2) if total else 0.0,
        "fraction_le_32": round(sum(v <= 32 for v in values) / total, 6) if total else 0.0,
        "fraction_le_64": round(sum(v <= 64 for v in values) / total, 6) if total else 0.0,
        "fraction_le_128": round(sum(v <= 128 for v in values) / total, 6) if total else 0.0,
    }


def profile(prepared: pathlib.Path, sample: int) -> dict:
    """Descriptive profile of the prepared corpus, train and dev separately.

    **On "recorded lengths".** The prepared payload does **not** carry
    `reference_length` / `base_length`: `chunk_record` persists `chunk_id`,
    `document_id`, `partition`, `chunk_index`, `text`, `source_start`,
    `source_end` and `source_shard`, and nothing else. Both lengths were computed
    and enforced during Stage 6 -- that is how `overflow_count = 0` was
    established -- but they were never written down.

    So a token-length profile cannot be read off the artifact. It can only be
    **recomputed** with the pinned tokenizer, which is a full re-tokenization
    pass over 2 633 067 chunks. Adding the lengths to the payload would change
    the artifact bytes and require re-running Stage 6, which is forbidden.

    This function therefore reports two different things, labelled as such:

    * **exact and free** -- character lengths and chunks-per-parent, straight
      from the payload, over every chunk;
    * **recomputed on a sample** (`--sample`) -- token lengths on both pathways,
      clearly marked `recomputed_not_recorded`.

    It streams the file rather than materialising it (Audit 030 F4), so it costs
    a few MB regardless of corpus size.
    """
    per_partition: dict[str, dict] = {}
    chunks_per_parent: dict[str, dict[str, int]] = {"train": {}, "dev": {}}
    characters: dict[str, list[int]] = {"train": [], "dev": []}
    sampled: dict[str, list[str]] = {"train": [], "dev": []}
    sampled_ids: dict[str, list[str]] = {"train": [], "dev": []}
    seen = 0

    # Pass 1 -- exact statistics over every chunk, and the populations the
    # partition-local strides are derived from.
    with open(prepared / CHUNKS_NAME, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            part = row["partition"]
            characters[part].append(len(row["text"]))
            counts = chunks_per_parent[part]
            counts[row["document_id"]] = counts.get(row["document_id"], 0) + 1
            seen += 1

    population = {part: len(values) for part, values in characters.items()}
    stride = {
        part: max(1, population[part] // sample) if sample else 0
        for part in population
    }

    # Pass 2 -- partition-local selection. A partition smaller than the cap is
    # taken whole, because its stride is 1.
    if sample:
        index = {"train": 0, "dev": 0}
        with open(prepared / CHUNKS_NAME, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                part = row["partition"]
                position = index[part]
                index[part] += 1
                if len(sampled[part]) < sample and position % stride[part] == 0:
                    sampled[part].append(row["text"])
                    sampled_ids[part].append(row["chunk_id"])

    for part in ("train", "dev"):
        per_partition[part] = {
            "chunks": len(characters[part]),
            "parent_documents": len(chunks_per_parent[part]),
            "characters_per_chunk": describe(characters[part]),
            "chunks_per_parent": describe(list(chunks_per_parent[part].values())),
        }
    return {
        "measurement": "prepared_corpus_profile",
        "chunks_seen": seen,
        "sample_selection": {
            "sampling_scheme_version": SAMPLING_SCHEME_VERSION,
            "selection_method": (
                "partition-local index; keep position % stride == 0 where "
                "stride = max(1, population // requested); a partition smaller "
                "than the cap is taken whole"
            ),
            "deterministic": True,
            "seed": None,
            "data_independent": True,
            "per_partition": {
                part: {
                    "partition": part,
                    "population_count": population[part],
                    "requested_count": sample,
                    "obtained_count": len(sampled[part]),
                    "stride": stride[part],
                    "complete_population": len(sampled[part]) == population[part],
                }
                for part in ("train", "dev")
            },
        },
        "note": (
            "reference_length/base_length are NOT persisted in chunks.jsonl; "
            "character lengths and chunks-per-parent are exact and cover every "
            "chunk, token lengths are recomputed on a sample and marked as such"
        ),
        "partitions": per_partition,
        "_sampled_texts": sampled,
        "_sampled_ids": sampled_ids,
    }


def token_profile(sampled: dict[str, list[str]], revision: str) -> dict:
    """Token lengths on both pathways, **recomputed** on the sample."""
    from transformers import AutoTokenizer  # noqa: PLC0415

    from unmark.stage1.lengths import build_length_functions  # noqa: PLC0415

    tokenizer = AutoTokenizer.from_pretrained(ENCODER_CHECKPOINT, revision=revision)
    reference_length, base_length, _ = build_length_functions(tokenizer)

    try:
        import transformers  # noqa: PLC0415

        transformers_version = transformers.__version__
    except Exception:  # noqa: BLE001 - identity is reported, never load-bearing
        transformers_version = None

    out: dict = {
        "measurement": "token_length_profile",
        "recomputed_not_recorded": True,
        "max_length": MAX_LENGTH,
        # Everything needed to reproduce these numbers exactly.
        "tokenizer_checkpoint": ENCODER_CHECKPOINT,
        "tokenizer_revision": revision,
        "tokenizer_revision_is_locked": revision == ENCODER_REVISION,
        "transformers_version": transformers_version,
        "pathways": ["reference", "base"],
        "sampling_scheme_version": SAMPLING_SCHEME_VERSION,
    }
    for part, texts in sampled.items():
        if not texts:
            continue
        started = time.monotonic()
        reference = [reference_length(t) for t in texts]
        base = [base_length(t) for t in texts]
        out[part] = {
            "sampled_chunks": len(texts),
            "reference": describe(reference),
            "base": describe(base),
            "over_max_length": sum(v > MAX_LENGTH for v in reference + base),
            "seconds": round(time.monotonic() - started, 2),
        }
    return out


class InstrumentedObjective:
    """A transparent proxy that **counts and times** the real objective.

    The whole point of Audit 030 §T was that the previous tool reimplemented
    nothing and measured nothing. This measures without reimplementing: every
    call is delegated to the real objective, so `validation.evaluate` runs the
    **authoritative** path -- its own `objective.eval()`, its own
    `torch.no_grad()`, its own batching, distance, pooling and aggregation --
    and this only observes it.

    Separating `reference_representation` from `adapted_representation` is what
    makes the clean-reference question answerable: the reference pathway is the
    candidate-invariant one a frozen cache would remove, and `evaluate` calls it
    once per batch per condition.
    """

    def __init__(self, objective, synchronize=None):
        self._objective = objective
        self._synchronize = synchronize or (lambda: None)
        self.reference_calls = 0
        self.adapted_calls = 0
        self.reference_seconds = 0.0
        self.adapted_seconds = 0.0
        self.eval_calls = 0
        self.grad_enabled_during_forward = False
        self.outputs_requiring_grad = 0

    # -- the surface `evaluate` uses -------------------------------------
    def eval(self):
        self.eval_calls += 1
        return self._objective.eval()

    def train(self, mode: bool = True):  # pragma: no cover - never used by evaluate
        return self._objective.train(mode)

    def reference_representation(self, *args, **kwargs):
        return self._timed("reference", self._objective.reference_representation, args, kwargs)

    def adapted_representation(self, *args, **kwargs):
        return self._timed("adapted", self._objective.adapted_representation, args, kwargs)

    def __getattr__(self, name):
        return getattr(self._objective, name)

    def _timed(self, kind, function, args, kwargs):
        import torch

        # Recorded, not assumed: `evaluate` is supposed to hold `no_grad`.
        if torch.is_grad_enabled():
            self.grad_enabled_during_forward = True
        self._synchronize()
        began = time.perf_counter()
        result = function(*args, **kwargs)
        self._synchronize()
        elapsed = time.perf_counter() - began
        if getattr(result, "requires_grad", False):
            self.outputs_requiring_grad += 1
        if kind == "reference":
            self.reference_calls += 1
            self.reference_seconds += elapsed
        else:
            self.adapted_calls += 1
            self.adapted_seconds += elapsed
        return result


def parameter_digest(module) -> dict:
    """sha256 over every parameter, split trainable vs frozen encoder."""
    import hashlib

    import torch

    trainable = hashlib.sha256()
    frozen = hashlib.sha256()
    counts = {"trainable_tensors": 0, "frozen_tensors": 0,
              "trainable_parameters": 0, "frozen_parameters": 0}
    for name, parameter in sorted(module.named_parameters(), key=lambda kv: kv[0]):
        payload = parameter.detach().to("cpu", torch.float64).numpy().tobytes()
        if parameter.requires_grad:
            trainable.update(name.encode()); trainable.update(payload)
            counts["trainable_tensors"] += 1
            counts["trainable_parameters"] += parameter.numel()
        else:
            frozen.update(name.encode()); frozen.update(payload)
            counts["frozen_tensors"] += 1
            counts["frozen_parameters"] += parameter.numel()
    return {"trainable_sha256": trainable.hexdigest(),
            "frozen_encoder_sha256": frozen.hexdigest(), **counts}


def validation_timing(
    prepared: pathlib.Path,
    revision: str,
    *,
    require_cuda: bool = False,
    build=None,
    loader=None,
) -> dict:
    """Time ONE full four-condition validation pass through the REAL evaluator.

    Repairs the Audit 030 §T defect. It executes `validation.evaluate` -- the
    same function `execute_stage` hands to `train_run` as `evaluate_fn` -- so
    the numbers describe the work training actually repeats, and no second
    evaluator exists.

    **One-time vs recurring**, the distinction §T.1's predecessor got wrong:
    `execute_stage` builds `prepared_by_condition` **once** before the run loop
    and reuses it for every evaluation, so condition preparation is *setup*.
    Only `evaluate` recurs, every `EVAL_EVERY_UPDATES` updates. They are timed
    and reported separately, and the projection uses the recurring figure alone.

    `build` and `loader` are injection points for tests. In production they are
    the repository's own `build_objective` and `load_prepared_chunks`; nothing
    else is substitutable, so a test cannot accidentally certify a fake path.
    """
    import torch

    from unmark.linguistics import make_classifier, try_load_inventory
    from unmark.stage1.execute import build_objective, lambdas_to_weights, load_prepared_chunks
    from unmark.stage1.trainer import verify_model_contract
    from unmark.stage1.validation import HeldOutExample, evaluate, prepare_condition_batch

    build = build or build_objective
    loader = loader or load_prepared_chunks

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if require_cuda and device.type != "cuda":
        raise RuntimeError(
            "--require-cuda was given but CUDA is unavailable. Refusing to "
            "report a CPU number as a GPU measurement."
        )
    on_cuda = device.type == "cuda"
    synchronize = torch.cuda.synchronize if on_cuda else (lambda: None)

    began = time.perf_counter()
    _, dev_text = loader(prepared)
    corpus_load_seconds = time.perf_counter() - began

    tokenizer, unmark_encoder, objective_cls = build(revision)
    contract = verify_model_contract(unmark_encoder)
    unmark_encoder.to(device)
    objective = objective_cls(unmark_encoder, lambdas_to_weights(1.0))
    classifier = make_classifier(try_load_inventory())
    held_out = [HeldOutExample(cid, text) for cid, text in sorted(dev_text.items())]

    before = parameter_digest(unmark_encoder)

    # ---- ONE-TIME setup: the held-out realisation, built once and reused ----
    began = time.perf_counter()
    prepared_by_condition = {
        condition: prepare_condition_batch(
            held_out, tokenizer, condition, truncation=None, classifier=classifier
        )
        for condition in VALIDATION_CONDITIONS
    }
    setup_seconds = time.perf_counter() - began

    # ---- RECURRING work: exactly what happens every 500 updates ------------
    proxy = InstrumentedObjective(objective, synchronize=synchronize)
    if on_cuda:
        torch.cuda.reset_peak_memory_stats()
    synchronize()
    began = time.perf_counter()
    point = evaluate(
        proxy, prepared_by_condition, tokenizer.pad_token_id, batch_size=BATCH_SIZE
    )
    synchronize()
    recurring_seconds = time.perf_counter() - began

    after = parameter_digest(unmark_encoder)
    identical = (before["trainable_sha256"] == after["trainable_sha256"]
                 and before["frozen_encoder_sha256"] == after["frozen_encoder_sha256"])

    forwards = proxy.reference_calls + proxy.adapted_calls
    batches = -(-len(held_out) // BATCH_SIZE)

    report: dict = {
        "measurement": "validation_wall_clock",
        "environment": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(0) if on_cuda else None,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda if on_cuda else None,
            "transformers_version": _transformers_version(),
            "cuda_synchronized_around_timing": on_cuda,
        },
        "inputs": {
            "dev_chunks": len(held_out),
            "batch_size": BATCH_SIZE,
            "batches_per_condition": batches,
            "conditions": list(VALIDATION_CONDITIONS),
            "validation_corruption_seed": VALIDATION_CORRUPTION_SEED,
            "precision": PRECISION,
        },
        "timing_seconds": {
            "corpus_verification": None,     # filled by the caller
            "prepared_corpus_load": round(corpus_load_seconds, 2),
            "one_time_condition_setup": round(setup_seconds, 2),
            "recurring_validation_total": round(recurring_seconds, 3),
            "per_condition_estimate": round(recurring_seconds / len(VALIDATION_CONDITIONS), 3),
        },
        "clean_reference": {
            "timing_available": True,
            "reference_forward_seconds": round(proxy.reference_seconds, 3),
            "adapted_forward_seconds": round(proxy.adapted_seconds, 3),
            "reference_calls": proxy.reference_calls,
            "adapted_calls": proxy.adapted_calls,
            "note": (
                "`evaluate` recomputes the clean reference h(x) once per batch "
                "per condition; that is the candidate-invariant work a frozen "
                "cache would remove. Reported, not cached (Audit 030 §S.6)."
            ),
        },
        "forward_passes": forwards,
        "conditions_executed": sorted(point.distances),
        "no_update_boundary": {
            "optimizer_constructed": False,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "parameter_updates": 0,
            "grad_enabled_during_forward": proxy.grad_enabled_during_forward,
            "outputs_requiring_grad": proxy.outputs_requiring_grad,
            "parameter_hash_before": before,
            "parameter_hash_after": after,
            "parameters_identical": identical,
        },
        "model_contract": contract,
    }
    if on_cuda:
        report["gpu_memory_bytes"] = {
            "peak_allocated": int(torch.cuda.max_memory_allocated()),
            "peak_reserved": int(torch.cuda.max_memory_reserved()),
            "current_allocated": int(torch.cuda.memory_allocated()),
            "current_reserved": int(torch.cuda.memory_reserved()),
        }

    report["failures"] = validation_failures(report)
    report["status"] = "PASS" if not report["failures"] else "FAIL"
    return report


def validation_failures(report: dict) -> list[str]:
    """Fail closed on a measurement that did not actually measure (B7).

    The old defect reported success while running zero forwards. Nothing here
    may report success unless the work provably happened.
    """
    failures: list[str] = []
    if report.get("forward_passes", 0) <= 0:
        failures.append("no forward pass executed; this is not a validation measurement")
    executed = set(report.get("conditions_executed") or ())
    missing = [c for c in VALIDATION_CONDITIONS if c not in executed]
    if missing:
        failures.append(f"conditions not executed: {missing}")
    boundary = report.get("no_update_boundary") or {}
    if boundary.get("grad_enabled_during_forward"):
        failures.append("gradients were enabled during a forward; no_grad was not active")
    if boundary.get("outputs_requiring_grad"):
        failures.append("a forward produced a tensor requiring grad")
    if not boundary.get("parameters_identical"):
        failures.append("PARAMETERS CHANGED during a no-update measurement")
    if boundary.get("optimizer_constructed"):
        failures.append("an optimizer was constructed")
    if boundary.get("backward_calls") or boundary.get("optimizer_steps"):
        failures.append("a backward or optimizer step occurred")
    environment = report.get("environment") or {}
    if str(environment.get("device", "")).startswith("cuda") and not environment.get(
        "cuda_synchronized_around_timing"
    ):
        failures.append("CUDA timing was not synchronized; wall-clock would be false")
    return failures


def _transformers_version():
    try:
        import transformers

        return transformers.__version__
    except Exception:  # noqa: BLE001 - identity only
        return None


def build_report(args) -> dict:
    prepared = pathlib.Path(args.prepared_corpus)
    completion = (pathlib.Path(args.completion_dir) if args.completion_dir
                  else prepared / "_checkpoint")
    began = time.perf_counter()
    verified = verify_prepared_corpus(prepared, completion)
    verification_seconds = round(time.perf_counter() - began, 2)

    report: dict = {
        "prepared_corpus_verified": True,
        "chunk_membership_digest": verified.chunk_membership_digest,
        "counts": verified.counts,
        "encoder": ENCODER_CHECKPOINT,
        "revision": args.revision,
        "measurements": {},
    }
    if args.profile:
        result = profile(prepared, args.sample)
        sampled = result.pop("_sampled_texts")
        result.pop("_sampled_ids")
        report["measurements"]["profile"] = result
        if args.tokens:
            report["measurements"]["token_profile"] = token_profile(sampled, args.revision)
    if args.validation:
        measured = validation_timing(
            prepared, args.revision, require_cuda=args.require_cuda
        )
        measured["timing_seconds"]["corpus_verification"] = verification_seconds
        # The projection the cost question actually needs, using the RECURRING
        # figure only -- never the one-time setup.
        recurring = measured["timing_seconds"]["recurring_validation_total"]
        measured["projection"] = {
            "eval_every_updates": EVAL_EVERY_UPDATES,
            "evaluations_per_20k_run": 20_000 // EVAL_EVERY_UPDATES + 1,
            "recurring_seconds_per_evaluation": recurring,
            "projected_validation_seconds_per_20k_run": round(
                recurring * (20_000 // EVAL_EVERY_UPDATES + 1), 1
            ),
            "note": "one-time condition setup is NOT multiplied into this figure",
        }
        report["measurements"]["validation"] = measured
        report["status"] = measured["status"]
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-corpus", required=True)
    parser.add_argument("--completion-dir", default=None)
    parser.add_argument("--revision", default=ENCODER_REVISION)
    parser.add_argument("--profile", action="store_true",
                        help="descriptive profile; needs no model")
    parser.add_argument("--tokens", action="store_true",
                        help="also recompute token lengths on the sample (loads the tokenizer)")
    parser.add_argument("--sample", type=int, default=20_000,
                        help="chunks per partition for the recomputed token profile")
    parser.add_argument("--validation", action="store_true",
                        help="time one full four-condition validation pass through "
                             "the authoritative evaluator; loads the real model")
    parser.add_argument("--require-cuda", action="store_true",
                        help="refuse to report a CPU number as a GPU measurement")
    args = parser.parse_args(argv)

    if not (args.profile or args.validation):
        parser.error("choose at least one of --profile / --validation")

    report = build_report(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("status", "PASS") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
