# Audit 025 — executable pre-G1 diagnostic runner

| | |
|---|---|
| **Audit id** | 025 |
| **Created (UTC)** | 2026-08-22 |
| **Baseline HEAD** | `7a0c682` (Audit-024 closure) |
| **Scope** | Make the committed pre-G1 CLI executable. **Run nothing.** |
| **Predecessor** | [024](024-preg1-frozen-head-trainer-and-evaluator.md) — all four verification gates closed |
| **Type** | Wiring + tests. No protocol change, no model, no data, no training, no score |

---

## A. VERDICT

**IMPLEMENTATION PASS — RUNNER WIRED; COLAB EXECUTION NOT RUN**

**2 337 local tests pass, 89 skip** (2 312 / 89 before; **+25**, all in
`tests/test_preg1_runner.py`, all ML-free).

No scientific constant changed. **D-B3B0-002 OPEN**, final Stage-2 pooling
**OPEN**, official TEST **SEALED**, PDF **STALE**, Stage-1 untouched. **No new
scientific decision** — wiring committed APIs together is not a decision.

---

## B. WHAT WAS MISSING

Audit 024 closed every verification gate: the library APIs, the real PhoBERT
integration and the real-data boundary all have runtime evidence. But
`scripts/preg1_head_diagnostic.py` validated the protocol and then printed

```
NOT RUN IN THIS BUILD.
```

So the protocol was executable **as a library** and not **as a run**. Nothing
was wrong with it; the last mile simply did not exist.

---

## C. WHAT WAS WIRED

`tune` now performs the ten required steps, entirely out of committed APIs:

| Step | API used |
|---|---|
| load the approved derived TRAIN csv | `load_derived_pool` — enforces the locked SHA, rows and label counts |
| load Audit-023 membership | `load_membership` + `require_partitions` against the real pool |
| materialise both splits **in membership order** | `materialise_split` (new, thin) |
| Vanilla text | `pathway_text` — `canon(x)`, no segmenter |
| load the pinned encoder, frozen and eval | `load_frozen_encoder` |
| extract / cache representations **once per role** | `extract_representations` + `RepresentationCache` |
| 15 head runs | `train_head` |
| select the LR | `select_learning_rate` → `freeze_learning_rate` |
| write the artifact | `tuning_artifact` (new, thin) |

**No second trainer, selector, metric or protocol was created.** A test asserts
the script defines no function named `train_head`, `select_learning_rate`,
`macro_f1`, `accuracy` or `select_checkpoint`.

### Why extraction uses `extract_representations` rather than the bound wrapper

9 139 rows do not fit one forward pass. `extract_bound_representations` binds a
key describing a **whole** split, so calling it per chunk would build keys whose
`count` was wrong. The runner therefore chunks with `extract_representations` —
the committed primitive `extract_bound_representations` itself calls, which
checks the encoder is frozen and in eval, runs under `no_grad`, pools position 0
and returns detached FP32 — then binds provenance **once**, over the full
tensor. Batch size affects nothing the key describes.

### Boundaries made structural, not documented

- **Pathway** is a module constant `TUNING_PATHWAY = VANILLA`; there is no
  argument in the tune path that could carry `BASE_ONLY`, and an AST test
  asserts the string never appears in `run_tune` or its helpers.
- **Roles** are `TUNING_ROLES = (PROTOCOL_TRAIN, PROTOCOL_DEV)`;
  `materialise_split` raises on anything else. `tune` has no
  `--official-validation` flag; `measure` does.
- **Official TEST** is unreachable — no enum member, no attribute access, no
  flag (asserted by AST, not by grep: the docstring legitimately *names* the
  absence).
- **No CLI override of a locked value.** `--seeds`, `--grid`, `--epochs`,
  `--batch-size`, `--max-length`, `--padding`, `--pooling` all absent; only
  resource paths (`--derived-train`, `--split-dir`, `--cache-root`,
  `--output-dir`) and provenance (`--repository-head`) are accepted.
  `--revision` exists but is **validated equal to the pinned revision** and
  refuses anything else, because D-B3B0-002 is OPEN.
- **`measure`** still requires `--frozen-lr`, and **does not execute a
  measurement**: a test asserts `run_measure` calls neither `train_head` nor
  `score_measurement`.

### The tuning artifact

Deterministic — no timestamp, hostname, path or elapsed time; the same inputs
and seeds produce the same bytes. It records the execution HEAD, the full
protocol identity, both representation keys, **all 15 runs**, the per-LR
aggregates, the selected LR and the selection/tie-break rules. **Ids, counts and
digests only — no raw text**, asserted by walking the serialised keys and
scanning for Vietnamese-marked characters. Output lands under `results/…`, which
`.gitignore:15 results/**` covers, and the runner **refuses an existing output
directory**.

### Progress output

Cache HIT/MISS with shapes, extraction progress, per-run `lr`/`seed`, epoch
progress every 5 epochs with dev macro-F1 and accuracy, the selected epoch per
run, a per-LR summary table, and the final selected LR.

---

## D. SCIENTIFIC CONTRACT — UNCHANGED

Verified from the modules after the change: LR grid `(1e-4, 3e-4, 1e-3, 3e-3,
1e-2)`; tuning seeds `(5509, 19422, 11800)`; measurement seeds `(53148, 59945,
42941, 720, 9428)`; epochs 30; batch 128; `max_length` 256; padding
`max_length`; pooling `FIRST_TOKEN`; revision
`01daacda68afe13d83023d16ec647239e344a1e6`.

The schedule is **derived**, not typed: an AST test asserts `tuning_schedule`
references `LR_GRID` and `TUNING_SEEDS` and contains **no numeric literal at
all**, and a second test asserts no locked value appears as a literal anywhere
in the script. 5 × 3 = 15 because the protocol says so.

---

## E. TESTS

**25 new tests**, all ML-free and all executing locally: import-without-torch;
exactly 5 × 3 distinct runs; schedule derived from the protocol; no restated
constants; Vanilla-only; `BASE_ONLY` absent from the tune path; roles limited to
protocol-train/dev; `materialise_split` refuses official validation; `tune` has
no official-validation flag; official TEST unreachable; no locked-value
override; `measure` requires a frozen LR and executes nothing; representation
key carries the locked contract and pins membership order; extraction happens
once per role and a cache hit short-circuits it; committed APIs reused and no
second trainer defined; the artifact records all 15 runs, the selector's choice,
provenance and boundaries, is JSON-safe, deterministic, free of raw text and of
runtime-varying fields; wrong revision refused; existing output directory
refused.

**Three boundary mutations were injected to confirm the guards are live:**
switching `TUNING_PATHWAY` to `BASE_ONLY` → 3 failures; adding
`OFFICIAL_VALIDATION` to `TUNING_ROLES` → 3 failures; adding an `--epochs`
override → 1 failure. All pass again after restoring.

**Two of my own tests were defective and were repaired**, both the same
prose-matching trap: one grepped for `"OFFICIAL_TEST"` and hit the docstring
explaining its absence; the other compared first *textual* occurrences and so
measured a lazy import rather than the call order. Both are now AST/structural.

---

## F. WHAT HAS NOT HAPPENED

**No Colab execution.** No model loaded, no corpus read, no representation
extracted, no head trained, no LR tuned, no downstream score. Official TEST
untouched. The local suite ran only ML-free tests.

---

## G. NEXT STEP

**Only after researcher review and commit**: run `tune` on Colab —

```
python scripts/preg1_head_diagnostic.py tune \
    --derived-train <derived_train.csv> --split-dir <preg1-split-v1-…> \
    --text-column text --label-column label --id-column id \
    --cache-root <cache> --output-dir <results/preg1-tune/…> \
    --repository-head <sha>
```

**15 Vanilla LR-tuning runs** (5 learning rates × 3 tuning seeds), frozen
encoder, protocol-dev selection only. It produces a frozen LR and **no
downstream score**. The paired measurement stays unwired until that LR is
reviewed.

---

## H. SELF-AUDIT

| # | Check | Result |
|---|---|---|
| 1 | Audit 025 created and persisted | **yes** |
| 2 | CLI is executable; `NOT RUN IN THIS BUILD` removed from `tune` | **yes** |
| 3 | Exactly 5 × 3 = 15 runs scheduled, derived from the protocol | **yes** — AST-asserted, no literals |
| 4 | Tuning is VANILLA-only | **yes** — module constant; `BASE_ONLY` absent from the tune path |
| 5 | `tune` cannot read official validation | **yes** — no flag, roles restricted, `materialise_split` raises |
| 6 | Official TEST unreachable | **yes** — AST, not grep |
| 7 | No CLI override of a locked scientific value | **yes** — `--revision` validated equal to the pin |
| 8 | Representations extracted once per role and reused across 15 runs | **yes** — cache hit short-circuits; asserted on call order |
| 9 | Selected LR comes from the committed selector | **yes** — `select_learning_rate` → `freeze_learning_rate` |
| 10 | Artifact records all 15 runs, aggregates, selection and provenance | **yes** |
| 11 | Artifact is deterministic and holds no raw text | **yes** |
| 12 | Artifacts land in a gitignored location; overwrite refused | **yes** — `.gitignore:15 results/**` verified |
| 13 | `measure` still gated on `--frozen-lr` and executes nothing | **yes** |
| 14 | No second trainer / selector / metric / protocol | **yes** — AST-asserted |
| 15 | No scientific constant changed | **yes** — re-verified from the modules |
| 16 | No production library file changed | **yes** — only the script and a new test file |
| 17 | Guards mutation-verified | **yes** — three injected violations, all caught |
| 18 | No model, data, training, LR tuning or score executed | **yes** |
| 19 | D-B3B0-002 OPEN; Stage-2 pooling OPEN; PDF STALE; Stage-1 untouched | **yes** |
| 20 | No new scientific decision | **yes** |
| 21 | Tests | **2 337 passed, 89 skipped**; new file 25 passed |
| 22 | `git diff --check` clean; everything unstaged; no prohibited git operation | **yes** |

### Limitations

1. **The runner has never executed.** Every test here is structural or
   arithmetic; the encoder pass, the 15 training runs and the artifact write
   have not run once. Audit 024 is the standing reminder that a real run finds
   what fixtures do not.
2. **`ENCODER_BATCH = 32` is a runtime knob I chose.** It cannot change the
   representations — batching is mathematically irrelevant to a frozen
   per-example forward — but it is untested against real memory limits.
3. **`materialise_split` builds an id→row dict over 11 424 rows.** Fine at this
   scale, and not something the local suite exercises at real size.
