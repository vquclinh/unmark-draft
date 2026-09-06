# Audit 050 - Stage-2 Dual-Finalist Infrastructure Implementation

**Scope:** static implementation acceptance for the Stage-2 frozen UNMARK
dual-finalist pathway and downstream corruption/input path required by Audit
049. Runtime acceptance with torch, the real finalist checkpoints and the real
pinned PhoBERT model remains pending.
**Date:** 2026-09-06
**Type:** implementation audit. No scientific protocol change. No runtime
artifact acceptance.

---

## 1. Executive verdict

**PASS WITH RUNTIME EVIDENCE PENDING - structural / torch-free infrastructure
review passed; authoritative real-torch artifact acceptance is pending.**

This audit implements the two gaps Audit 049 deliberately left open:

1. a frozen UNMARK downstream representation pathway for `UNMARK-A` and
   `UNMARK-B`;
2. a downstream corruption/input path for `FULL`, `P25`, `P50`, `P75`, `P100`
   and `STRIP_ALL`.

No classification head was trained. No UIT-VSFC row was read. No
measurement-dev result was inspected. Official TEST remains sealed. No A/B
selection, winner rule, margin, p-value or significance test was added.

This audit does not claim authoritative runtime acceptance. The local
environment has no torch, 15 focused runtime-critical tests were skipped, no
real finalist checkpoint was loaded and no real pinned PhoBERT forward was
executed.

---

## 2. What was already frozen

Audit 049 and D-S2-001 remain the protocol authority. This implementation does
not revise:

* the two Stage-1 finalists;
* the `UNMARK-A` / `UNMARK-B` arm count;
* the prohibition on selecting A vs B;
* dataset, split, head, optimizer, seed, pooling or reporting constants;
* the sealed TEST state.

`docs/spec/decisions.md` remains append-only and was not edited in this audit.
No new scientific decision was created.

---

## 3. Files changed

Implementation:

* `unmark/evaluation/stage2_dual_finalist.py`

Tests:

* `tests/test_stage2_dual_finalist_infra.py`
* `tests/test_preg1_import_contract.py`

Audit:

* `docs/audits/050-stage2-dual-finalist-infrastructure-implementation.md`

No checkpoint, dataset, cache, model-weight or binary artifact was added.

---

## 4. Implementation contracts

### 4.1 Pathway identity

`Stage2UnmarkArm` has exactly two values:

* `UNMARK-A` -> `FINALIST_A`
* `UNMARK-B` -> `FINALIST_B`

Unknown arms, third finalists and ensemble-style names fail closed. The legacy
`SystemPathway` enum remains untouched, preserving the existing VANILLA /
BASE_ONLY pre-G1 diagnostic contract.

### 4.2 Checkpoint binding

`load_frozen_unmark_pathway(...)` requires an explicit checkpoint path. There is
no hard-coded Drive path and no default checkpoint location.

The loader delegates to `unmark.stage1.finalists.verify_finalist_checkpoint`.
It does not duplicate the checkpoint verifier. The existing Stage-1 verifier
therefore remains authoritative for:

* bound checkpoint SHA-256;
* Stage-1 run provenance;
* update;
* adapter state key set;
* adapter tensor count;
* adapter FP32 dtype;
* finite adapter tensors;
* adapter parameter count.

After verification, the implementation loads only `adapter_state` into the exact
Stage-1 adapter architecture with `strict=True`.

### 4.3 Freezing

After load:

* encoder parameters have `requires_grad=False`;
* adapter parameters have `requires_grad=False`;
* encoder is in `eval()` mode;
* adapter is in `eval()` mode;
* encoder revision and checkpoint identity are checked against the frozen
  downstream PhoBERT pin when an encoder object is supplied;
* adapter key set and parameter count are rechecked on the constructed module;
* a helper for future head parameter enumeration returns head parameters only
  and refuses leakage of encoder/adapter parameter objects.

### 4.4 Representation

`extract_stage2_unmark_representations(...)` builds adapted input embeddings
from:

* frozen encoder word embeddings over the base-token ids;
* Stage-1 tone ids/masks;
* Stage-1 letter ids/masks;
* the existing authoritative PhoBERT position-id path.

It returns detached FP32 first-token representations shaped `[batch, 768]`.
Stage-1 masked-mean pooling is not used.

### 4.5 Downstream corruption/input path

`prepare_stage2_unmark_input(...)` calls the existing corruption API:

```
corrupt(text, condition, seed, sample_id, purpose=SCIENTIFIC)
```

The production default is `SCIENTIFIC`; tests explicitly use `SELF_CHECK`
because the local environment does not provision external scientific inputs.

For each sample/condition it:

1. canonicalizes the source text;
2. corrupts through `unmark.corruption.corrupt`;
3. projects both clean and corrupted strings through existing Stage-1
   `project_text`;
4. refuses if `b(C(x)) != b(x)`;
5. refuses if base token ids or projection counts differ;
6. constructs the downstream UNMARK input from the invariant base ids and the
   corrupted orthographic side-channel state;
7. applies the frozen downstream max-length truncation consistently to content
   ids and channel projections;
8. pads batches to the frozen `max_length` at collation.

`VARIANT` remains fail-closed through the existing condition registry.

---

## 5. Tests

Focused new tests:

```
pytest -q tests/test_stage2_dual_finalist_infra.py
```

Result:

```
16 passed, 15 skipped
```

The 15 skips include the runtime-critical tensor/checkpoint/freezing/
representation checks. Therefore UNMARK-A/B real checkpoint loading, pinned
PhoBERT forward execution, freezing, FP32/finite/detached representation
properties and first-token runtime extraction are not yet empirically accepted
in the authoritative environment.

Importer contract:

```
pytest -q tests/test_preg1_import_contract.py
```

Result:

```
15 passed, 1 skipped
```

The skip is the committed-tree check for the newly added, uncommitted importer.

Relevant regression set:

```
pytest -q tests/test_corruption.py tests/test_evaluation_harness.py \
  tests/test_preg1_head.py tests/test_preg1_runner.py \
  tests/test_preg1_import_contract.py tests/test_stage2_dual_finalist_infra.py \
  tests/test_stage1_finalist_freeze.py tests/test_stage1_finalist_checkpoint_torch.py \
  tests/test_adapter_contract.py tests/test_neural_adapter.py tests/test_stage1.py \
  tests/test_stage1_validation_preparation.py tests/test_channel_projection.py \
  tests/test_alignment_contracts.py
```

Result:

```
1368 passed, 91 skipped
```

Full repository, excluding the sandbox-blocked multiprocessing file:

```
pytest -q --ignore=tests/test_stage1_parallel.py
```

Result:

```
4168 passed, 124 skipped
```

Stage-1 parallel tests outside the restricted sandbox:

```
pytest -q tests/test_stage1_parallel.py
```

Result:

```
18 passed
```

An initial full `pytest -q` run inside the restricted sandbox failed seven
`tests/test_stage1_parallel.py` cases with `PermissionError: [Errno 1]
Operation not permitted` while Python's forkserver attempted to bind a local
AF_UNIX socket. The same file passed when rerun outside the restricted sandbox.
The only non-environmental full-suite failure was the PREG1 importer registry;
it was fixed by adding `unmark/evaluation/stage2_dual_finalist.py` to
`tests/test_preg1_import_contract.py`.

---

## 6. Diff hygiene

```
git diff --check
```

produced no output.

The three new untracked files were also checked with `git diff --no-index
--check /dev/null <path>`; those commands produced no whitespace diagnostics
and returned non-zero only because each file is new.

---

## 7. Limitations

Runtime acceptance is PENDING, not PASS. Real finalist `.pt` files were not
present in this repository and were not copied or mutated. No real finalist A
checkpoint was verified or loaded. No real finalist B checkpoint was verified
or loaded. Torch is unavailable in the local sandbox, so tensor-level Stage-2
tests skipped locally; they are written against synthetic checkpoints and
should run where torch is installed.

The real pinned PhoBERT checkpoint was not downloaded or loaded. No real
PhoBERT forward was executed. No authoritative runtime `[batch, 768]`
representation evidence exists yet. No network access was needed for this
implementation audit.

No representation cache schema was added for Stage-2. The implemented boundary
produces detached `[batch, 768]` representations; cache artifacts can be bound
when the actual Stage-2 runner is implemented.

No Stage-2 head training runner was implemented. No optimizer, training loop,
checkpoint selection execution, measurement-dev reporting, or TEST path was
added.

---

## 8. Exact next allowed step

Author review of this uncommitted diff. If accepted, the author may commit it.
After that, the next acceptance task is a fresh Colab runtime at the exact
future committed SHA with:

* real torch;
* the pinned PhoBERT revision;
* the real finalist A checkpoint;
* the real finalist B checkpoint;
* no downstream dataset rows;
* no head training;
* synthetic text only.

The authoritative smoke must:

1. pass all Stage-2 focused real-torch tests;
2. verify/load A through the existing authoritative Stage-1 finalist verifier;
3. verify/load B through the existing authoritative Stage-1 finalist verifier;
4. assert encoder frozen and `eval()`;
5. assert adapter frozen and `eval()`;
6. assert exactly 3,551,232 adapter parameters and expected state keys;
7. assert A/B checkpoint hashes remain unchanged before and after verification;
8. run a real forward on synthetic Vietnamese text;
9. assert representation shape `[batch, 768]`, FP32, finite and detached;
10. prove first-token position 0 is the downstream representation;
11. exercise `FULL`, `P25`, `P50`, `P75`, `P100` and `STRIP_ALL` on synthetic
    examples;
12. assert invariant base token grid across conditions;
13. assert deterministic keyed corruption;
14. confirm `VARIANT` refuses;
15. read no UIT-VSFC rows;
16. perform no optimizer step and no training.

Only after that smoke passes may the runtime pathway be marked accepted. Actual
Stage-2 head training remains a later task under the already frozen Audit-049
protocol.

---

## 9. Final state

```
STAGE2_PROTOCOL_FROZEN=YES
STAGE2_DUAL_FINALIST_INFRASTRUCTURE_STATIC=PASS
REAL_TORCH_ACCEPTANCE=PENDING
UNMARK_A_REAL_CHECKPOINT_LOAD=NOT_EXECUTED
UNMARK_B_REAL_CHECKPOINT_LOAD=NOT_EXECUTED
PINNED_PHOBERT_REAL_FORWARD=NOT_EXECUTED
DOWNSTREAM_CORRUPTION_PATH_STATIC=PASS
FINAL_ADAPTER_SELECTED=NO
DOWNSTREAM_MAY_SELECT_A_VS_B=NO
DOWNSTREAM_RESULTS_SEEN=NO
DOWNSTREAM_TEST=SEALED
STAGE2_TRAINING_STARTED=NO
READY_FOR_STAGE2_TRAINING=NO
```
