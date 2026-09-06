# Audit 050 - Stage-2 Dual-Finalist Infrastructure Implementation

**Scope:** static implementation acceptance for the Stage-2 frozen UNMARK
dual-finalist pathway and downstream corruption/input path required by Audit
049. Runtime acceptance with torch, the real finalist checkpoints and the real
pinned PhoBERT model remains pending after the device-coherence repair recorded
below.
**Date:** 2026-09-06
**Type:** implementation audit. No scientific protocol change. No runtime
artifact acceptance.

---

## 1. Executive verdict

**PASS WITH RUNTIME EVIDENCE PENDING - structural / torch-free infrastructure
review passed; authoritative real-torch artifact acceptance is pending retest.**

This audit implements the two gaps Audit 049 deliberately left open:

1. a frozen UNMARK downstream representation pathway for `UNMARK-A` and
   `UNMARK-B`;
2. a downstream corruption/input path for `FULL`, `P25`, `P50`, `P75`, `P100`
   and `STRIP_ALL`.

No classification head was trained. No UIT-VSFC row was read. No
measurement-dev result was inspected. Official TEST remains sealed. No A/B
selection, winner rule, margin, p-value or significance test was added.

This audit does not claim authoritative runtime acceptance. The first
authoritative A100 runtime attempt verified both real finalist checkpoints and
loaded the real pinned PhoBERT encoder, but the first real `UNMARK-A` forward
failed because the adapter stayed on CPU while the encoder was on `cuda:0`.
That implementation blocker is repaired here. Runtime acceptance remains
pending until a fresh exact-commit A100 retest proves the repaired forward path.

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
* encoder parameters must occupy one coherent device;
* adapter parameters must occupy one coherent device;
* encoder and adapter devices must match;
* a newly loaded adapter is moved to the supplied encoder's actual parameter
  device, preserving FP32;
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
Stage-1 masked-mean pooling is not used. The representation path accepts the
normal CPU tensor batch produced by `collate_stage2_unmark_batch(...)` and moves
only the required forward tensors onto the coherent pathway device before
embedding, adapter and encoder computation.

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
16 passed, 21 skipped
```

The skips are torch/CUDA-gated in a local environment without torch. They
include the runtime-critical tensor/checkpoint/freezing/representation checks
and the new device-coherence regressions. Therefore UNMARK-A/B real checkpoint
loading, pinned PhoBERT forward execution, freezing, FP32/finite/detached
representation properties, first-token runtime extraction and CUDA transfer are
not locally executed by this audit repair.

Importer contract:

```
pytest -q tests/test_preg1_import_contract.py
```

Result:

```
16 passed
```

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
1369 passed, 96 skipped
```

Full repository, excluding the sandbox-blocked multiprocessing file:

```
pytest -q --ignore=tests/test_stage1_parallel.py
```

Result:

```
4169 passed, 129 skipped
```

An initial full `pytest -q` run inside the restricted sandbox failed seven
`tests/test_stage1_parallel.py` cases with `PermissionError: [Errno 1]
Operation not permitted` while Python's forkserver attempted to bind a local
AF_UNIX socket. The broader repair regression therefore excludes that known
sandbox-blocked file. No test failure was observed in the repair runs above.

---

## 6. Diff hygiene

```
git diff --check
```

produced no output.

---

## 7. Authoritative runtime attempt 1 - device-coherence blocker

Commit tested:

```
11cc8813280f22ad2b20421b4029faf2c9a526ac
```

Evidence completed before the first real forward:

* Stage-2 focused real-torch tests: `31 passed, 0 skipped`;
* PREG1 committed importer contract: `16 passed, 0 skipped`;
* authoritative Stage-1 finalist torch tests: `40 passed, 0 skipped`;
* real finalist A authoritative verification: PASS, sha256
  `6773fbb59c7381ba8ddaa944302124a124f5b8a5cb0a5dbb1a5063f3db4a2a91`,
  8 adapter tensors, 3,551,232 adapter parameters, FP32, finite;
* real finalist B authoritative verification: PASS, sha256
  `9405bd76c04939641170cb71507ce8eb669eb2987016b86b495a403ceafcb9d2`,
  8 adapter tensors, 3,551,232 adapter parameters, FP32, finite;
* both checkpoint hashes unchanged before and after verification;
* pinned PhoBERT `vinai/phobert-base` revision
  `01daacda68afe13d83023d16ec647239e344a1e6` loaded frozen, eval, FP32,
  hidden size 768, device `cuda:0`, trainable parameters 0;
* synthetic-only `SCIENTIFIC` corruption smoke passed for `FULL`, `P25`,
  `P50`, `P75`, `P100` and `STRIP_ALL`;
* base-grid invariance passed;
* keyed corruption determinism passed;
* row-order independence passed;
* `VARIANT` fail-closed passed.

Runtime failure:

* first real `UNMARK-A` forward failed;
* immediately before forward, `pathway.adapter` device was CPU;
* immediately before forward, `pathway.encoder` device was `cuda:0`;
* the collated Stage-2 input tensors were on CPU;
* first observed exception:

```
RuntimeError: Expected all tensors to be on the same device, but got index is
on cpu, different from other tensors on cuda:0
```

The exception occurred inside torch embedding / `index_select`. This is a real
implementation blocker, not a scientific protocol change.

Negative evidence preserved:

* no UIT-VSFC row was read;
* no downstream result was read;
* no head was built;
* no optimizer existed;
* no training happened;
* checkpoint hashes were unchanged.

---

## 8. Device-coherence repair

The repair is implementation-only and does not change any Audit-049 protocol
constant.

`load_frozen_unmark_pathway(...)` now determines the supplied encoder's actual
single parameter device, refuses incoherent encoder devices, loads the exact
Stage-1 adapter as before, then moves the adapter to the encoder device before
freezing and validation. CUDA is not hard-coded; CPU remains valid.

`require_frozen_unmark_pathway(...)` now fails closed when encoder parameters
are not on one coherent device, adapter parameters are not on one coherent
device, encoder and adapter devices differ, either module is trainable, either
module is in train mode, or FP32 constraints are violated.

`extract_stage2_unmark_representations(...)` now accepts the normal CPU batch
from `collate_stage2_unmark_batch(...)`, identifies the coherent pathway device
and moves the required forward tensors there before word-embedding, adapter and
encoder computation. Tensor dtype and shape are checked after transfer; sample
ids and condition metadata are not moved.

---

## 9. Limitations

Runtime acceptance is PENDING_RETEST, not PASS. Attempt 1 verified both real
finalist checkpoints and loaded the real pinned PhoBERT encoder, but no real
UNMARK forward was accepted because the device-coherence blocker stopped the
first `UNMARK-A` forward. No `UNMARK-B` real forward was attempted or accepted.

Torch is unavailable in the local sandbox, so tensor-level Stage-2 tests and
CUDA-specific device regressions skipped locally. They are written against
synthetic checkpoints/modules and should run where torch and CUDA are available.

The repaired real pinned PhoBERT forward has not been executed yet. No
authoritative runtime `[batch, 768]` representation evidence exists for the
repaired code. No network access was needed for this implementation repair.

No representation cache schema was added for Stage-2. The implemented boundary
produces detached `[batch, 768]` representations; cache artifacts can be bound
when the actual Stage-2 runner is implemented.

No Stage-2 head training runner was implemented. No optimizer, training loop,
checkpoint selection execution, measurement-dev reporting, or TEST path was
added.

---

## 10. Exact next allowed step

Author review of this uncommitted diff. If accepted, the author may commit it.
After that, the next acceptance task is a fresh A100 Colab runtime at the exact
future committed SHA, not a notebook-side workaround, with:

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

Only after that fresh exact-commit smoke passes may the runtime pathway be
marked accepted. Actual Stage-2 head training remains a later task under the
already frozen Audit-049 protocol.

---

## 11. Final state

```
STAGE2_PROTOCOL_FROZEN=YES
STAGE2_DUAL_FINALIST_INFRASTRUCTURE_STATIC=PASS
AUTHORITATIVE_RUNTIME_ATTEMPT_1=FAIL_DEVICE_COHERENCE
DEVICE_COHERENCE_REPAIR=IMPLEMENTED
REAL_TORCH_ACCEPTANCE=PENDING_RETEST
UNMARK_A_REAL_FORWARD=NOT_ACCEPTED
UNMARK_B_REAL_FORWARD=NOT_ACCEPTED
DOWNSTREAM_CORRUPTION_PATH_STATIC=PASS
FINAL_ADAPTER_SELECTED=NO
DOWNSTREAM_MAY_SELECT_A_VS_B=NO
DOWNSTREAM_RESULTS_SEEN=NO
DOWNSTREAM_TEST=SEALED
STAGE2_TRAINING_STARTED=NO
READY_FOR_STAGE2_TRAINING=NO
```
