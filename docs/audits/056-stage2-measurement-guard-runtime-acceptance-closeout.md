# Audit 056 - Stage-2 Measurement Guard: Authoritative Runtime Acceptance Closeout

**Scope:** close the Stage-2 measurement-guard runtime acceptance opened by
Audit 054 §12 — Attempt 1's stale-fixture failure, the Audit-055 repair, and the
authoritative A100 retest that passed.
**Date:** 2026-09-08
**Type:** **documentation-only runtime closeout. NOT a scientific decision.** No
decision record is appended. No implementation, protocol, spec, scientific
constant, cache schema, head artifact or config changes.

---

## 1. Executive verdict

**MEASUREMENT_GUARD_RUNTIME_ACCEPTANCE = PASS.**

The D-S2-002 measurement guards executed against real torch on an A100 at the
exact committed repair SHA and passed completely: the direct smoke, the formerly
stale test, the focused suites, and the broader Stage-2 surface at
`232 passed, 0 failed, 0 skipped`.

**Zero Stage-2 runtime skips is the load-bearing figure.** Every one of the 62
torch-gated Stage-2 tests that the local no-torch environment can only collect
and skip actually ran on the A100. The guard surface is no longer *merely* a
contract; it is now backed by authoritative runtime evidence.

**No scientific Stage-2 measurement was performed.** The acceptance proves the
guards work; it did not read `measurement-dev`, produce a real degraded
measurement cache, or produce a `measurement-dev` scientific score. The suites
that ran are unit tests over synthetic data, and they do train synthetic heads and
compute synthetic scores — §8 states that scope in full.

| | |
|---|---|
| Attempt 1 | **FAIL** — stale test fixture, §3 |
| Audit-055 repair | test fixture only, no production change — §4 |
| Retest at `57b3d31e…` | **PASS** — §5 |
| Stage-2 runtime skips | **0** |
| Stage-2 runtime failures | **0** |
| End-to-end provenance, real torch | **PASS** |
| `measurement-dev` read | **NO** |
| Real measurement score produced | **NO** |
| A/B selection, winner, ranking | **NONE** |
| Official TEST | **SEALED**, not read |
| New scientific decision appended | **NO** |

---

## 2. Starting state and what the repair commit contains

```
branch : main
HEAD   : 57b3d31e1cf8145835117efee43a3f5172075ab8
status : clean (git status --short produced no output)
```

Both preconditions were verified before any edit.

**The repair commit introduces no scientific protocol or production change.**
Verified in the repository rather than assumed:

```
git diff --name-status 3f53c2d771db..57b3d31e1cf8

A  docs/audits/055-stage2-measurement-guard-runtime-test-fixture-repair.md
M  tests/test_stage2_head_campaign_torch.py
```

Two paths: one new audit, one test file. Restricting the same diff to `unmark/`,
`docs/spec/`, `configs/` and `scripts/` produces **no output at all** — not one
production, protocol, spec, decision or config file differs between the commit
that failed and the commit that passed.

That matters for what the acceptance means. The retest did not pass because the
implementation was fixed; the implementation was never broken. It passed because
a stale test stopped contradicting a correct guard.

---

## 3. Attempt 1 — the stale-fixture failure

Scientific/runtime commit tested:

```
3f53c2d771dbbf7c10f991e36a4601fb31ab5847
```

| | |
|---|---|
| Python | 3.13.15 |
| torch | 2.11.0+cu128 |
| CUDA | available |
| GPU | NVIDIA A100-SXM4-40GB |
| Pinned inventory sha256 | `78eeb840d50455b14bd564da5aed7318d96468b8deaad5986b77bf5c538315d2` |
| Pinned inventory bytes | `116290` |

After correcting a **notebook-only** `Preg1Role` import path — that name lives in
`unmark/evaluation/preg1_head.py`, not in `preg1_protocol`, and no repository file
needed editing for it:

| Gate | Result |
|---|---|
| Direct D-S2-002 smoke | **PASS** |
| Focused real-torch | `162 passed, 0 failed, 0 errors, 0 skipped` |
| Broader Stage-2 real-torch | `231 passed, **1 failed**, 0 errors, 0 skipped` |

The single failure:

```
tests/test_stage2_head_campaign_torch.py::test_measurement_scores_a_frozen_head_and_refuses_a_selection_split
```

Its *valid* synthetic measurement fixture still used `condition="P50"` with
`corruption_seed=99`. The production D-S2-002 guard refused it, correctly:
degraded Stage-2 measurement keys must bind exactly `19225`.

**Classification: `STALE_TORCH_TEST_FIXTURE`** — not a scientific implementation
failure. Audit 055 records this in full and is not restated here.

---

## 4. The Audit-055 repair, verified in source

Audit 055 repaired only the stale fixture. Confirmed at this HEAD by inspecting
the file rather than trusting the audit prose:

| Repair-gate claim | Verified |
|---|---|
| stale `measurement` / `P50` / `99` fixture absent | **YES** — `corruption_seed=99` occurs 0 times |
| valid degraded measurement fixture uses the constant | **YES** — `corruption_seed=STAGE2_MEASUREMENT_CORRUPTION_SEED`, 1 occurrence |
| `protocol-dev` / `P50` / `77` non-measurement fixture preserved | **YES** — `corruption_seed=77` still present |
| literal `19225` not hard-coded in the torch test | **YES** — 0 occurrences |
| no production repair was required | **YES** — §2's diff touches no production path |
| D-S2-002 constants intact | **YES** — `STAGE2_MEASUREMENT_CORRUPTION_SEED = 19225`, `..._PINNED = True` |

---

## 5. Authoritative retest — PASS

Fresh runtime at the exact committed repair SHA:

```
57b3d31e1cf8145835117efee43a3f5172075ab8
```

| | |
|---|---|
| Fresh detached checkout | **PASS** |
| Clean worktree | **PASS** |
| Python | 3.13.15 |
| torch | 2.11.0+cu128 |
| GPU | NVIDIA A100-SXM4-40GB |
| Pinned inventory sha256 | `78eeb840d50455b14bd564da5aed7318d96468b8deaad5986b77bf5c538315d2` |
| Pinned inventory bytes | `116290` |
| Inventory source | reused from the persistent MyDrive runtime-input cache, exact digest verified |

The inventory was **reused and verified, not re-fetched blindly** — the same
pinned identity Audit 052 §14 established as a preflight prerequisite of any
fresh clone.

### 5.1 Audit-055 source repair gate — PASS

The runtime re-checked the four §4 properties in its own checkout before running
anything, so the retest could not silently execute a tree that still held the
stale fixture.

### 5.2 Direct D-S2-002 smoke — PASS

| Verified | Result |
|---|---|
| `STAGE2_MEASUREMENT_CORRUPTION_SEED == 19225` | **PASS** |
| `STAGE2_MEASUREMENT_CORRUPTION_SEED_PINNED is True` | **PASS** |
| `stage2_measurement_extraction_plan` has no Python default for `corruption_seed` | **PASS** |
| exact 12-request plan | **PASS** |
| exactly `UNMARK-A` and `UNMARK-B` | **PASS** |
| six conditions per arm — `FULL, P25, P50, P75, P100, STRIP_ALL` | **PASS** |
| `FULL` binds `corruption_seed=None` | **PASS** |
| every degraded condition binds `19225` | **PASS** |
| `19224` refused | **PASS** |
| `19226` refused | **PASS** |
| non-integer and bool values refused | **PASS** |
| official TEST has no role member | **PASS** |

### 5.3 Test surface

| Suite | Result |
|---|---|
| Targeted formerly stale torch test | `1 passed, 0 failed, 0 errors, 0 skipped` |
| Focused real-torch suites | `162 passed, 0 failed, 0 errors, 0 skipped` |
| Broader Stage-2 real-torch suites | `232 passed, 0 failed, 0 errors, 0 skipped` |

**The 232 reconciles three ways**, checked against this repository:

| Where | Composition | Total |
|---|---|---|
| Local collection at `57b3d31e…` | 232 collected | **232** |
| Local execution (no torch) | `170 passed + 62 skipped` | **232** |
| Attempt 1 on A100 | `231 passed + 1 failed` | **232** |
| Retest on A100 | `232 passed + 0 skipped` | **232** |

The same test surface throughout. The 62 tests that the local no-torch
environment could only collect and skip are exactly the ones the authoritative
A100 retest executed, so:

* all former **62** torch-gated Stage-2 tests executed;
* Stage-2 runtime skips: **0**;
* Stage-2 runtime failures: **0**;
* **end-to-end provenance real-torch acceptance: PASS.**

---

## 6. Durable evidence

Persistent root:

```
/content/drive/MyDrive/UNMARK/UNMARK-BACKUP/stage2-measurement-guard-runtime-acceptance/57b3d31e1cf8/20260908T060247Z
```

Final evidence file:

```
/content/drive/MyDrive/UNMARK/UNMARK-BACKUP/stage2-measurement-guard-runtime-acceptance/57b3d31e1cf8/20260908T060247Z/055-measurement-guard-runtime-retest-final.json
```

| | |
|---|---|
| **Raw file SHA-256** | `c8c74eefbd5638785823d76ae20ed6a7327a24ccb13ee0e66b4fd2491b69a454` |

**This is the raw SHA-256 of the final runtime evidence JSON file** — the digest
of its bytes on disk. It is **not** a semantic digest and **not** a canonical
payload digest. Audit 053 §9 and §6 record two places where this project carries
digests of those other kinds; this is neither of them, and conflating them would
misdescribe what was hashed.

As Audit 052 §11 records, a Colab-resolved `.shortcut-targets-by-id/...`
rendering of this location is the resolved backing path of the persistent MyDrive
destination, not a runtime-only local path. Nothing here was read from Drive by
this documentation task; the runtime facts above were supplied by the author from
the authoritative runtime.

---

## 7. Execution-identity closeout

Two execution identities now exist and must not be conflated.

| Identity | SHA | What it is the provenance of |
|---|---|---|
| **Clean Stage-2 head-training execution** | `6693e728ccaebc987e4786bd5cd5e0f5c16143f7` | the four real clean `FULL` representation caches; the ten independently trained frozen Stage-2 heads; their selected within-head epochs and artifacts |
| **Measurement-guard accepted execution** | `57b3d31e1cf8145835117efee43a3f5172075ab8` | the measurement guards and the complete Stage-2 real-torch test surface accepted on A100 |

**`6693e728…` is immutable and stays bound to what it produced.** The four clean
caches and the ten head artifacts of Audits 052 and 053 were produced at that
commit and **must not be rewritten to a later repository SHA** — not to
`57b3d31e…`, not to any documentation commit. **The old clean caches were not
produced at `57b3d31e…`**; they remain bound to `6693e728…`, and their
`repository_head` field says so.

**`57b3d31e…` is the accepted measurement execution identity.** Future
documentation-only closeout commits — including the commit of this audit — do
**not** silently replace it. A later commit becomes the measurement execution
identity only if it is a scientific implementation or spec change *and* receives
its own authoritative runtime acceptance.

Until that happens, the future real Stage-2 measurement notebook must:

* **checkout detached at exactly** `57b3d31e1cf8145835117efee43a3f5172075ab8`;
* **assert `HEAD` equals that SHA before and after execution**;
* keep the tree clean;
* bind every new Stage-2 measurement representation cache's `repository_head` to
  that exact execution SHA.

This is the same execution-versus-documentation discipline the clean-head
execution used — Audit 053 §2 and §16.3, where the notebook checked out the
execution SHA detached and asserted it either side of training, rather than
supplying the right constant from whatever tree happened to be checked out.
`repository_head` is caller-supplied and never derived from git, so the assertion
is what makes the binding trustworthy.

---

## 8. Negative scientific boundary

| | |
|---|---|
| `MEASUREMENT_DEV_READ` | **NO** |
| `DEGRADED_MEASUREMENT_CACHE_CREATED` | **NO** |
| `DEGRADED_MEASUREMENT_RESULTS_SEEN` | **NO** |
| `REAL_STAGE2_CAMPAIGN_HEAD_RETRAINED` | **NO** |
| `A_B_SELECTION` | **NO** |
| `A_B_WINNER` | **NONE** |
| `A_B_RANKING` | **NONE** |
| `OFFICIAL_TEST_READ` | **NO** |
| `DOWNSTREAM_TEST` | **SEALED** |

**Scoped precisely, because the runtime suites used synthetic test data.** Those
suites *do* build synthetic heads, train them to their frozen budget and compute
scores over synthetic tensors — that is what
`test_measurement_scores_a_frozen_head_and_refuses_a_selection_split` exists to
do. This audit therefore does **not** claim no head or score was created. The
meaningful claims are narrower and all true:

* **no real `measurement-dev` row was read;**
* **no real measurement representation cache was produced;**
* **no scientific measurement score was produced;**
* **none of the ten real frozen campaign heads was retrained or modified.**

The broad `DOWNSTREAM_RESULTS_SEEN` flag is deliberately absent, for the reason
Audit 053 §11 gives: clean `protocol-dev` results have historically been seen —
the frozen selection rule required it — while `measurement-dev` and degraded
results have not. One flag covering both would be false in one direction or the
other.

---

## 9. Files changed

| File | Change |
|---|---|
| `docs/audits/056-stage2-measurement-guard-runtime-acceptance-closeout.md` | **new** — this file |

One path. Audits 054 and 055 are **not** edited: audit history is append-only, and
rewriting Audit 055 to say the retest passed would destroy the record of what
Attempt 1 found. No implementation, test, protocol, spec, decision, scientific
constant, cache schema, head artifact, Stage-1 artifact or config was modified.

No dataset was read. `measurement-dev` was not accessed. No measurement was run,
no representation cache created, no head scored.

No pytest run was required for this closeout, and none is claimed: no code
changed. The authoritative results are §5's, produced in the A100 runtime.

---

## 10. Inconsistencies found

**None.** Everything checkable from the repository agreed with the supplied
runtime facts:

* the repair commit's diff touches no production, spec, decision or config path
  (§2), so "no new scientific protocol or production implementation change" holds;
* all four Audit-055 repair-gate properties are true in the working tree (§4);
* `STAGE2_MEASUREMENT_CORRUPTION_SEED = 19225` and `..._PINNED = True` at this
  HEAD;
* the broader Stage-2 surface is **232** tests here, matching `231 + 1` in Attempt
  1 and `232 + 0` in the retest, and `170 + 62` locally — the four figures
  describe one surface (§5.3).

The one thing worth stating rather than leaving implicit: **`231 + 1 = 232`**, so
Attempt 1 and the retest ran the identical test surface. Had the retest reported a
smaller total, a test would have been lost rather than fixed, and the acceptance
would not have been earned.

---

## 11. Exact next allowed step

**Prepare — but do not execute — the real Stage-2 measurement notebook.**

Preparation must establish fail-closed bindings for:

| Binding | Value |
|---|---|
| exact execution SHA | `57b3d31e1cf8145835117efee43a3f5172075ab8`, detached, asserted before and after |
| measurement corruption seed | `19225` (D-S2-002) |
| split identity | `measurement-dev` official validation |
| arms | both — `UNMARK-A` and `UNMARK-B` |
| heads | all five already-frozen selected heads per arm |
| conditions | six per arm — `FULL, P25, P50, P75, P100, STRIP_ALL` |
| representation caches | **twelve** total |
| degraded realisation | the **same** for both arms |
| `FULL` | **no** corruption seed in cache identity |
| output namespace | immutable, Drive-only |
| A/B selection, ranking, winner | **none** |
| official TEST | unreachable |

**Notebook preparation itself must not read `measurement-dev`.** Preparation and
execution are separate steps, and only the first is authorised. **No real
measurement results may be included, because none exist.**

---

## 12. Final state

```
git status --short
?? docs/audits/056-stage2-measurement-guard-runtime-acceptance-closeout.md
```

`git diff --check` produced no output. `HEAD` is
`57b3d31e1cf8145835117efee43a3f5172075ab8`, unchanged. Uncommitted and awaiting
author review; nothing staged, committed, pushed or history-mutated.

---

```
AUDIT054_RUNTIME_ACCEPTANCE_ATTEMPT_1=FAIL_STALE_TEST_FIXTURE
AUDIT055_FIXTURE_REPAIR_RUNTIME_RETEST=PASS

MEASUREMENT_GUARD_RUNTIME_ACCEPTANCE=PASS
MEASUREMENT_GUARD_ACCEPTED_EXECUTION_HEAD=57b3d31e1cf8145835117efee43a3f5172075ab8

STAGE2_MEASUREMENT_CORRUPTION_SEED=19225
STAGE2_MEASUREMENT_CORRUPTION_SEED_PINNED=True

STALE_FIXTURE_TARGETED_RETEST=1_OF_1_PASS
FOCUSED_REAL_TORCH=162_OF_162_PASS
BROADER_STAGE2_REAL_TORCH=232_OF_232_PASS
FORMER_62_TORCH_GATED_STAGE2_TESTS_EXECUTED=YES
STAGE2_RUNTIME_TEST_SKIPS=0
STAGE2_RUNTIME_TEST_FAILURES=0

EXACT_12_REQUEST_PLAN=PASS
FULL_CORRUPTION_SEED=None
DEGRADED_CORRUPTION_SEED=19225
END_TO_END_PROVENANCE_REAL_TORCH=PASS

MEASUREMENT_DEV_READ=NO
DEGRADED_MEASUREMENT_CACHE_CREATED=NO
DEGRADED_MEASUREMENT_RESULTS_SEEN=NO

REAL_STAGE2_CAMPAIGN_HEAD_RETRAINED=NO
A_B_SELECTION=NO
A_B_WINNER=NONE
A_B_RANKING=NONE
OFFICIAL_TEST_READ=NO

READY_FOR_MEASUREMENT_GUARD_RUNTIME_RETEST=COMPLETED
READY_FOR_REAL_MEASUREMENT_NOTEBOOK_PREPARATION=YES
READY_FOR_REAL_MEASUREMENT_EXECUTION=NO
```

`READY_FOR_REAL_MEASUREMENT_EXECUTION=NO` is the flag that has not moved. The
guards are accepted and the notebook may now be **prepared**; execution needs that
prepared notebook reviewed first. An accepted guard and a reviewed execution plan
are different things, and only the first exists.
