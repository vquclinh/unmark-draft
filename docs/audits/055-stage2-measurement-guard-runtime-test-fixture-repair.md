# Audit 055 - Stage-2 Measurement Guard Runtime: Stale Test-Fixture Repair

**Scope:** record Attempt 1 of the Audit-054 measurement-guard runtime acceptance
and repair the one stale torch test fixture it exposed.
**Date:** 2026-09-08
**Type:** **engineering / runtime-test repair. NOT a scientific decision.** No
decision record is appended. No protocol, corruption mechanism, cache schema,
head artifact, measurement metric, A/B policy or scientific constant changes.

---

## 1. Executive verdict

**Attempt 1 FAILED on a stale test fixture, not on the production guard.**

The direct D-S2-002 smoke passed, the focused real-torch suites passed clean, and
the broader suite produced exactly one failure — a torch test that constructed a
degraded Stage-2 **measurement** representation under `corruption_seed=99`. The
new D-S2-002 guard refused it, correctly: every degraded measurement cache must
bind `19225`.

**The exception was evidence the guard works, not evidence of a defect.** The
fixture predates the freeze; the production code is right and is unchanged.

| | |
|---|---|
| Failure classification | **`STALE_TORCH_TEST_FIXTURE`** |
| D-S2-002 seed decision | **VALID** — unchanged, `19225` |
| D-S2-002 implementation guard | **VALID** — unchanged, behaved as designed |
| Production code changed | **NO** |
| Test code changed | **YES** — one fixture, one import |
| `measurement-dev` read | **NO** |
| Degraded artifact produced | **NO** |
| Runtime acceptance | **NOT COMPLETE** — §8 |

---

## 2. Starting state

```
branch : main
HEAD   : 3f53c2d771dbbf7c10f991e36a4601fb31ab5847
status : clean (git status --short produced no output)
```

Both preconditions were verified before any edit: HEAD matched the SHA the
runtime tested, and the working tree was clean.

---

## 3. Authoritative runtime, Attempt 1

Commit tested — the exact committed Audit-054 SHA:

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
| Pinned inventory size | `116290` bytes |

Persistent evidence:

```
MyDrive/UNMARK/UNMARK-BACKUP/stage2-measurement-guard-runtime-acceptance/3f53c2d771db/20260908T052331Z
```

### 3.1 A notebook-only import-path error, corrected in the same runtime

The first invocation failed while importing `Preg1Role` from
`unmark.evaluation.preg1_protocol`. That name does not live there: `Preg1Role` is
defined in `unmark/evaluation/preg1_head.py`, and `preg1_protocol` contains **zero**
occurrences of it — verified in this repository, not assumed.

**This was a notebook defect, not a repository defect.** It was corrected in the
same runtime, it changed no scientific state, and no repository file needed
editing for it. It is recorded only so the run's history is complete.

### 3.2 Direct D-S2-002 measurement-guard smoke — PASS

| Verified | Result |
|---|---|
| seed `== 19225` | **PASS** |
| pinned `== True` | **PASS** |
| no Python default for `corruption_seed` | **PASS** |
| exactly 12 requests | **PASS** |
| two arms x six conditions | **PASS** |
| `FULL` binds `corruption_seed=None` | **PASS** |
| every degraded request binds `19225` | **PASS** |
| `19224` refused | **PASS** |
| `19226` refused | **PASS** |
| non-integers and bools refused | **PASS** |
| official TEST role absent | **PASS** |

Every direct-smoke invariant above held in the authoritative A100 runtime. The
end-to-end batch and cache provenance guards are **not** covered by this smoke —
it exercises the constant, the plan and the key rules, not a collation or an
extraction — and were additionally exercised by the focused real-torch suite in
§3.3, which passed `162/162` with zero skips.

### 3.3 Focused real-torch suites — PASS

```
162 passed, 0 failed, 0 errors, 0 skipped
```

Zero skips is the load-bearing figure: the torch-gated half actually executed.

### 3.4 Broader real-torch suite — one failure

```
231 passed, 1 failed, 0 errors, 0 skipped
```

The single failure:

```
tests/test_stage2_head_campaign_torch.py::test_measurement_scores_a_frozen_head_and_refuses_a_selection_split
```

---

## 4. The stale fixture, and why the guard was right

The failing test constructed a representation it intended to be a **valid**
degraded measurement input:

```python
meas, meas_y = synthetic(N_DEV, seed=34, role=STAGE2_MEASUREMENT_ROLE,
                         condition="P50", corruption_seed=99)
```

`Stage2RepresentationKey.__post_init__` refused it. Under D-S2-002 (Audit 054
§7a.2) a key whose role is the Stage-2 measurement role and whose condition is
degraded must bind exactly `STAGE2_MEASUREMENT_CORRUPTION_SEED`. `99` is not that
value, so the key could not be constructed.

**The production guard is not wrong.** The fixture was written before the seed was
frozen, when any integer was a legitimate degraded measurement seed. The freeze
made one integer legitimate and the fixture kept the old one. That is the
definition of a stale test, and the refusal is the guard doing precisely the job
it was added for: refusing a degraded measurement cache that claims a realisation
the protocol did not freeze.

Consequently:

* the **scientific seed decision remains valid** — D-S2-002 is untouched;
* the **implementation seed guard remains valid** — not one line changed;
* the failure classification is **`STALE_TORCH_TEST_FIXTURE`**.

### Why local runs did not catch it

`test_measurement_scores_a_frozen_head_and_refuses_a_selection_split` carries the
`@requires_torch` marker. Torch is not installed in the local environment, so
before the A100 run this test was **collected and skipped**, never executed. Audit
054 §8 recorded exactly that and did not claim otherwise: the 62 Stage-2 skips
were reported as torch-gated and their execution was deferred to the runtime
acceptance in Audit 054 §12. This is that acceptance finding what only it could
find.

The per-test `skipif` convention is what made this visible rather than silent: a
module-level `importorskip` would have removed the test from collection entirely,
and a stale fixture inside an uncollected file is invisible until someone
eventually runs it.

---

## 5. The minimal repair

One test file. **No production file was modified.**

### Import

```python
from unmark.evaluation.stage2_head_campaign import (
    ...
    STAGE2_MEASUREMENT_CORRUPTION_SEED,
    STAGE2_MEASUREMENT_ROLE,
    ...
)
```

### Fixture

```python
    meas, meas_y = synthetic(N_DEV, seed=34, role=STAGE2_MEASUREMENT_ROLE,
                             condition="P50",
                             corruption_seed=STAGE2_MEASUREMENT_CORRUPTION_SEED)
    assert meas.key.corruption_seed == STAGE2_MEASUREMENT_CORRUPTION_SEED
```

The literal `19225` does **not** appear anywhere in the torch test file: the
exported constant is used, so if the frozen value ever changed under a future
recorded decision this fixture would follow it rather than silently disagree.

The added assertion states the invariant at the point of construction, so a
future reader sees what the fixture depends on without having to reason about the
key's internal rules.

---

## 6. What was deliberately NOT rewritten

**Arbitrary degraded seeds for non-measurement roles remain allowed and were
preserved.** D-S2-002 binds the value only for the Stage-2 measurement role;
Audit 054 §7a.2 scoped it that way on purpose, so that historical synthetic
fixtures for other roles keep their coverage.

Every degraded representation in the torch file, classified by parsing the file's
call sites rather than by eye:

| Line | Role | Condition | Seed | Action |
|---|---|---|---|---|
| 313 | `STAGE2_SELECTION_ROLE` (`protocol-dev`) | `P50` | `77` | **PRESERVED** — not the measurement role, so D-S2-002 does not apply |
| 425 | `STAGE2_MEASUREMENT_ROLE` | `P50` | was `99` | **REPAIRED** — now the frozen constant |

The clean measurement fixtures at lines 298, 304 and 445 use the default
`condition=FULL` with `corruption_seed=None` and are unaffected: `FULL` binds no
seed by construction.

No test in this file intentionally exercises wrong-seed refusal, so no invalid
value needed preserving for that purpose. That refusal is covered torch-free in
`tests/test_stage2_head_campaign.py`
(`test_a_measurement_degraded_key_refuses_another_realisation`, and per-condition),
which is also why the repair could be validated without torch — see §7.

**Exactly one fixture was stale. The repair was not broadened past it.**

---

## 7. Tests

`python -m py_compile tests/test_stage2_head_campaign_torch.py` — **OK**.

```
.venv/bin/python -m pytest -q \
  tests/test_stage2_head_campaign.py tests/test_stage2_head_campaign_torch.py

122 passed, 29 skipped, 0 failed
```

```
.venv/bin/python -m pytest -q \
  tests/test_stage2_head_campaign.py tests/test_stage2_head_campaign_torch.py \
  tests/test_stage2_campaign.py tests/test_stage2_campaign_torch.py \
  tests/test_stage2_dual_finalist_infra.py

170 passed, 62 skipped, 0 failed
```

Whole repository:

```
.venv/bin/python -m pytest -q

4343 passed, 170 skipped in 156.37s (0:02:36)
```

Zero failures, zero errors — and **identical to the totals Audit 054 §8 recorded**
at `3f53c2d771db`. That the counts did not move is the expected result, not a
missing signal: the only changed test is torch-gated and still skipped here, so a
correct repair must leave the local totals untouched. A shift would have meant
this change reached something it should not have.

**The repaired test did not execute locally, and this audit does not claim it
did.** `import torch` raises `ModuleNotFoundError` here. The test is **collected**
and reported as:

```
SKIPPED [1] tests/test_stage2_head_campaign_torch.py:416:
torch is not installed locally; tensor checks run where available
```

Torch was **not** installed to remove that skip, per the repository's established
workflow.

**What local execution can and does prove.** `Stage2RepresentationKey` is
torch-free, so the exact construction the repaired fixture performs is already
covered by passing torch-free tests: a measurement `P50` key binding `19225`
constructs, and one binding another integer is refused. Those six cases pass here.
That establishes the repair is correct at the point that failed; it does **not**
establish that the whole torch test now passes, which only the retest can.

---

## 8. Runtime acceptance is not complete

Attempt 1 did not pass. Runtime acceptance of the D-S2-002 guards remains
**NOT_YET_PASS** and stays that way until a fresh authoritative real-torch retest
passes at the **future committed repair SHA** — not at
`3f53c2d771dbbf7c10f991e36a4601fb31ab5847`, which is the commit that failed.

That retest must reproduce §3.2's direct smoke and run the broader suite to
`0 failed`.

---

## 9. Negative evidence

Attempt 1 failed while constructing a synthetic fixture, before any scientific
work could occur:

| | |
|---|---|
| Degraded representation cache created | **NO** — none, at any condition |
| `measurement-dev` row read | **NO** |
| **Real** `measurement-dev` scientific score produced | **NO** |
| Official TEST read | **NO** — remains SEALED |
| A/B selection performed | **NO** |
| **Real** Stage-2 campaign head retrained | **NO** — the ten Audit-053 selected-head artifacts are untouched and unmodified |
| Clean caches altered | **NO** — the four Audit-052 caches are untouched |

**The two rows marked "real" are scoped deliberately.** The suites that ran are
unit tests, and they do construct synthetic heads, train them for their frozen
budget and compute scores over synthetic tensors — that is what
`test_measurement_scores_a_frozen_head_and_refuses_a_selection_split` exists to
do. None of that touches science: **not one of the ten Audit-053 real
selected-head artifacts was retrained or modified**, and **no `measurement-dev`
scientific score was produced**, because no `measurement-dev` row was read at all.
Saying "no head was trained" without that scope would be false; saying "no real
head was retrained" is both true and the claim that matters.

**Runtime evidence did persist, and should be kept.** The attempt wrote logs,
result XML and the notebook-correction record under:

```
MyDrive/UNMARK/UNMARK-BACKUP/stage2-measurement-guard-runtime-acceptance/3f53c2d771db/20260908T052331Z
```

Those files are **runtime-acceptance and negative evidence, not scientific
measurement artifacts**. They are the record that Attempt 1 ran, what it checked
and how it failed, and they should be retained for exactly that reason. What does
not exist is any *scientific* output: no degraded representation cache, no
measurement score, no artifact that any later stage could consume. There is no
scientific artifact to discard, and the runtime evidence is not a candidate for
discarding.

---

## 10. Files changed

| File | Change |
|---|---|
| `tests/test_stage2_head_campaign_torch.py` | one import added; one stale fixture repaired; one assertion added |
| `docs/audits/055-stage2-measurement-guard-runtime-test-fixture-repair.md` | **new** — this file |

Two paths. **Audit 054 was not edited** — audit history is append-only, and
rewriting it would destroy the record of what Attempt 1 actually found. No
implementation, protocol, spec, decision or config file was modified.

---

## 11. Inconsistencies found

None beyond the stale fixture itself. Specifically checked and found consistent:

* `STAGE2_MEASUREMENT_CORRUPTION_SEED` is exported from
  `unmark/evaluation/stage2_head_campaign.py` and importable by the torch test;
* no other degraded measurement fixture exists in the torch file (§6);
* the literal `19225` appears nowhere in the torch test file;
* `Preg1Role` is in `preg1_head`, not `preg1_protocol` — confirming §3.1 was a
  notebook error, with no repository-side ambiguity to fix.

---

## 12. Exact next allowed step

1. **Author review and commit** of this two-path repair.
2. **A fresh authoritative real-torch retest at the resulting commit** — the
   direct D-S2-002 smoke plus the broader Stage-2 suite to `0 failed`.
3. Only after that passes may real measurement-notebook **preparation** begin —
   preparation, separately reviewed, before anything is run.

Still not permitted: reading any `measurement-dev` row, creating any degraded
cache, producing any measurement score, opening official TEST, or any A/B
selection.

---

## 13. Final state

```
git status --short
 M tests/test_stage2_head_campaign_torch.py
?? docs/audits/055-stage2-measurement-guard-runtime-test-fixture-repair.md
```

`git diff --check` produced no output. `HEAD` is
`3f53c2d771dbbf7c10f991e36a4601fb31ab5847`, unchanged. Uncommitted and awaiting
author review; nothing staged, committed, pushed or history-mutated.

---

```
AUDIT054_RUNTIME_ACCEPTANCE_ATTEMPT_1=FAIL_STALE_TEST_FIXTURE
AUDIT054_DIRECT_GUARD_SMOKE=PASS
AUDIT054_FOCUSED_REAL_TORCH=162_OF_162_PASS
AUDIT054_BROADER_REAL_TORCH=231_PASS_1_FAIL
D_S2_002_GUARD_BEHAVIOR=CORRECT_REFUSAL

STALE_TORCH_TEST_FIXTURE_REPAIR=IMPLEMENTED
SCIENTIFIC_PROTOCOL_CHANGE=NO
SCIENTIFIC_IMPLEMENTATION_CHANGE=NO

STAGE2_MEASUREMENT_CORRUPTION_SEED=19225
STAGE2_MEASUREMENT_CORRUPTION_SEED_PINNED=True

MEASUREMENT_DEV_READ=NO
DEGRADED_MEASUREMENT_CACHE_CREATED=NO
DEGRADED_MEASUREMENT_RESULTS_SEEN=NO
A_B_SELECTION=NO
OFFICIAL_TEST_READ=NO

READY_FOR_MEASUREMENT_GUARD_RUNTIME_RETEST=YES
MEASUREMENT_GUARD_RUNTIME_ACCEPTANCE=NOT_YET_PASS
READY_FOR_REAL_MEASUREMENT_NOTEBOOK_PREPARATION=NO
READY_FOR_REAL_MEASUREMENT_EXECUTION=NO
```

`D_S2_002_GUARD_BEHAVIOR=CORRECT_REFUSAL` is the finding this audit exists to
record. The one failure was the new guard rejecting a fixture that predated it —
the guard working, caught by the first execution able to reach it.
