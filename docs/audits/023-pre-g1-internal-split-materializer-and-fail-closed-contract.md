# Audit 023 — pre-G1 internal split materialiser and fail-closed splitter contract

| | |
|---|---|
| **Audit id** | 023 |
| **Created (UTC)** | 2026-08-21 |
| **Baseline HEAD** | `819d09f2df95ca57444a63c86363e614b44ce458` |
| **Scope** | Harden the committed splitter against three probe-found failures; build the deterministic split materialiser; do **not** materialise the real split |
| **Predecessors** | [021](021-pre-g1-dataset-profile-and-protocol-precommit.md), [022](022-uit-vsfc-real-data-profile-integrity-closure.md) |
| **Phase** | pre-G1, after Audit 022 FINAL PASS |
| **Type** | **Implementation hardening + materialiser.** No real data, no training, no optimizer, no downstream score |

---

## A. VERDICT

**IMPLEMENTATION PASS — REAL SPLIT NOT YET MATERIALISED**

The splitter is hardened, the materialiser exists, and both are covered by
executable tests. **No real membership has been produced or observed.** This
audit will be revised **in place** after the real Colab materialisation; there
will be no Audit 024 for that closure.

**2199 local tests pass, 56 skip** — 55 of them new in `tests/test_preg1_split.py`
(2144/56 at Audit 022).

| Item | State |
|---|---|
| Splitter | **hardened, single implementation** — no competing splitter was created |
| Fraction-mapping insertion order | **invariant** |
| Record input order | **invariant** |
| Conflicting-label canonical group | **fail-closed** |
| Duplicate sample id | **fail-closed** |
| Materialiser | **implemented; real run pending** |
| Real membership | **not observed** |

**No head trainer. No downstream score. No Stage-1 training or HPO.**
**D-B3B0-002 remains OPEN.** The compiled PDF **remains stale**.

---

## B. PHASE BOUNDARY FROM AUDIT 022

Audit 022 closed at **FINAL PASS**: the derived pre-G1 pool is verified
(11 424 rows, digest `a20c0f77…`, zero duplicate / conflicting / cross-split
groups), the channel densities are measured, and the tokenizer geometry is
characterised. Audit 022 §L explicitly deferred the 80/20 split to a later task
**after researcher review**.

This is that task, minus the run. It builds the mechanism and refuses to use it
on real data, because the probes found the mechanism was not yet trustworthy.

---

## C. THE THREE FINDINGS, AND WHY EACH MATTERS

The researcher ran synthetic probes against the committed
`profiling.stratified_group_split`. Five properties already held — determinism
across calls, input-order invariance, atomic canonical duplicates, empty-input
rejection, and no global RNG. Three failed. **All three were confirmed here by
reading the code, not taken on faith.**

### S23-F1 — fraction-mapping insertion order changed membership

`names = list(fractions)` returns **insertion order**, and parts were allocated
sequentially from that list. So `{"protocol-train": 0.8, "protocol-dev": 0.2}`
gave protocol-train the prefix of each label's ordered group list, while the
logically identical `{"protocol-dev": 0.2, "protocol-train": 0.8}` gave that
prefix to protocol-dev.

**Why it matters.** The order in which a dict literal was typed was a scientific
variable. Both mappings mean the same thing, produce the same *counts*, and are
recorded identically in any artifact — yet assign different examples. A
reproduction attempt that retyped the mapping in a different order would get a
different split and no diagnostic.

### S23-F2 — conflicting-label canonical groups failed open

Each group's label came from `Counter(...).most_common(1)[0][0]` — a majority
vote, with ties broken by `Counter`'s internal ordering. A group containing one
`negative` and one `positive` **did not raise**. It was assigned.

**Why it matters.** This directly contradicts `DUPLICATE_CONTRACT`, which
requires such groups to **STOP for researcher review** and never be silently
relabelled, dropped or assigned. The vote manufactures a gold label no annotator
assigned, and does it *precisely* in the case a human was supposed to inspect.
Audit 022 handled the one real conflicting group by explicit whole-group
exclusion after researcher review — the splitter would have quietly voted on it.

### S23-F3 — duplicate sample ids failed open

Two records sharing a `sample_id` did not raise; both occurrences were emitted.

**Why it matters.** A membership artifact **is** a list of ids. If two rows share
one, the artifact cannot say which was assigned, and any downstream join either
doubles a row or drops one — silently, and in a way that survives every
count-based check.

---

## D. THE REPAIR

**One implementation, strengthened. No second splitter was created** — a
competing "safe" splitter would leave the unsafe one importable and callable.

| Contract | Implementation |
|---|---|
| **Empty input** | unchanged: raises `EvaluationContractViolation` |
| **Fractions** | `_validate_fractions` — non-empty mapping, non-empty string names, numeric, finite, strictly positive, summing to 1.0 within the existing `1e-9` tolerance |
| **Allocation order** | `_canonical_split_order` — sort by `(-fraction, name)`; a pure function of the mapping's **content** |
| **Unique ids** | `_require_unique_sample_ids` — raises, naming ids only |
| **Grouping** | unchanged: `text_digest(canon(text))`, groups atomic |
| **Labels** | `_group_label` — exactly one distinct label per group or raise; **no vote, no tie-break** |
| **Output** | ids only, sorted, disjoint, complete |
| **RNG** | unchanged keyed-digest ordering; no `random`, numpy, torch or sklearn |

### The allocation order preserves every locked value

Sorting by descending fraction gives `protocol-train` (0.80) then `protocol-dev`
(0.20) — **the order the locked mapping already had**. Verified empirically, not
argued: the pre-hardening logic was re-implemented and compared against the
hardened function on a synthetic pool carrying the real class totals. **Identical
membership**, both with and without same-label canonical duplicates. Equal
fractions still get a total order via the name tie-break.

### "Majority label" is gone from the contract

The docstring's *"using each group's majority label"* is removed. After the
fail-closed check every group has exactly one distinct label, so
`SPLIT_STRATIFICATION_RULE` now states that groups are stratified by **that**
label and that conflicts are an error rather than a tie-break. The scientific
target was not silently redefined — the vote was never authorised by the
proposal in the first place.

---

## E. THE MATERIALISER

`unmark/evaluation/preg1_split.py` and `scripts/materialize_preg1_split.py`.
The workflow is committed code, not a notebook.

### Nothing is restated

Fractions, seed, seed tag, expected rows, expected class counts and the input
digest are all **imported** from `preg1_protocol`. A test asserts by AST that
`preg1_split.py` contains no integer literal equal to any locked value, and that
the script contains neither `17486` nor the fractions. The CLI exposes **no**
`--seed` and **no** `--fractions` flag: a command-line override of a
precommitted constant is exactly the hole the protocol exists to close.

### Preflight, before any assignment

1. input file exists;
2. **SHA-256 equals `a20c0f77…` exactly** — the gate that ties this split to the
   corpus approved in Audit 022, exclusion already applied;
3. exactly 11 424 rows;
4. label counts exactly 5 324 / 458 / 5 642 (either encoding accepted, nothing
   coerced);
5. sample ids globally unique;
6. text non-empty;
7. canonical duplicate / conflict analysis re-run **here**, not assumed from a
   document;
8. conflicting-label groups must be zero;
9. exclusion policy agrees with Audit 022;
10. **official validation and test are not parameters** of this program.

A pool with the right row count is not the same pool. The digest is what makes
the difference checkable.

### Postflight, before anything is written

Disjoint parts; union equals all input ids; every id emitted exactly once; no
duplicate emitted id; no canonical group straddling parts; class counts matching
the allocation rule; **and, when the input digest is the locked one**, exact
equality with the precommitted aggregates below.

**The locked-digest gate is proven live, not dead code**: a test constructs a
pool that claims the locked digest but carries different contents and asserts
the gate fires. Without that test, the module's strongest guarantee could have
been unreachable while every other test passed.

### Expected split — precommitted, computed from committed aggregates

| Part | negative | neutral | positive | total |
|---|---|---|---|---|
| `protocol-train` | 4 259 | 366 | 4 514 | **9 139** |
| `protocol-dev` | 1 065 | 92 | 1 128 | **2 285** |

These are **derived, not observed**. The derived pool has zero canonical
duplicate groups (Audit 022), so every group is a singleton and the per-class
allocation is fully determined by the class totals and the allocation rule. That
is what makes them a **precommitment**: they were computable before any
membership existed, and a synthetic pool with the real class totals reproduces
them exactly under the locked seed.

**No real membership ids are hard-coded anywhere.** None have been observed.

---

## F. DETERMINISTIC ARTIFACTS

Schema **`preg1-split-v1`**, defined once in `SPLIT_SCHEMA_VERSION` and imported
everywhere — the Audit-022 schema-drift lesson applied before the fact.

| File | Content |
|---|---|
| `protocol-train.ids.txt` | sorted ids, one per line |
| `protocol-dev.ids.txt` | sorted ids, one per line |
| `split-manifest.json` | the deterministic scientific record |
| `report.md` | human-readable summary |
| `runtime-environment.json` | *optional, and deliberately **not** part of the scientific artifact* |

The manifest records schema, dataset/version/task, repository HEAD as supplied
provenance, input digest, rows, label counts, exclusion policy, fractions, seed
tag, seed, the grouping / allocation-order / stratification / determinism rules,
per-part totals and class counts, per-file digests, a combined assignment
digest, duplicate-id count, conflicting-group count, cross-part leakage, and the
four boundary flags (raw text persisted, official validation used, official test
used, downstream score used — all `false`).

**Byte-determinism.** No timestamp, run uuid, hostname, absolute path or elapsed
time enters the manifest. Two runs on the same input produce **byte-identical**
id files, manifest and report — asserted by test, not asserted in prose. Runtime
facts go to a separate file.

**Overwrite is refused**, and writing is **staged**: everything goes to a
`.partial` sibling directory and is moved into place only after every invariant
passes. A test kills a run mid-contract and asserts that neither the target nor
the staging directory survives — a failed run must not leave something that
reads as an authoritative membership.

---

## G. RAW-TEXT, LICENSING AND SPLIT BOUNDARIES

**No corpus text is written anywhere.** Not to the id files, not to the manifest,
not to the report — and **not into exception messages**, which are files like any
other. The conflict error reports canonical digest, labels and sample ids; the
duplicate-id error reports ids. Both are covered by tests that plant a
distinctive string and assert it never appears.

This is not fastidiousness: UIT-VSFC is `OFFICIAL_PUBLIC_DISTRIBUTION` with
license `NOT_ESTABLISHED` (D-PREG1-002b), and the B3A inventory is
`NO_EXPLICIT_LICENSE`. Nothing redistributable may leak into a committed
artifact by way of a stack trace.

**Official validation** remains untouched measurement-dev. **Official test**
remains sealed for downstream scores and protocol selection. Neither is a
parameter of the materialiser — asserted structurally, by parsing the CLI's
argument list rather than reading its documentation.

---

## H. TESTS

`tests/test_preg1_split.py`, **55 tests**, ML-free and network-free. Every
fixture is synthetic; the real derived csv is never read.

Covering: repeated-call determinism; input-order invariance; **fraction-dict
insertion-order invariance**; equal-fraction tie-break; atomic canonical
duplicates; **conflicting-label group raises**; conflict error leaks no text;
**majority does not resolve a conflict** (3-to-1 fails exactly as 2-to-2);
**duplicate id raises**; duplicate-id error leaks no text; empty input; nine
invalid-fraction cases; disjointness; completeness; no duplicate output; sorted
output; seed sensitivity; no global-RNG dependency; no RNG/ML imports;
**exact 9139 / 2285 and 4259/366/4514 · 1065/92/1128 from a synthetic pool**;
wrong digest; missing file; wrong row count; wrong label counts; duplicate ids;
empty text; missing column; conflicting groups; artifact presence; overwrite
refusal; **failed run leaves nothing**; **byte-identical reruns**; no
runtime-varying manifest fields; runtime evidence separated; locked values in
the manifest; id-file digests match; **no raw text in artifacts**; sorted and
complete id files; assignment-digest stability and sensitivity; CLI boundary
flags; no restated literals; **locked-digest gate is live**; CLI refuses a wrong
digest with a non-zero exit.

Three of these tests were **rewritten mid-task** because my first versions were
prose-matching — they banned substrings that legitimately appear in the module's
own documentation. See §J.

---

## I. WHAT HAS NOT HAPPENED

- **The real 80/20 split has not been materialised.** No membership exists.
- **No real UIT-VSFC data was read**, locally or otherwise. Nothing downloaded.
- **No head trainer.** No optimizer, no checkpoint, no LR selection.
- **No downstream score.** No Vanilla-vs-Base-only result exists.
- **No Stage-1 training or HPO.** Stage-1 is untouched.
- **No locked value changed** — fractions 80/20, seed 17486, tag
  `UNMARK-PREG1-SPLIT-UITVSFC-v1`, `max_length` 256, dataset, task, split roles
  and seal all verified unchanged from the committed modules.
- **No prohibited git operation.** Nothing staged, committed, tagged or pushed.

---

## J. LIMITATIONS

1. **The precommitted counts are verified only on synthetic pools.** A synthetic
   pool with the real *class totals* reproduces 9139 / 2285 exactly, and the
   real pool has zero canonical duplicates so the arithmetic should carry over —
   but "should" is doing work there. The real run is what settles it.
2. **The digest gate is the only thing tying this to the approved corpus**, and
   this environment cannot check that the digest belongs to the corpus it claims
   to. If the wrong file were produced upstream with a matching digest, nothing
   here would notice — that is what a digest is.
3. **The equal-fraction tie-break is arbitrary.** Ascending name is a total order,
   not a principled one. It does not affect the locked 80/20 mapping, where the
   fractions differ, but a future equal-fraction protocol should decide the rule
   deliberately rather than inherit lexical ordering.
4. **`_label_name` accepts two encodings.** Convenient, and a place where a
   mislabelled column could pass. The class-count check is the backstop.
5. **I wrote three prose-matching tests, again.** They banned substrings —
   `"timestamp"`, `"uuid"`, `"9139"` — that appear inside the module's own
   documentation of what it excludes. All three failed immediately and were
   rewritten structurally (walk the manifest's keys; AST-scan for integer
   literals). This is at least the eighth instance of this failure mode in this
   project, and the first where I caught it within the same task.
6. **A real defect reached the postflight and was caught by a test, not by
   review.** `Counter` omits absent keys, so a part with zero members of a class
   compared unequal to an expectation that listed it as `0`. On the locked pool
   no class is empty in either part, so it would not have fired — a latent
   failure waiting for a different corpus.

---

## K. TASK-END SELF-AUDIT

| # | Check | Result |
|---|---|---|
| 1 | Audit 022 reread; remains **FINAL PASS**, unmodified | **yes** |
| 2 | Audit 023 created; **no Audit 024** | **yes** |
| 3 | Splitter remains **one** implementation, not duplicated | **yes** — hardened in place |
| 4 | Fraction-mapping order no longer affects membership | **yes** — tested both directions |
| 5 | Record input order no longer affects membership | **yes** |
| 6 | Conflicting-label groups fail closed | **yes** — 2-to-2 and 3-to-1 both raise |
| 7 | No majority-label behaviour remains | **yes** — `Counter.most_common` removed from labelling |
| 8 | Duplicate sample ids raise | **yes** |
| 9 | Raw text never in conflict errors or artifacts | **yes** — planted-string tests |
| 10 | Canonical duplicates stay atomic | **yes** |
| 11 | Split seed remains **17486** | **yes** |
| 12 | Seed tag unchanged | **yes** — `UNMARK-PREG1-SPLIT-UITVSFC-v1` |
| 13 | Fractions remain **80/20** | **yes** |
| 14 | Expected totals **9139 / 2285** | **yes** — computed and reproduced |
| 15 | Expected class counts **4259/366/4514** and **1065/92/1128** | **yes** |
| 16 | Materialiser imports locked values rather than restating them | **yes** — AST-asserted |
| 17 | Real derived SHA enforced; wrong SHA fails | **yes** — unit and CLI (exit 2) |
| 18 | Artifacts byte-identical across repeated synthetic runs | **yes** |
| 19 | Overwrite refused; failed run leaves nothing | **yes** — staged writes |
| 20 | **No real membership materialised** | **yes** |
| 21 | No UIT-VSFC or model downloaded locally | **yes** — `.venv` remains ML-free |
| 22 | Official validation role unchanged | **yes** |
| 23 | Official test seal unchanged | **yes** |
| 24 | No head trainer, no downstream score, no Stage-1 training/HPO | **yes** |
| 25 | Hardening changed **no** membership under the locked mapping | **yes** — compared against the pre-hardening logic |
| 26 | `SPLITTER_STATUS` no longer hard-codes run state; stale-phrase test updated | **yes** |
| 27 | Decision log updated (**D-PREG1-014**); D-PREG1-011/012/013 unchanged | **yes** |
| 28 | D-B3B0-002 | **OPEN** |
| 29 | Compiled PDF | **STALE** |
| 30 | Tests pass | **2199 passed, 56 skipped**; targeted: 145 profiling + 55 split |
| 31 | `git diff --check` clean | **yes** |
| 32 | Everything unstaged | **yes** |
| 33 | No prohibited git operation | **yes** |

---

## L. REQUIRED NEXT ACTION

**Only after researcher review and commit**, run
`scripts/materialize_preg1_split.py` on Colab against the derived train csv,
into a **new** directory, and revise **this audit in place** with the observed
manifest, the id-file digests and the assignment digest.

The real run must reproduce: `protocol-train` **9139** (4259/366/4514),
`protocol-dev` **2285** (1065/92/1128), zero cross-part canonical leakage, and
`preg1-split-v1` in the manifest. Any deviation is a finding, not a surprise to
be accepted.

---

```
AUDIT 023 CREATED:
YES

VERDICT:
IMPLEMENTATION PASS — REAL SPLIT NOT YET MATERIALIZED

BASELINE HEAD:
819d09f2df95ca57444a63c86363e614b44ce458

SPLITTER:
HARDENED / SINGLE IMPLEMENTATION

FRACTION-MAPPING ORDER:
INVARIANT

RECORD INPUT ORDER:
INVARIANT

CONFLICTING-LABEL GROUP:
FAIL-CLOSED

DUPLICATE SAMPLE ID:
FAIL-CLOSED

SPLIT SEED:
17486

SPLIT FRACTIONS:
80 / 20

EXPECTED PROTOCOL-TRAIN:
9139
NEG 4259 / NEU 366 / POS 4514

EXPECTED PROTOCOL-DEV:
2285
NEG 1065 / NEU 92 / POS 1128

MATERIALIZER:
IMPLEMENTED / REAL RUN PENDING

ARTIFACT SCHEMA:
preg1-split-v1

REAL MEMBERSHIP:
NOT OBSERVED

OFFICIAL VALIDATION:
UNTOUCHED MEASUREMENT-DEV

OFFICIAL TEST:
SEALED

HEAD TRAINER:
NOT IMPLEMENTED

DOWNSTREAM SCORE:
NONE

STAGE-1 TRAINING:
NOT RUN

D-B3B0-002:
OPEN

PDF:
STALE

LOCAL TESTS:
2199 passed, 56 skipped (targeted: 145 profiling + 55 split)

COMMIT CREATED:
NO
```
