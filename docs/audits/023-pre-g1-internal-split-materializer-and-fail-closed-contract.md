# Audit 023 — pre-G1 internal split materialiser and fail-closed splitter contract

| | |
|---|---|
| **Audit id** | 023 |
| **Created (UTC)** | 2026-08-21 |
| **Baseline HEAD** | `819d09f2df95ca57444a63c86363e614b44ce458` (implementation task) |
| **Execution HEAD** | **`66f4522fa86e5f02f583204ddcad560a62b013c0`** (real materialisation) |
| **Scope** | Harden the committed splitter against three probe-found failures; build the deterministic split materialiser; **and — Revision 1 — close the real materialisation** |
| **Revision 1b** | **2026-08-21** — **temporal-truthfulness repair only.** §J limitation 7 claimed the reproduction recipe was "now committed"; it is recorded in this **pending, uncommitted** revision, and the notebook dependency stays live until the researcher commits. A related implicature in §R is qualified the same way. **Nothing else changed.** |
| **Revision 1a** | **2026-08-21** — **documentation-consistency repair only.** Header `Type` row now distinguishes the implementation-era task from the Revision-1 real-data closure; §K is split into **K.1 (historical)** and **K.2 (current)** so the implementation-era "no real membership" item reads as superseded rather than contradictory; over-literal "unchanged" claims about §§B–L are corrected. **No constant, hash, count, decision, membership, result or code changed.** |
| **Revision 1** | **2026-08-21** — **REAL SPLIT CLOSED.** The derived pool's exact bytes were reconstructed from historical notebook evidence, the split was materialised twice independently at HEAD `66f4522f…`, and the two runs are **byte-identical**. Verdict moves **IMPLEMENTATION PASS → FINAL PASS**. §§B–L are preserved substantively as the implementation-era record, with explicit historical/discharged annotations where a claim would otherwise read as current; §§M–P carry the execution evidence. **No scientific constant, decision or implementation changed.** |
| **Predecessors** | [021](021-pre-g1-dataset-profile-and-protocol-precommit.md), [022](022-uit-vsfc-real-data-profile-integrity-closure.md) |
| **Phase** | pre-G1, after Audit 022 FINAL PASS |
| **Type** | **Implementation hardening + materialiser** (original task: **no real data**), **closed by Revision 1 with externally observed real-data split-materialisation evidence**. At no point: no model weights, no optimizer, no training, no HPO, no classifier execution, no downstream score |

---

## A. VERDICT

**FINAL PASS — REAL SPLIT MATERIALISED AND BYTE-DETERMINISTIC**

Five distinct things were established, in order, and they must not be collapsed:

| Stage | What it established | Where |
|---|---|---|
| 1. **Implementation verification** | splitter hardened against three probe-found failures; materialiser built; 55 executable tests. **No real data at that task.** | §§B–L (implementation-era record) |
| 2. **Reproducibility-chain closure** | the Audit-022 derived csv was not retained; its exact bytes were reconstructed from historical notebook evidence and **matched the locked digest** | §M |
| 3. **First real materialisation** | the split ran on the approved pool at HEAD `66f4522f…` and hit the precommitted aggregates **exactly** | §N |
| 4. **Independent determinism** | a second run into a fresh directory produced **byte-identical** artifacts | §O |
| 5. **Safe persistence** | five files to Drive; **no corpus text, no derived csv, no raw data** | §P |

**The state transition is:**

**IMPLEMENTATION PASS — REAL SPLIT NOT YET MATERIALISED → FINAL PASS — REAL
SPLIT MATERIALISED AND BYTE-DETERMINISTIC**

**§§B–L below are preserved substantively as the implementation-era record**,
with explicit historical / discharged annotations (§I, §L, §K.1) wherever a
sentence would otherwise read as a current claim. They describe a task in which
no real data was touched, and that remains true **of that task**. Nothing is
back-dated, and nothing is erased.

**Nothing downstream has happened.** No head was trained, no optimizer created,
no model weights loaded, no classifier executed, no Vanilla-vs-Base-only result
produced, no Stage-1 training or HPO run. The split is a **membership**, not a
result.

**2199 local tests pass, 56 skip** — 55 of them new in `tests/test_preg1_split.py`
(2144/56 at Audit 022).

| Item | State |
|---|---|
| Splitter | **hardened, single implementation** — no competing splitter was created |
| Fraction-mapping insertion order | **invariant** |
| Record input order | **invariant** |
| Conflicting-label canonical group | **fail-closed** |
| Duplicate sample id | **fail-closed** |
| Materialiser | **implemented and executed on real data** |
| Real membership | **materialised, verified, byte-deterministic** |
| Assignment digest | `7bd5d1892e23b96035c376936d2168f547661da07b8abc769f5656e9438f4f84` |

**No head trainer. No downstream score. No Stage-1 training or HPO.**
**D-B3B0-002 remains OPEN. Final Stage-2 pooling remains OPEN.** The compiled
PDF **remains stale**.

---

## B. PHASE BOUNDARY FROM AUDIT 022

Audit 022 closed at **FINAL PASS**: the derived pre-G1 pool is verified
(11 424 rows, digest `a20c0f77…`, zero duplicate / conflicting / cross-split
groups), the channel densities are measured, and the tokenizer geometry is
characterised. Audit 022 §L explicitly deferred the 80/20 split to a later task
**after researcher review**.

This was that task, minus the run: it built the mechanism and refused to use it
on real data, because the probes found the mechanism was not yet trustworthy.

*(Revision 1: the run has since happened — §§M–P. Sections B–L are preserved
substantively as the implementation-era record, with explicit historical /
discharged annotations where a claim would otherwise read as current, and
describe the state at that time.)*

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

**No real membership ids are hard-coded anywhere** — that remains true of the
code and of this audit. *(At the implementation task none had been observed at
all; the real membership was subsequently materialised — §§N–P — and lives only
in the persisted artifacts, never in committed source or documentation.)*

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

- *(Historical, at the implementation task: the real split had not been
  materialised and no real UIT-VSFC data had been read. **Superseded by
  Revision 1** — §§M–P. Nothing was downloaded to this local environment at any
  point; the real run was Colab-only.)*
- **No head trainer.** No optimizer, no checkpoint, no LR selection.
- **No downstream score.** No Vanilla-vs-Base-only result exists.
- **No Stage-1 training or HPO.** Stage-1 is untouched.
- **No locked value changed** — fractions 80/20, seed 17486, tag
  `UNMARK-PREG1-SPLIT-UITVSFC-v1`, `max_length` 256, dataset, task, split roles
  and seal all verified unchanged from the committed modules.
- **No prohibited git operation.** Nothing staged, committed, tagged or pushed.

---

## J. LIMITATIONS

1. *(Settled by Revision 1.)* At the implementation task the precommitted counts
   were verified only on synthetic pools, and "should carry over" was doing the
   work. **The real run reproduced 9139 / 2285 and the exact per-class counts.**
2. **The digest gate is still the only thing tying this to the approved corpus**,
   and this environment cannot check that a digest belongs to the corpus it
   claims to. §M substantially strengthens the chain — the ten official raw
   files re-verified, and the derived bytes rebuilt from them — but the argument
   remains hash-based, which is what a digest is.
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
7. **The derived csv still is not persisted** (Revision 1). §M is the recipe for
   rebuilding it, and it worked — but it worked partly because a notebook still
   existed to inspect. The recipe is now recorded in **this pending,
   uncommitted documentation revision**; **once this closure is committed**,
   reproducing the derived view will no longer depend on rediscovering the
   historical notebook. Until then the dependency is still live. The underlying
   tension between "commit no corpus" and "be reproducible" is managed, not
   eliminated.
8. **Revision 1 rests on evidence this session did not produce.** §Q states
   exactly what was and was not checked here. The arithmetic and byte-size
   cross-checks are strong and all passed, but a run that was internally
   consistent and wrong would pass every one of them.

---

## K. TASK-END SELF-AUDIT

### K.1 — original implementation-task self-audit (HISTORICAL RECORD)

> **Rows 1–33 are the self-audit as performed at the implementation task**, when
> no real data had been touched. They are preserved as the historical record and
> are **not** restated as current. In particular **row 20 was true at that time
> and is superseded by Revision 1** (§§M–P), where the real membership was
> materialised. Nothing here is erased.

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
| 20 | **No real membership materialised** | **yes at the implementation task** — **SUPERSEDED by Revision 1 §§M–P**, where the real split was materialised, independently reproduced and persisted |
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

### K.2 — Revision 1 self-audit (CURRENT)

| # | Check | Result |
|---|---|---|
| 34 | Audit 023 revised **in place**, not duplicated; **no Audit 024** | **yes** |
| 35 | Original implementation-only state preserved as historical fact | **yes** — §§B–L preserved substantively, with §I, §K.1 and §L explicitly annotated historical/discharged |
| 36 | Real materialisation evidence clearly separated from implementation | **yes** — §§M–P, and §A stages them |
| 37 | All reported hashes and counts copied exactly from the observed evidence | **yes** — transcribed, then arithmetically cross-checked (§Q) |
| 38 | `5bf858…` described as the **adapter TRAIN csv** SHA, **not** a code hash | **yes** — §M states it explicitly |
| 39 | Assignment digest exactly `7bd5d1892e23b96035c376936d2168f547661da07b8abc769f5656e9438f4f84` | **yes** — §§A, N, O, and the status block |
| 40 | Deterministic vs runtime artifacts correctly distinguished | **yes** — four vs one; §O proves the separation empirically |
| 41 | **Both** independent runs recorded | **yes** — §N and §O |
| 42 | Safe Drive persistence recorded, with the copied/not-copied boundary | **yes** — §P |
| 43 | Failed reconstruction probe recorded honestly, not hidden, not called corruption | **yes** — §M, framed as forensic evidence and as the mechanism working |
| 44 | No digest lock weakened; no corpus identity substituted | **yes** — the near-miss was rejected, not adopted |
| 45 | Raw corpus text absent from all committed docs | **yes** — ids, digests, counts only |
| 46 | Official validation / test boundaries preserved | **yes** — neither was an input |
| 47 | No downstream, training, HPO or model-weight claim made | **yes** — §A and §S say so explicitly |
| 48 | D-B3B0-002 **OPEN**; Stage-2 pooling **OPEN** | **yes** — §S |
| 49 | Compiled PDF **STALE** | **yes** |
| 50 | No scientific constant silently changed | **yes** — seed, tag, fractions, digest, rows, exclusion re-verified from the modules |
| 51 | No unnecessary new decision invented | **yes** — §R explains why, and what was recorded instead |
| 52 | Committed code unmodified by this task | **yes** — documentation only |
| 53 | Worktree left unstaged; no git add/commit/push/tag/stash/reset/checkout/restore | **yes** |
| | **— Revision 1a: documentation-consistency repair —** | |
| 54 | Header `Type` row distinguishes implementation-era (no real data) from Revision 1 (externally observed real-data evidence) | **yes** |
| 55 | Header `Type` still states no weights / optimizer / training / HPO / classifier / downstream score **at any point** | **yes** |
| 56 | §K split into **K.1 (historical)** and **K.2 (current)**; row 20 preserved and marked superseded, not erased | **yes** |
| 57 | Over-literal "unchanged / exactly as written" claims about §§B–L replaced with "preserved substantively … with explicit annotations" | **yes** — three sites |
| 58 | Swept §§B–L for any other unqualified claim that real membership does not exist | **yes** — one found (§E "None have been observed") and scoped |
| 59 | No scientific constant, hash, count, decision, membership or code changed | **yes** — documentation only |
| | **— Revision 1b: temporal-truthfulness repair —** | |
| 60 | §J limitation 7 no longer claims the recipe is already committed | **yes** — states it is recorded in this **pending, uncommitted** revision, dependency still live until commit |
| 61 | Swept all three modified docs for other "already committed" claims | **yes** — one borderline implicature in §R qualified; three remaining uses verified true (workflow code **is** committed at HEAD `66f4522f…`; "no corpus file is committed" is standing policy) |
| 62 | No claim that this uncommitted documentation is already committed | **yes** |
| 63 | No hash, count, membership, constant, decision, result, code or verdict changed | **yes** — wording only |

---

## L. REQUIRED NEXT ACTION — DISCHARGED BY REVISION 1

*(Historical.)* The implementation task required: run
`scripts/materialize_preg1_split.py` on Colab against the derived train csv into
a **new** directory, and revise this audit in place with the observed manifest,
id-file digests and assignment digest. It required the run to reproduce
`protocol-train` **9139** (4259/366/4514), `protocol-dev` **2285**
(1065/92/1128), zero cross-part canonical leakage and `preg1-split-v1` — and
stated that any deviation would be a finding rather than a surprise to accept.

**Every one of those conditions was met.** See §§M–P. There was no deviation.

**The next phase is the pre-G1 Vanilla-vs-Base-only burden diagnostic**, which
requires a head trainer that does **not** exist. Nothing in this audit authorises
starting it.

---

## M. REPRODUCIBILITY-CHAIN CLOSURE — RECONSTRUCTING THE DERIVED CSV

**Evidence status.** Everything in §§M–P was **observed on Colab** and supplied
to this session. The artifacts are not in this repository and were **not**
independently opened, hashed or recomputed here. What this session did verify is
recorded in §Q.

### The gap

The Audit-022-approved derived TRAIN csv was **deliberately not retained as raw
data** — no corpus file is committed, by standing policy. That policy has a
cost, and this task paid it: to materialise the split, the exact bytes behind
digest `a20c0f77…` had to be reproduced from the official distribution.

### Re-verification of the official source

The official UIT-VSFC v1.0 public-distribution files were downloaded again.
**All ten official raw-file SHA-256 values matched the Audit-022 historical
records.** The starting point was therefore provably the same corpus.

The historical conflicting canonical group was independently recovered:

| | |
|---|---|
| Canonical digest | `a193a8ff49cc5ab43da189f9126aea19a0a0e9df1e16acc0a710cf7e880d0daa` |
| Members | `train:11293`, `train:11417` |
| Labels | `0` and `2` |
| Source texts | confirmed identical |
| Id convention | confirmed **zero-based** |

This independently reproduces the Audit-022 / D-PREG1-011 finding from the raw
distribution.

### The failed first reconstruction — recorded, not hidden

An initial controlled reconstruction probe **failed visibly**. It did not
reproduce the historical **five-digit zero-padded** id serialisation, so its
bytes did not match. **It wrote no authoritative derived output.**

This is **historical forensic evidence, not a scientific failure**, and it is
worth being precise about what it was and was not:

- **no digest lock was weakened** — the target digest `a20c0f77…` was never
  relaxed to accommodate a near-miss;
- **no new corpus identity was substituted** — a differently-serialised csv was
  rejected rather than adopted;
- **nothing was corrupted** — the scientific dataset was never at risk; a probe
  produced non-matching bytes and was discarded;
- the probe **failing** is the mechanism working. A reconstruction that silently
  accepted a different serialisation would have been the actual failure.

### Recovered historical adapter semantics

Forensic inspection of the historical Colab notebook recovered the exact adapter
that produced the Audit-022 inputs:

| | |
|---|---|
| Source | official `sents.txt` + `sentiments.txt` |
| Normalisation | **none** |
| Label transformation | **none** |
| Columns | exactly `id,text,label` |
| Writer | Python `csv.DictWriter` |
| Line terminator | **LF** |
| Id format | `<split>:<zero-padded five-digit zero-based index>`, e.g. `train:00000` |
| Round-trip | text and label **lossless**, verified |

**Historical adapter TRAIN** — 11 426 rows, **1 067 637 bytes**, SHA-256
`5bf8587343ef76231f14d57f1806387d387900c3cbc1635ecb24b97c248c9a9f`.

**`5bf858…` is the SHA-256 of the historical adapter TRAIN csv *bytes*.** It is
**not** a hash of source code, and must not be cited as one.

### Derived construction and the match

Consume the adapter train csv; exclude the **entire** conflicting group
(`train:11293`, `train:11417`); **no relabel**; retain all other rows in
original order; write with the same `csv.DictWriter`, fields `id,text,label`,
LF terminator; copy adapter dev/test byte-for-byte.

The reconstructed artifacts matched the historical Audit-022 bytes **exactly**:

| Split | Rows | Bytes | SHA-256 |
|---|---|---|---|
| **train** | 11 424 | 1 067 331 | **`a20c0f7760f32dc48263a79d73ddf5363526c17e9de2afc32d8346b23444d301`** |
| dev | 1 583 | 139 001 | `9c475c8998871c0c7317ee200b3e7db827128cd2dfec9de5c689aca299acc8d0` |
| test | 3 166 | 291 625 | `33b58c83a0783e45a12954f8aa761104d2ae0a59a81a641df066e356f6162910` |

Train label counts: `0` 5 324 · `1` 458 · `2` 5 642 — the locked values.

**This is a reproducibility-chain closure, not a protocol or scientific
change.** The previously missing upstream link was **recovered** from historical
notebook evidence. No locked digest was altered, no decision was revisited, and
the bytes that were approved in Audit 022 are the bytes that were split.

---

## N. FIRST REAL MATERIALISATION — S3P12

| | |
|---|---|
| **Repository HEAD** | `66f4522fa86e5f02f583204ddcad560a62b013c0` |
| **Materialiser** | `scripts/materialize_preg1_split.py` (committed, unmodified) |
| **Input digest** | `a20c0f7760f32dc48263a79d73ddf5363526c17e9de2afc32d8346b23444d301` |
| **Schema** | `preg1-split-v1` |
| **Seed tag / seed** | `UNMARK-PREG1-SPLIT-UITVSFC-v1` / **17486** |

### The observed split matched the precommitment exactly

| Part | negative | neutral | positive | total |
|---|---|---|---|---|
| `protocol-train` | 4 259 | 366 | 4 514 | **9 139** |
| `protocol-dev` | 1 065 | 92 | 1 128 | **2 285** |

These are **exactly** the aggregates §E recorded **before** any membership
existed, derived from committed class totals and the allocation rule. The
precommitment was falsifiable and was not falsified.

**Assignment digest:**
`7bd5d1892e23b96035c376936d2168f547661da07b8abc769f5656e9438f4f84`

### Independent verification observed

| Check | Result |
|---|---|
| train / dev disjoint | **PASS** |
| union equals all 11 424 approved derived TRAIN ids | **PASS** |
| each sample occurs exactly once | **PASS** |
| all 11 424 canonical groups singleton in the approved pool | **confirmed** |
| canonical cross-part leakage | **0** |
| conflicting-label groups | **0** |
| official validation as input | **no** |
| official test as input | **no** |
| downstream score / head / optimizer / model weights / training | **none** |
| raw text printed by the verification | **none** |

The singleton-group confirmation matters: it is the premise the precommitted
per-class arithmetic rested on, and it was verified on the real pool rather than
assumed from the Audit-022 duplicate counts.

### Deterministic artifacts

| File | SHA-256 | Bytes |
|---|---|---|
| `protocol-train.ids.txt` | `275ae66d16582418093a1f4500904faefedd5936bb5cf383c52be302e151172e` | 109 668 |
| `protocol-dev.ids.txt` | `d342950ae183e6c08bfeecaeacfb0e42aaf3751c12dec0baf0ca515922ca5e31` | 27 420 |
| `split-manifest.json` | `225b109ea5fa58476e98bdf050a42ca89f12c6df02b37a882dc09cdc958b3685` | 3 240 |
| `report.md` | `17a9a6f116b1277bc063ff53d0840e20cae1b034177d6d2ae014a6428ee20459` | 796 |

### Runtime metadata is NOT part of the scientific artifact

| File | SHA-256 | Bytes | Keys |
|---|---|---|---|
| `runtime-environment.json` | `0ed15fc3f717e1d316194021969ec6fc8288073de99a051980d94f6b86bc2c6e` | 168 | `note`, `platform`, `python` |

This file is **runtime evidence only**. It is excluded from the deterministic
membership set by design (§F), and §O is what demonstrates the design works:
run 2 omitted it entirely and the four deterministic files were unchanged.

---

## O. INDEPENDENT SECOND MATERIALISATION — S3P13

A second materialisation into a **different, previously absent** directory,
using the same approved derived TRAIN bytes, the same committed materialiser and
the same repository HEAD — **without** runtime-environment output.

It produced **exactly four** files, and **all four were byte-identical to run 1**,
with the same four SHA-256 values recorded in §N.

| | |
|---|---|
| Manifest semantic identity | **PASS** |
| Assignment digest | `7bd5d1892e23b96035c376936d2168f547661da07b8abc769f5656e9438f4f84` — identical |
| `protocol-train` membership | **identical** |
| `protocol-dev` membership | **identical** |
| Class counts | 4 259 / 366 / 4 514 and 1 065 / 92 / 1 128 — identical |

**PASS — THE REAL SPLIT IS BYTE-DETERMINISTIC ACROSS INDEPENDENT RUNS.**

This is the strongest evidence in the audit, and it is stronger than equal
counts. Identical aggregates would be consistent with a different assignment;
identical **bytes** are not. It also confirms the runtime/deterministic
separation empirically: run 1 wrote runtime metadata, run 2 did not, and the
scientific artifacts did not move.

---

## P. SAFE PERSISTENCE — S3P14

Destination:
`/content/drive/MyDrive/UNMARK/preg1-uit-vsfc-internal-split/preg1-split-v1-66f4522a-7bd5d189`

Exactly five files were persisted — the four deterministic artifacts plus
`runtime-environment.json`. **Post-copy SHA verification PASS for all five**,
against the digests in §N.

Persisted membership: `protocol-train` **9 139**, `protocol-dev` **2 285**,
disjointness **PASS**.

### The persistence boundary, explicitly

| Item | Copied |
|---|---|
| raw UIT-VSFC | **NO** |
| reconstructed derived TRAIN csv | **NO** |
| validation csv | **NO** |
| test csv | **NO** |
| any corpus-text artifact | **NO** |
| split membership ids | yes |
| scientific manifest | yes |
| report | yes |
| runtime metadata | yes |
| model weights loaded | **NO** |
| training performed | **NO** |
| downstream score | **NO** |

The reconstructed derived csv was **not** persisted, which is the same choice
Audit 022 made and the reason §M was necessary at all. That is a deliberate,
repeated cost: the recipe in §M is what makes it affordable, and it is now
recorded rather than living only in a notebook.

---

## Q. WHAT THIS SESSION VERIFIED, AND WHAT IT DID NOT

**Not verified:** this session did not execute the materialiser on real data,
did not open the Colab artifacts, and did not recompute any reported digest. All
of §§M–P is externally observed evidence.

**Verified here, from the committed code and arithmetic:**

| Check | Result |
|---|---|
| observed per-class counts equal `expected_split_counts()` computed by committed code | **exact** |
| observed totals equal `expected_split_totals()` | **exact** |
| 9 139 + 2 285 = 11 424 = `DERIVED_TRAIN_SIZE` | **exact** |
| per class: 4 259+1 065 = 5 324, 366+92 = 458, 4 514+1 128 = 5 642 | **exact**, matches the locked pool |
| **id-file byte sizes** — 9 139 × 12 = **109 668**, 2 285 × 12 = **27 420** | **exact** |
| committed `_id_file_body` writes `"<id>\n"`, sorted | confirmed |
| committed materialiser writes exactly those four deterministic files plus optional runtime | confirmed |
| repository HEAD equals the execution HEAD `66f4522f…` | confirmed |
| all three reported digests are well-formed 64-hex | confirmed |

The **id-file size check is an independent cross-check I derived rather than
took**: `train:NNNNN` is 11 characters plus LF = 12 bytes, so the file sizes
confirm *both* the counts *and* the five-digit zero-padded id convention
recovered in §M. Two separately-sourced facts agree, which is worth more than
either alone.

**A byte delta consistent with the exclusion:** adapter 1 067 637 → derived
1 067 331 is **306 bytes** for the two removed rows. Consistent, not proof.

---

## R. DECISION-LOG REVIEW — NO NEW DECISION WARRANTED

**No new scientific decision was created**, and none should be. Execution of a
precommitted measurement is not a decision; the whole point of precommitting was
that running it would settle a question already framed. Specifically:

- **§M recovered a fact, it did not choose one.** The adapter serialisation was
  historical and was reconstructed, not decided. Had it been unrecoverable, or
  had a *different* recipe been adopted to reach the digest, that would have
  been a decision — and a serious one.
- **§§N–P confirmed a precommitment.** The aggregates were derived and recorded
  in §E before any membership existed. Matching them changes nothing.

What was recorded instead: an **empirical closure** note on
[D-PREG1-014](../spec/decisions.md#d-preg1-014--internal-split-materialisation-is-fail-closed-and-mapping-order-independent),
and the **reproduction recipe** appended to
[D-PREG1-011](../spec/decisions.md#d-preg1-011--conflicting-canonical-groups-are-excluded-whole),
which is the decision that defines the derived pool and was not reproducible
from committed information alone — and, until this revision is committed, still
is not. Both follow the existing `**Empirical result.**` precedent. D-PREG1-012 and D-PREG1-013 are untouched.

---

## S. STILL OPEN AFTER THIS CLOSURE

| | |
|---|---|
| [D-B3B0-002](../spec/decisions.md#d-b3b0-002--the-first-backbone-checkpoint-is-not-locked) | **OPEN** — the backbone checkpoint is a probe revision |
| Final Stage-2 pooling | **OPEN** |
| Compiled proposal PDF | **STALE** |
| Head trainer | **not implemented, never executed on downstream data** |
| Vanilla-vs-Base-only downstream result | **none exists** |
| Stage-1 HPO / training | **not run** |
| Official TEST | **sealed** for downstream scoring and protocol selection |
| Official validation | **measurement-dev; took no part in the internal 80/20 split** |

**A split is not a result.** This closure establishes which examples are in
which internal part, and nothing whatsoever about whether UNMARK works.

---

```
AUDIT 023 REVISED IN PLACE:
YES

NEW AUDIT CREATED:
NO

VERDICT:
FINAL PASS — REAL SPLIT MATERIALISED AND BYTE-DETERMINISTIC

TRANSITION:
IMPLEMENTATION PASS — REAL SPLIT NOT YET MATERIALISED
-> FINAL PASS — REAL SPLIT MATERIALISED AND BYTE-DETERMINISTIC

BASELINE HEAD (implementation):
819d09f2df95ca57444a63c86363e614b44ce458

EXECUTION HEAD (real run):
66f4522fa86e5f02f583204ddcad560a62b013c0

DERIVED CSV RECONSTRUCTION:
EXACT BYTE MATCH TO a20c0f77... (10/10 official raw hashes re-verified)

ADAPTER TRAIN CSV SHA (bytes, NOT code):
5bf8587343ef76231f14d57f1806387d387900c3cbc1635ecb24b97c248c9a9f

ASSIGNMENT DIGEST:
7bd5d1892e23b96035c376936d2168f547661da07b8abc769f5656e9438f4f84

BYTE-DETERMINISM ACROSS INDEPENDENT RUNS:
PASS

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

OBSERVED PROTOCOL-TRAIN:
9139
NEG 4259 / NEU 366 / POS 4514

OBSERVED PROTOCOL-DEV:
2285
NEG 1065 / NEU 92 / POS 1128

CROSS-PART CANONICAL LEAKAGE:
0

MATERIALIZER:
IMPLEMENTED AND EXECUTED ON REAL DATA

ARTIFACT SCHEMA:
preg1-split-v1

REAL MEMBERSHIP:
MATERIALISED / VERIFIED / PERSISTED (IDS ONLY)

RAW CORPUS PERSISTED:
NO

NEW SCIENTIFIC DECISION:
NONE WARRANTED

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

STAGE-2 POOLING:
OPEN

PDF:
STALE

LOCAL TESTS:
2199 passed, 56 skipped (targeted: 145 profiling + 55 split)

COMMIT CREATED:
NO
```
