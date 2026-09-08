# Audit 053 - Stage-2 Clean 10-Head Campaign Runtime Closeout

**Scope:** record the completed frozen Stage-2 clean 10-head campaign — ten
independent heads over the four verified clean caches — and the state transition
it produces.
**Date:** 2026-09-08
**Type:** **documentation-only audit.** No implementation, test, protocol
constant, scientific hyperparameter, corruption setting, Stage-2 A/B policy,
decision record or runner code is changed. **No new scientific decision is
authorised or appended.**

---

## 1. Executive verdict

**PASS.**

The frozen ten-run campaign that Audit 049 specified, Audit 051 accepted at
runtime and Audit 052 supplied verified inputs for has now **executed to
completion**: 10 of 10 heads, 5 per arm, every one over clean `protocol-train`
with within-head epoch selection on clean `protocol-dev`. The registry reports
`complete = true`, `pending = []`, `winner = null`.

Stage-2 head training is **complete**. Stage-2 measurement has **not begun** and
is still blocked on a value nobody has decided.

| | |
|---|---|
| Audit 049 remains protocol authority | **YES** — nothing here changes it |
| Audit 051 runner acceptance | **PASS** — the accepted runner executed this |
| Audit 052 cache/manifest acceptance | **PASS** — its four caches are the inputs |
| Ten heads completed | **YES** — 10 of 10, 5 per arm |
| Clean `protocol-dev` results observed | **YES** — required by the frozen selection rule, §11 |
| `measurement-dev` read | **NO** |
| Degraded measurement results | **NONE EXIST** |
| Official TEST | **SEALED** — not read, structurally unnameable |
| A/B selection, winner or ranking | **NONE** |
| Measurement corruption seed | **STILL UNRESOLVED** — not invented here, §13 |
| New scientific decision appended | **NO** |

---

## 2. Starting state

Two distinct commits appear in this audit, and conflating them would misstate
where the science ran. They are kept apart from here on.

| | |
|---|---|
| **Actual scientific runtime Git HEAD** | `6693e728ccaebc987e4786bd5cd5e0f5c16143f7` |
| Campaign `repository_head` | `6693e728ccaebc987e4786bd5cd5e0f5c16143f7` |
| All four clean-cache `repository_head` values | `6693e728ccaebc987e4786bd5cd5e0f5c16143f7` |
| **Documentation Audit-052 commit** | `dbc30be7b74ff6868623fe4f30477132277cc91b` |
| Campaign manifest **semantic** digest | `571100c365d7dd262966a8bb4b7ca89a2106da29e7d8062a084ada024f3cc7e6` |
| Accepted pre-training manifest **file** SHA-256 | `4a940d87f13c545d4b2f95ecbf71f31260d900fb1e0e60f9e5ca686504e8e962` |

**The campaign ran at `6693e728…`, not at `dbc30be7…`.** The execution notebook
checked out `6693e728ccaebc987e4786bd5cd5e0f5c16143f7` **detached**, and asserted
`git rev-parse HEAD == 6693e728ccaebc987e4786bd5cd5e0f5c16143f7` both **before**
and **after** training. Runtime Git HEAD, the campaign's `repository_head` and the
`repository_head` of all four clean caches are therefore **one and the same
value**, checked rather than assumed.

`dbc30be7…` is the **documentation-history commit** — the Audit-052 closeout. It
is not the scientific runtime HEAD and was never the checked-out state during
training. It appears in this audit only as a documentation identity: in the
evidence path segment `audit052-dbc30be7` (§14), and as the HEAD of the working
tree in which this audit file is being written (§17).

**Audit 052 §14.1 was followed** in the strongest available form. That section
warned that `repository_head` is always caller-supplied and never derived from
git, so a campaign given the wrong value would fail closed on "a campaign may not
mix commits". The operator did not merely supply the right value alongside a
different checkout: they **checked out the exact execution SHA** and passed that
same identity into the campaign runner, so the supplied value and the runtime
HEAD could not diverge. See §16.3.

The tree at `6693e728…` is also byte-identical, outside documentation, to the one
Audit 051 §15 accepted: `dbc30be7`'s diff against `6693e728` touches **two
markdown files only** — the Audit-051 and Audit-052 documents — and no file under
`unmark/`, `tests/`, `docs/spec/`, `scripts/` or `configs/`.

---

## 3. What this audit does and does not change

| | |
|---|---|
| Implementation | **unchanged** — no file under `unmark/` modified |
| Tests | **unchanged** |
| Protocol constants | **unchanged** — `stage2-dual-finalist-protocol-v1` |
| Scientific hyperparameters | **unchanged** |
| Corruption settings | **unchanged** |
| Stage-2 A/B policy | **unchanged** — D-S2-001 option (c) intact |
| `docs/spec/decisions.md` | **untouched** — no decision appended |
| Runner code | **unchanged** — called, not edited |
| Files added | one: this audit |

No repository test was executed for this audit and none was needed: no code
changed. The standing test evidence is Audit 051 §12 and §15.

---

## 4. The frozen campaign as executed

Every parameter below is inherited, not chosen here.

| | |
|---|---|
| Arms | 2 — `UNMARK-A`, `UNMARK-B` |
| Seeds | 5 — `53148, 59945, 42941, 720, 9428` |
| Heads | `2 × 5 = 10`, independent |
| Epochs per head | `30` |
| Learning rate | `0.01` |
| Batch size | `128` |
| Early stopping | **none** |
| Fitting data | clean `protocol-train` **only** |
| Checkpoint / epoch selection | clean `protocol-dev` **only** |
| A/B selection | **none** |

This is exactly the scope Audit 052 §16 authorised when
`READY_FOR_STAGE2_TRAINING` moved to `YES`, and exactly the plan Audit 051 §15
observed under synthetic caches. Nothing outside that scope was executed.

**All ten runs completed.**

---

## 5. The ten runs and their selected epochs

| # | Arm | Seed | Selected epoch |
|---|---|---|---|
| 1 | `UNMARK-A` | `53148` | `30` |
| 2 | `UNMARK-B` | `53148` | `11` |
| 3 | `UNMARK-A` | `59945` | `27` |
| 4 | `UNMARK-B` | `59945` | `26` |
| 5 | `UNMARK-A` | `42941` | `9` |
| 6 | `UNMARK-B` | `42941` | `19` |
| 7 | `UNMARK-A` | `720` | `26` |
| 8 | `UNMARK-B` | `720` | `22` |
| 9 | `UNMARK-A` | `9428` | `25` |
| 10 | `UNMARK-B` | `9428` | `25` |

**These are within-head selections only.** Each value is the epoch that one head,
at one seed, on one arm, scored highest on its own clean `protocol-dev` history
under the frozen total order — highest macro-F1, then highest accuracy, then
earliest epoch. **They MUST NOT be read as an arm comparison**, a quality signal,
or evidence that one arm converges faster. Comparing the two columns is not a
permitted operation and produces no meaning under D-S2-001 option (c).

Two readings a reviewer might reach for, and why both are wrong:

* **Run 1 selected epoch `30`, the last one.** This is not evidence of truncation
  or of a run still improving when it stopped. All 30 epochs ran — there is no
  early stopping, and `require_full_schedule` checks the evidence rather than a
  flag — so epoch 30 was selected out of a complete 30-epoch history like any
  other. A best-at-the-end history is an ordinary outcome, not a warning sign.
* **Runs 9 and 10 both selected epoch `25`.** A coincidence between two
  independent heads at a shared seed. It is not agreement, not a tie, and not a
  comparison; nothing in the protocol reads across arms.

Every selected epoch lies within `[1, 30]`, and the ten runs cover exactly five
distinct seeds twice each, once per arm.

---

## 6. Scientific artifact digests

These ten values are **canonical artifact-payload SHA-256 digests**: the closeout
notebook computed each as `canonical_json_sha256(artifact)` — SHA-256 over a
deterministic canonical JSON serialisation of that run's `Stage2HeadArtifact`
payload.

**They are not raw on-disk file hashes** of the Drive artifact files, and are not
presented as such. The supplied authoritative evidence establishes the payload
digests; it does not establish byte hashes of the stored files, so **no raw file
SHA-256 is reported here.** None has been invented or inferred. Should raw file
digests later be produced from authoritative evidence, they belong in a separate
record alongside — never substituted for — the payload digests below.

| Arm | Seed | Canonical artifact-payload SHA-256 |
|---|---|---|
| `UNMARK-A` | `53148` | `75ed0db7158e8f4912689d19923fb43f2bb6887d8592f942cb3adca3ddf0208d` |
| `UNMARK-B` | `53148` | `c0379fff439efcf611599b97af18117f907794e48eb3cc945b604b2e55c0ab19` |
| `UNMARK-A` | `59945` | `a4aa17f4453138fec90c8e1c6681753bb1b13ccf8e252f1654fbda6ab9609932` |
| `UNMARK-B` | `59945` | `4befb7ccf887bec51f5b5086a70a269da11ad10df154ba8c2146529ae9c55827` |
| `UNMARK-A` | `42941` | `e8e0108938e0385e65879c33c64f35a342ca69f78025d44224f5ee468cf3ec43` |
| `UNMARK-B` | `42941` | `afe12275d4fa9bb6bdf9f4d337807ed9895d8820fc947fc3de742ad0d7f4113c` |
| `UNMARK-A` | `720` | `7c8c2996be41543bfa223d0d697dd73dae82aafc4c5a6cba0f4c2ecdd9f40a48` |
| `UNMARK-B` | `720` | `4defc22b40c59858e778891f057749fc79e086cdc93f7bc94b8a7fa333e88021` |
| `UNMARK-A` | `9428` | `20caea0a7bbb4e94c004e9f90717aa806a01ea213e1f56646d59c53bf6020091` |
| `UNMARK-B` | `9428` | `b5f5417343846e678d87c0be5c024c540864b1a46f0fd6c73320fed251a2e4ee` |

All ten are distinct, and all ten are **new to the repository** — this audit is
their first record. Being payload digests, they are deterministic functions of
the artifact content: two runs producing identical payloads would digest
identically regardless of file layout, and any drift in a bound field changes the
digest.

Each payload binds the closed 23-field artifact schema of Audit 051 §8, which
refuses an A artifact loading as B, a seed mismatch, a checkpoint sha that is not
that arm's, a drifted LR or epoch budget, a selection role or condition that is
not clean `protocol-dev`, a cache key from another arm or commit, any missing
field and any unknown field — and which requires `measurement_used_for_selection`
and `ab_selection_performed` to both be `false`.

These artifacts are the **exact scientific authority** for this campaign.

---

## 7. Campaign registry final status

| | |
|---|---|
| Expected runs | `10` |
| Completed runs | `10` |
| Pending runs | `[]` |
| `complete` | `true` |
| `winner` | `null` |
| `ranks_arms` | `false` |
| Both arms carry all five frozen seeds | **YES** |

`require_campaign_complete` refuses to report a partial campaign — both arms and
all five seeds, or nothing — so a `complete = true` status is itself the pairing
guarantee, not a claim beside it.

`campaign_status` is a **completion report, not a comparison**: it carries counts
and pending runs and no score, delta or ranking. `winner = null` and
`ranks_arms = false` are the two safety declarations whose purpose is to record
the absence of a thing.

---

## 8. Resume contract

| | |
|---|---|
| Campaign-level resume | **enabled** |
| Completed runs on re-entry | **immutable, skipped** |
| `atomic_rerun_no_midrun_resume` | **preserved** |
| Optimiser or mid-head resume | **NOT introduced** |

This is the contract Audit 051 §8 argued for and did not weaken under execution:
a head run is atomic — complete, or absent. Resume happens between runs, never
inside one. Persisting optimiser state would have added a surface that could
reattach a partial run to a drifted identity while buying no scientific value,
because one run is deterministic and reproduces bit-for-bit. Nothing in this
campaign created that surface.

Re-entry remains permitted **only under a byte-identical manifest**, so a partial
campaign cannot absorb run state produced under different caches, commit or
protocol.

---

## 9. Manifest identity: two digests, one manifest

Audit 052 §10 recorded the manifest **file** SHA-256. This audit records the
manifest **semantic** digest as well. They are different values of different
things, and the difference is not a discrepancy:

| Digest | Value | What it is |
|---|---|---|
| File SHA-256 | `4a940d87f13c545d4b2f95ecbf71f31260d900fb1e0e60f9e5ca686504e8e962` | SHA-256 of the manifest JSON file as written to disk |
| Semantic digest | `571100c365d7dd262966a8bb4b7ca89a2106da29e7d8062a084ada024f3cc7e6` | the manifest's own `digest` — SHA-256 over the **serialised manifest content**, which re-entry compares |

The semantic digest is the identity `Stage2CampaignRegistry` enforces: a stored
campaign whose `manifest_digest` differs is refused rather than adopted. The file
digest is the on-disk byte identity of the artifact recorded in Audit 052. Both
are reported so that neither is mistaken for the other, and so a later reader
does not read Audit 052 §10 and this section as contradicting each other.

The accepted pre-training manifest file is unchanged from Audit 052: the same
`4a940d87…` bytes carried through to execution.

---

## 10. W&B observability

| | |
|---|---|
| Monitoring | enabled |
| Project | `UNMARK-stage2` |
| Group | `stage2-clean-10-head-6693e728ccae-571100c365d7` |
| Persistent W&B root | under `MyDrive/UNMARK/UNMARK-BACKUP/stage2-wandb/...` |
| Scientific authority | **NO** |

The group name decomposes exactly into the campaign's two identities — the
implementation SHA prefix `6693e728ccae` and the manifest semantic digest prefix
`571100c365d7` — so the telemetry is bound to this campaign and no other.

**W&B is not, and structurally cannot be, scientific authority here.** The
package `unmark/evaluation/` contains **zero** references to `wandb`, and the
repository enforces the separation with a source-level invariant — the scientific
process never imports `wandb` — under D-S1B-018, which places structured
telemetry in-process and W&B monitoring **out-of-process**. The runner that
produced the ten artifacts could not have consulted W&B even in principle.

The exact scientific authority remains the Drive registry and the ten head
artifacts of §6.

### 10.1 `epoch_history_logged=False` — a telemetry limitation, not a failure

`epoch_history_logged=False` was observed for the W&B syncs. Summaries and
artifact mirrors synced; the 30-epoch curves were **not** populated.

**This is an observability/telemetry limitation only.** It is:

* **NOT a scientific failure** — no scientific quantity is sourced from W&B;
* **NOT a protocol deviation** — the protocol specifies no telemetry backend, and
  every frozen requirement (30 epochs, no early stopping, clean-only selection)
  is checked by the runner against the run's own evidence, not against a chart;
* **NOT a reason to retrain.** The full 30-epoch history of every head is
  persisted inside that head's immutable artifact, together with its history
  digest, as Audit 051 §7 records.

**No retraining is required to repair W&B telemetry.** Any later curve sync must
be **post-hoc, read-only, from the immutable scientific artifacts** — never by
re-executing a head. Re-running to produce a prettier chart would replace ten
accepted artifacts with ten new ones for a non-scientific reason.

---

## 11. What changed in observed-results state

Stage-2 head training is complete, and that has one honest consequence.

**Clean `protocol-dev` metrics have necessarily now been observed**, because
every head used them for its own frozen within-head epoch selection. That is not
a leak and not a deviation — it is the frozen protocol executing exactly as
D-S2-001 specifies: `protocol-dev` exists for head-checkpoint selection on clean
`FULL`, and a head cannot be selected without scoring it.

A generic `DOWNSTREAM_RESULTS_SEEN=NO` is therefore **no longer accurate and is
not retained**. It is replaced by four specific statements, because the generic
flag conflates four separate things that now have different answers:

| | |
|---|---|
| `CLEAN_PROTOCOL_DEV_RESULTS_SEEN` | **YES** — required by the selection rule |
| `MEASUREMENT_DEV_READ` | **NO** |
| `DEGRADED_MEASUREMENT_RESULTS_SEEN` | **NO** |
| `OFFICIAL_TEST_READ` | **NO** |

Replacing the coarse flag with these four is a **precision improvement in the
audit record**, not a relaxation: nothing newly permitted, and three of the four
gates unchanged. Retaining `DOWNSTREAM_RESULTS_SEEN=NO` would have been the
inaccurate choice.

What clean `protocol-dev` observation does **not** license is any use of those
scores beyond within-head epoch selection. They may not compare arms, rank arms,
select an arm, tune a hyperparameter, or influence anything measurement-side.

---

## 12. A/B: no comparison may become a selection

| | |
|---|---|
| `A_B_SELECTION` | **NO** |
| `A_B_WINNER` | **NONE** |
| `A_B_RANKING` | **NONE** |
| Both arms | **mandatory** — neither may be dropped |

D-S2-001 adopted option (c) and D-S1B-001 stands with no exception. No winner
rule, tie-break, no-decision margin, significance test or p-value exists, and
none may be added. `aggregate_stage2_campaign` requires both arms and refuses a
report missing either, or one whose conditions do not cover all five seeds — a
dropped arm or seed is exactly what Audit 049 forbids.

**The selected epochs of §5 are within-head selections only and MUST NOT be
interpreted as an arm comparison.** This is restated here, next to the flags,
because §5 is the one table in this audit that a reader could misuse that way.

---

## 13. Measurement remains blocked

| | |
|---|---|
| `STAGE2_MEASUREMENT_CORRUPTION_SEED_PINNED` | **`False`** — unchanged |
| `measurement-dev` read | **NO** |
| Degraded measurement cache | **none exists** |
| Degraded measurement score | **none exists** |
| Official TEST | **SEALED**, not read |

The measurement corruption seed is **the exact next scientific blocker**, and
**this audit does not invent, choose, suggest or default it.** It remains what
Audit 051 §10 described: a value the project has not decided, which
`stage2_measurement_extraction_plan(corruption_seed=...)` refuses to default.

**Before any degraded measurement extraction, the author must make and record a
separate prospective measurement-seed decision or spec amendment.** That decision
must be made and recorded **before** `measurement-dev` is read and **before** any
degraded measurement score is produced — prospectively, not alongside or after
the first result. A seed chosen once numbers are visible is no longer a
prospective choice, which is the entire reason it was refused a default.

---

## 14. Durable evidence

```
MyDrive/UNMARK/UNMARK-BACKUP/stage2-training/6693e728ccae/audit052-dbc30be7/clean-10-head-campaign-v1/stage2-clean-10-head-campaign-final.json
```

| | |
|---|---|
| SHA-256 | `69dc1447f1c340c18285f363d1ebfaa7d5d93f954e487e2fd99c4f556e9faf59` |

The path encodes both identities this campaign depends on — the implementation
SHA prefix `6693e728ccae` and the Audit-052 documentation SHA prefix
`dbc30be7` — under the persistent `MyDrive/UNMARK/UNMARK-BACKUP` destination used
by Audits 050, 051 and 052. As Audit 052 §11 records, a Colab-resolved
`.shortcut-targets-by-id/...` rendering of this location is the resolved backing
path of that persistent destination, not a runtime-only local path.

Consistent with Audit 051 §13, no Drive path is hard-coded in repository code.

---

## 15. Reconciliation

| Source | Reconciled |
|---|---|
| `docs/audits/049-stage2-dual-finalist-protocol-freeze.md` | every executed parameter is the frozen one; option (c) intact; no adjudication exists |
| `docs/audits/050-stage2-dual-finalist-infrastructure-implementation.md` | the accepted forward pass produced the caches this campaign consumed; not re-entered during head training, which sees tensors only |
| `docs/audits/051-stage2-head-campaign-runner-implementation.md` | the runner accepted in §15 executed this campaign; the §8 resume contract held; the §7 selection rule produced §5 |
| `docs/audits/052-stage2-real-clean-cache-materialization-pre-training-acceptance.md` | its four verified caches and its manifest are the inputs; its §14.1 guidance was followed by pinning the checkout to the execution SHA (§16.3); its §16 scope was not exceeded |
| `docs/spec/stage2-dual-finalist-protocol.json` | protocol version, head, optimisation, seeds, split roles and selection rule all as pinned. **Two `state` fields are now stale — §16.1** |
| `docs/spec/decisions.md` | unmodified; no decision appended. D-S2-001 and D-S1B-001 stand with no exception |

---

## 16. Inconsistencies found

**Two open findings** (§16.1, §16.2), one **resolved safeguard** recorded so it
is not mistaken for an open risk (§16.3), and one **disambiguation** that exists
to prevent a wrong "fix" (§16.4). Neither open finding is a defect in the runner,
the caches or the artifacts, and neither is repaired here — both touch files this
task may not modify.

### 16.1 The Stage-2 protocol artifact's `state` block is now stale

`docs/spec/stage2-dual-finalist-protocol.json` still declares:

| Field | Value | Now |
|---|---|---|
| `state.stage2_started` | `false` | **stale** — Stage-2 training started and completed |
| `state.downstream_results_seen` | `false` | **stale under the broad reading** — clean `protocol-dev` scores have been observed (§11) |

Both are classified `safety_gate`, so correcting them is an author decision, not
a documentation edit. Both are also **pinned by a test** —
`tests/test_stage2_dual_finalist_infra.py`, in
`test_stage2_protocol_artifact_still_says_no_selection_or_training`, asserts each
is `False` — so a spec correction requires a paired test change. That is an
implementation change and out of scope here.

`state.downstream_results_seen` is the field whose ambiguity §11 addresses: under
"any downstream result" it is now false-as-written; under "any *measurement* or
reportable downstream result" it still holds. The audit record resolves this with
the four specific flags; **the spec still needs the author's decision.**

Every other `state` field remains accurate: `adjudication` is
`CLOSED_WITHOUT_SELECTION`, `downstream_may_select_a_vs_b` is `false`,
`final_adapter_selected` is `false`, `downstream_test` is `SEALED`,
`finalist_count` is `2`.

### 16.2 Two module constants now read as stale project state

`unmark/evaluation/stage2_dual_finalist.py` declares:

| Constant | Value | Reading |
|---|---|---|
| `STAGE2_TRAINING_STARTED` | `False` | **stale as project state** — training started and completed |
| `STAGE2_HEAD_TRAINING_IMPLEMENTED` | `False` | **stale as project state** — implemented in `stage2_head_campaign.py` (Audit 051) |
| `STAGE2_A_B_SELECTION_IMPLEMENTED` | `False` | accurate |
| `STAGE2_MEASUREMENT_DEV_SELECTION_IMPLEMENTED` | `False` | accurate |
| `STAGE2_OFFICIAL_TEST_REACHABLE` | `False` | accurate |

The ambiguity is scope. Read as **module-scoped** declarations — the test that
pins them is named for adding no training, selection or TEST pathway *to that
module* — all five remain literally true: `stage2_dual_finalist.py` still
contains no training loop, no A/B selection and no TEST path. Read as
**project-state** flags, which their names invite, the first two are now wrong.

Not repaired here: the constants are implementation and their pins are tests.
This audit makes the scope explicit instead, which is the part within its remit.

### 16.3 Audit 052's `§14.1` concern is closed by a provenance safeguard

Recorded as a **resolved safeguard, not a risk that materialised.** Audit 052
§14.1 observed that `repository_head` is caller-supplied and never derived from
git, so a campaign could in principle be given a value that disagreed with the
tree it ran on.

That gap was closed at the strongest point available: the execution notebook
**checked out `6693e728ccaebc987e4786bd5cd5e0f5c16143f7` detached** and asserted
`git rev-parse HEAD == 6693e728ccaebc987e4786bd5cd5e0f5c16143f7` **before and
after** training, then passed that same identity into the campaign runner. The
supplied `repository_head`, the runtime Git HEAD and the four caches' bound
`repository_head` are therefore one value, verified at runtime rather than
asserted in prose.

This is stronger than merely supplying the correct constant while sitting on a
different checkout. Supplying the right value alone would satisfy
`validate_stage2_campaign_manifest` and `run_stage2_campaign` — both compare the
caller's value against the manifest's — but would leave open the question of
which code actually executed. Pinning the checkout closes that question too.

**Carry the same practice into the measurement phase**, which will run further
from `6693e728…` in commit distance: check out the execution SHA, assert it, and
pass the asserted value. Nothing here needs repair.

### 16.4 Not an inconsistency — the Stage-1 freeze artifact must stay `false`

`docs/spec/stage1-adapter-finalists.json` also carries
`adjudication.downstream_results_seen = false`, and `unmark/stage1/finalists.py`
**enforces** it, raising if it is ever not `False`.

**That field is a historical claim and remains permanently true.** It asserts
that the Stage-1 finalist freeze was made *before* any downstream result existed
— the guard's own message says exactly that — and no later downstream
observation can change what was known at freeze time. It is the record of
D-S1B-001 being honoured.

Flagged explicitly because §16.1 could invite someone to "make the artifacts
consistent" by flipping this one too. **Doing so would falsify the Stage-1 record
and trip the guard.** It must stay `false`.

---

## 17. Final state

```
git status --short
 ?? docs/audits/053-stage2-clean-10-head-campaign-runtime-closeout.md
```

`git diff --check` produced no output. The **documentation** working tree's HEAD
is `dbc30be7b74ff6868623fe4f30477132277cc91b`, unchanged — this is the repository
in which this audit file is written, not the scientific runtime state, which was
`6693e728ccaebc987e4786bd5cd5e0f5c16143f7` throughout the campaign (§2).

This audit is **uncommitted and awaiting author review**. It adds exactly one
path — this file — and modifies none. Nothing has been staged, committed, pushed
or history-mutated.

---

## 18. Exact next allowed step

**Not** "measure." The next thing is a decision, not an execution.

1. **Make and record a prospective measurement corruption seed decision** — a
   `docs/spec/decisions.md` entry plus the corresponding spec amendment,
   authored **before** `measurement-dev` is read and **before** any degraded
   score exists. This audit does not choose it, narrow it, or suggest a value.
2. Only after that decision is recorded may degraded measurement extraction be
   planned at all.

Until step 1 exists, explicitly not permitted:

* reading `measurement-dev` for any purpose;
* extracting any degraded cache — `P25`, `P50`, `P75`, `P100`, `STRIP_ALL`;
* producing any measurement score;
* opening official TEST, which remains SEALED;
* choosing, promoting, dropping or ranking A versus B;
* re-running any completed head, including to repair W&B curves (§10.1).

---

```
STAGE2_PROTOCOL_FROZEN=YES
AUDIT049_PROTOCOL_AUTHORITY=UNCHANGED
AUTHORITATIVE_RUNNER_TORCH_ACCEPTANCE=PASS

FOUR_CLEAN_STAGE2_CACHES_VERIFIED=YES
STAGE2_CAMPAIGN_MANIFEST_VERIFIED=YES

STAGE2_TRAINING_STARTED=YES
STAGE2_TRAINING_COMPLETE=YES
STAGE2_HEAD_RUNS_COMPLETED=10_OF_10
UNMARK_A_HEADS_COMPLETED=5_OF_5
UNMARK_B_HEADS_COMPLETED=5_OF_5

CLEAN_PROTOCOL_DEV_RESULTS_SEEN=YES

STAGE2_AB_SELECTION_IMPLEMENTED=NO
A_B_SELECTION=NO
A_B_WINNER=NONE
A_B_RANKING=NONE

STAGE2_MEASUREMENT_CORRUPTION_SEED_PINNED=False
MEASUREMENT_DEV_READ=NO
DEGRADED_MEASUREMENT_RESULTS_SEEN=NO

DOWNSTREAM_TEST=SEALED
OFFICIAL_TEST_READ=NO

READY_FOR_STAGE2_TRAINING=COMPLETED
READY_FOR_STAGE2_MEASUREMENT=NO
```

`READY_FOR_STAGE2_TRAINING` moves from `YES` to `COMPLETED`: the work that flag
authorised is done, and it authorises nothing further.

`READY_FOR_STAGE2_MEASUREMENT=NO` is blocked on exactly one thing — the
unresolved measurement corruption seed (§13, §18). It is not blocked on the
runner, the caches, the heads or the artifacts, all of which are complete and
verified.

The generic `DOWNSTREAM_RESULTS_SEEN` flag is **deliberately absent**, superseded
by the four specific flags above for the reason given in §11. Its absence records
a distinction, not a relaxation.
