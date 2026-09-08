# Audit 052 - Stage-2 Real Clean-Cache Materialization and Pre-Training Acceptance

**Scope:** close the Stage-2 clean representation preparation gate opened by
Audit 051 §18 — verify the frozen derived TRAIN file, materialise the frozen
protocol split, extract and verify the four clean `FULL` representation caches,
and validate the ten-run campaign manifest.
**Date:** 2026-09-08
**Type:** **documentation-only audit.** No implementation, test, protocol
constant, corruption setting, runner behaviour or scientific decision is changed,
and no decision record is appended.

---

## 1. Executive verdict

**PASS.**

Audit 051 §18 pre-specified three steps and one gate: materialise and verify the
frozen split, extract exactly four clean `FULL` caches, verify their identities
and digests — and only then may `READY_FOR_STAGE2_TRAINING` transition from `NO`
to `YES`. All three steps executed on the authoritative A100 runtime at the
accepted implementation commit, and all three passed. **The gate therefore opens
in exactly the manner Audit 051 pre-committed to, not by a new judgement made
after seeing a result.**

Nothing downstream was observed. No head exists, no head was trained, no
measurement-dev row was read, and official TEST remains sealed.

| | |
|---|---|
| Audit 049 remains protocol authority | **YES** — nothing here changes it |
| Audit 050 runtime path | **PASS** — reused, not re-executed |
| Audit 051 runner acceptance | **PASS** — the runner used here is the accepted one |
| Frozen derived TRAIN verified | **YES** — §4 |
| Frozen split materialised | **YES**, and byte-identical to the historical PREG1 split — §5 |
| Four clean `FULL` caches extracted | **YES** — §9 |
| Four clean `FULL` caches verified | **YES**, second pass — §9 |
| Campaign manifest validated | **YES**, exact 10-run plan — §10 |
| Measurement caches created | **NO** — none, at any condition |
| Head built or trained | **NO** |
| Downstream metric observed | **NO** |
| Official TEST | **SEALED** — structurally unnameable |
| A/B selection | **NONE** |
| New scientific decision appended | **NO** |
| Measurement corruption seed | **STILL UNRESOLVED** — not invented here, §8 |

---

## 2. Starting state

```
branch : main
HEAD   : ff0713b18b23c8ef28bb76dd594e955c35215972
status : clean (git status --short produced no output)
```

| | |
|---|---|
| Authoritative scientific implementation executed | `6693e728ccaebc987e4786bd5cd5e0f5c16143f7` |
| Documentation-closeout HEAD before this audit | `ff0713b18b23c8ef28bb76dd594e955c35215972` |
| GPU | NVIDIA A100-SXM4-40GB |

`ff0713b1` is the commit of the Audit-051 runtime closeout. Its tree differs from
`6693e728` in **exactly one path** — the Audit-051 file itself, `311` insertions
and `56` deletions — and in no other file. The implementation,
tests, specs, scripts and configs at `ff0713b1` are byte-identical to those at
`6693e728`, so the code that produced the caches recorded here is the code Audit
051 §15 accepted. See §14 for the two consequences of that fact.

---

## 3. What this audit does and does not change

| | |
|---|---|
| Implementation | **unchanged** — no file under `unmark/` was modified |
| Tests | **unchanged** |
| Protocol constants | **unchanged** — `stage2-dual-finalist-protocol-v1` |
| Corruption settings | **unchanged** |
| Runner behaviour | **unchanged** — the accepted Audit-051 runner was called, not edited |
| Scientific decisions | **none appended** — `docs/spec/decisions.md` untouched |
| Files added | one: this audit |

No repository test was executed for this audit and none was needed: no code
changed. The standing test evidence is Audit 051 §12 (static) and §15
(authoritative A100, `144 passed, 0 failed, 0 errors, 0 skipped`).

---

## 4. Frozen derived TRAIN verification

The derived TRAIN file was verified against the value the repository already
pins, before anything consumed it.

| | |
|---|---|
| SHA-256 | `a20c0f7760f32dc48263a79d73ddf5363526c17e9de2afc32d8346b23444d301` |
| Rows | `11424` |
| Official validation read | **NO** |
| Official TEST read | **NO** |

That digest is not new to this audit. It is pinned in
`unmark/evaluation/preg1_protocol.py`, asserted in `tests/test_preg1_split.py`,
and recorded in Audits 022, 023 and 024 and in
`docs/experiments/preg1-uit-vsfc-internal-split.md`. The row count `11424` is the
same one Audits 022 and 023 record. **This is a re-verification of an already
frozen quantity, not a new measurement.**

---

## 5. Frozen split materialisation

The newly materialised membership is **byte-identical to the historical frozen
PREG1 split**. It was not re-derived under new parameters; the existing frozen
splitter reproduced the existing frozen assignment.

| Role | Count | ID-file SHA-256 |
|---|---|---|
| `protocol-train` | `9139` | `275ae66d16582418093a1f4500904faefedd5936bb5cf383c52be302e151172e` |
| `protocol-dev` | `2285` | `d342950ae183e6c08bfeecaeacfb0e42aaf3751c12dec0baf0ca515922ca5e31` |
| **total** | `11424` | — |

Both ID-file digests match the values recorded in Audit 023, Audit 024 and
`docs/experiments/preg1-uit-vsfc-internal-split.md`. The counts match
`expected_split_totals()` in `tests/test_preg1_split.py`, which asserts exactly
`{protocol-train: 9139, protocol-dev: 2285}` under the locked split seed `17486`
and tag `UNMARK-PREG1-SPLIT-UITVSFC-v1`. `9139 + 2285 = 11424`, the derived TRAIN
row count of §4.

The split is drawn from the **official train split only**. `measurement-dev`
(official validation) was not touched, and official TEST is structurally
unnameable — `Preg1Role` has exactly `protocol-train`, `protocol-dev` and
`official-validation`, and no `OFFICIAL_TEST` member.

---

## 6. Role digests and class balance

Both digests are order-sensitive by construction: `ordered_id_digest` is
SHA-256 over the ids joined in order, and `label_digest` is SHA-256 over the
integer labels joined in order — "a re-ordered label vector is caught".

### `protocol-train`

| | |
|---|---|
| `ordered_id_digest` | `2cad022dd4aabbc875e388030e453ba2479f1046bea884309398b79f7d878cbd` |
| `label_digest` | `1c1377b2cd8c8016765229079fb469c2ead5c6ec85229d986f4344fa817f6128` |

| Class | Count |
|---|---|
| negative | `4259` |
| neutral | `366` |
| positive | `4514` |
| **total** | **`9139`** |

### `protocol-dev`

| | |
|---|---|
| `ordered_id_digest` | `63192edf811f8249404597103dc5ba4f48bb7843d242a592d13764013c095860` |
| `label_digest` | `546817741453edcb968c4d5de3530e435c5de9374995c1665260e92aea07ca1c` |

| Class | Count |
|---|---|
| negative | `1065` |
| neutral | `92` |
| positive | `1128` |
| **total** | **`2285`** |

The four digests above are **new to the repository** — this audit is their first
record. The per-class counts are not new: Audit 023 records `protocol-train`
`9139` as `4259/366/4514` and `protocol-dev` `2285` as `1065/92/1128`, in the
same negative/neutral/positive order. They reconcile exactly.

---

## 7. The real label representation observation

The repository loader returned `builtins.str` for all `11424` labels. Raw values:

| Raw value | Count |
|---|---|
| `"0"` | `5324` |
| `"1"` | `458` |
| `"2"` | `5642` |

Execution-only notebook normalisation mapped these to integer labels through the
**already-frozen class semantics** — `docs/spec/stage2-dual-finalist-protocol.json`
pins `label_mapping` as `negative: 0`, `neutral: 1`, `positive: 2` — giving:

| Class | Global count | = train + dev |
|---|---|---|
| negative | `5324` | `4259 + 1065` |
| neutral | `458` | `366 + 92` |
| positive | `5642` | `4514 + 1128` |
| **total** | **`11424`** | `9139 + 2285` |

Every row reconciles with §6 and with Audit 024, which records the derived TRAIN
label distribution as `5324-458-5642` in negative/neutral/positive order.

**The mapping was unambiguous.** Each raw string is the decimal spelling of
exactly one frozen class index; there is no candidate second reading, no
tie-break, and no new mapping was chosen. Normalisation could not have altered
which class any row belongs to.

What this corrected was an **overly strict notebook assertion**,
`type(label) is int`, which rejected the string spelling of a correct label.

* It is **NOT a repository defect** — the loader's return type is unchanged and
  no repository contract requires `int` at that boundary.
* It is **NOT a scientific deviation** — the class semantics used are the frozen
  ones, and the resulting counts match the historical record exactly.
* **No repository code was changed.** The fix was confined to the execution
  notebook.

The `label_digest` values in §6 are computed over the normalised integers, which
is what `label_digest` requires and what the campaign will later re-derive from
the caller's label vector and compare.

---

## 8. Clean `FULL` preparation and the API-only placeholder seed

All `11424` real rows were verified to make clean `FULL` a no-op:

| Check | Result over all `11424` rows |
|---|---|
| `canonical_text == corrupted_text` | **true for every row** |
| `condition_probability == 0.0` | **true for every row** |

`prepare_stage2_unmark_input(...)` declares `corruption_seed: int` as a required
keyword parameter with **no default**, so a call mechanically requires an
integer even when the condition cannot use one. Execution therefore passed an
**API-only placeholder `0`**.

**That placeholder is not a scientific measurement corruption seed**, and it is
structurally incapable of becoming one:

1. `FULL` is defined in `unmark/corruption/conditions.py` as
   `CorruptionCondition("FULL", CorruptionScope.NONE, 0.0, ...)` — scope `NONE`,
   probability `0.0`. No seed value can change a no-op, which is why the two
   empirical checks above hold for every row.
2. `Stage2RepresentationKey.__post_init__` **refuses** a `FULL` key that carries
   any seed: "FULL is the clean condition and must carry no corruption seed". A
   clean cache key binding a seed cannot be constructed at all.
3. Accordingly, all four official clean cache keys have `corruption_seed=None` —
   enforced, not merely observed.

`STAGE2_MEASUREMENT_CORRUPTION_SEED_PINNED=False` **remains unchanged**. The
measurement corruption seed is still undecided and **was not invented here**. It
blocks degraded measurement-dev extraction only, exactly as Audit 051 §10 and §17
item 2 state; it did not block any step in this audit, because every cache here
is clean `FULL`.

---

## 9. The four accepted clean caches

Exactly four caches were extracted — precisely `stage2_training_extraction_plan()`,
which is 4 items: clean `FULL` × {`protocol-train`, `protocol-dev`} × {A, B}.

| Arm | Role | Condition | Tensor SHA-256 |
|---|---|---|---|
| `UNMARK-A` | `protocol-train` | `FULL` | `8efacd4b10af0b90ecde8b11c7fe25105fe9fe0a739bb75bc01c7f5af99b08fb` |
| `UNMARK-A` | `protocol-dev` | `FULL` | `6252ecd5400ffa8537aba46d267baafb31ae3cd148f63c57c02f50f251c60de0` |
| `UNMARK-B` | `protocol-train` | `FULL` | `0136c5e5108cd12788c78ce10d6aeced953cd0388d8ff10372953b64b29183b2` |
| `UNMARK-B` | `protocol-dev` | `FULL` | `ea0d31a95ee50a744b789ea1f40e50d5a31e4e0a3600beddcebef0a4e6e6c963` |

All four:

* shape `[count, 768]` — `count` is `9139` for `protocol-train`, `2285` for
  `protocol-dev`;
* FP32;
* finite;
* **second-pass cache identity verification PASS** — reloaded and re-checked
  after writing, not trusted from the write path;
* arm, finalist checkpoint sha256, role, commit, protocol, ordered-id digest and
  label digest bindings **exact**.

Exactness is what `require_compatible` means here: it diffs **every** field of
the 22-field `Stage2RepresentationKey` and refuses on any difference, with no
tolerance and no coercion, because "a cache reused across arms, commits,
protocols, roles or conditions produces a silent, plausible-looking result".

**No measurement cache was created**, at `FULL` or at any degraded condition.

---

## 10. Campaign manifest

The exact 10-run plan was validated: 2 frozen arms × 5 frozen measurement seeds,
paired at every seed.

| # | Arm | Seed |
|---|---|---|
| 1 | `UNMARK-A` | `53148` |
| 2 | `UNMARK-B` | `53148` |
| 3 | `UNMARK-A` | `59945` |
| 4 | `UNMARK-B` | `59945` |
| 5 | `UNMARK-A` | `42941` |
| 6 | `UNMARK-B` | `42941` |
| 7 | `UNMARK-A` | `720` |
| 8 | `UNMARK-B` | `720` |
| 9 | `UNMARK-A` | `9428` |
| 10 | `UNMARK-B` | `9428` |

| | |
|---|---|
| Manifest file SHA-256 | `4a940d87f13c545d4b2f95ecbf71f31260d900fb1e0e60f9e5ca686504e8e962` |

This is the plan Audit 051 §15 observed under synthetic caches, now bound to the
four real ones. `validate_stage2_campaign_manifest` accepted it, which means each
of the four cache slots carries the right arm and role, the training path is
clean `FULL`, no slot came from a different commit or protocol, and the two arms
bind **different** finalist checkpoint sha256 values.

**No A/B winner, selection or ranking exists.** No such rule is implemented, and
the manifest is a completion structure, not a comparison.

---

## 11. Durable evidence

| | |
|---|---|
| Final clean-cache evidence SHA-256 | `cc6da181e0c4b05f92c48e0dc3d05faace45e5980cbac72a87af235d57b3b4b0` |

The evidence and output tree is stored persistently beneath the user's
`MyDrive/UNMARK/UNMARK-BACKUP` location, the same durable destination used by
Audit 050 and Audit 051.

**On the displayed path.** Colab may render the resolved location as
`.shortcut-targets-by-id/...`. That is the **resolved backing path of the
persistent MyDrive destination** — the target a Drive shortcut points at — and
**not** a local runtime-only output path. The distinction matters: a
`/content/`-local path would vanish with the runtime and the evidence would not
be durable. It is durable.

Consistent with Audit 051 §13, no Drive path is hard-coded in repository code —
the paths appear in audit prose and execution notebooks only.

---

## 12. Negative evidence

| | |
|---|---|
| `measurement-dev` read | **NO** |
| Official TEST read | **NO** |
| Degraded measurement extraction | **NO** |
| Head built | **NO** |
| Head training | **NO** |
| Stage-2 campaign training | **NO** |
| Downstream metric observed | **NO** |
| A/B selection performed | **NO** |

Nothing in this audit produced a trained parameter or a downstream number. What
it produced is four tensors of frozen representations and one manifest.

---

## 13. Reconciliation

| Source | Reconciled |
|---|---|
| `docs/audits/049-stage2-dual-finalist-protocol-freeze.md` | protocol authority unchanged; the split, roles, pooling, head, optimisation, seeds and selection rule used here are the frozen ones; `option (c)` intact — no A-vs-B selection exists |
| `docs/audits/050-stage2-dual-finalist-infrastructure-implementation.md` | the accepted forward pass was **called**, not re-implemented; both finalist checkpoint sha256 values are the Audit-048/050 ones |
| `docs/audits/051-stage2-head-campaign-runner-implementation.md` | the runner accepted in §15 is the one executed here; §18 steps 1–3 are now discharged |
| `docs/spec/stage2-dual-finalist-protocol.json` | `stage2-dual-finalist-protocol-v1`, unmodified; label mapping, split fractions, split seed and measurement seeds all as pinned |
| `docs/spec/decisions.md` | unmodified; **no decision appended**. D-S2-001 and D-S1B-001 both stand with no exception |
| Audits 022 / 023 / 024 | derived TRAIN digest, `11424` rows, `9139`/`2285` membership, both ID-file digests and the per-class counts all match the historical record exactly |

**Supersession, recorded rather than rewritten.** Audit 051 §17 item 8 states
"no real representation cache exists" and "the frozen UIT-VSFC protocol split has
not been materialised yet". Both were true when written and are now discharged by
§5 and §9 of this audit. **Audit 051 is not edited**: this project keeps its
audit history append-only, so the supersession is recorded here.

---

## 14. Inconsistencies found

Two, both about commit identity rather than science. Neither is a defect in the
runner, the protocol or the caches.

### 14.1 The campaign must be run at `6693e728…`, not at the then-current HEAD

`repository_head` is bound into every `Stage2RepresentationKey`, so the four
caches carry `6693e728ccaebc987e4786bd5cd5e0f5c16143f7`. Two guards compare it:

* `validate_stage2_campaign_manifest` refuses a cache slot "produced at X but the
  campaign is Y: a campaign may not mix commits";
* `run_stage2_campaign` refuses a caller head that "does not match the campaign's".

`repository_head` is **never derived from git anywhere in the package** — it is
always caller-supplied. Meanwhile HEAD has already advanced to `ff0713b1`, and
committing this audit will advance it again.

**Consequence for the next execution:** the campaign must be given
`6693e728ccaebc987e4786bd5cd5e0f5c16143f7` as its `repository_head`, matching the
caches and the manifest. An operator who passes `git rev-parse HEAD` instead will
get a fail-closed refusal — which is the guard working correctly, not a bug, but
an unexplained stop wastes an A100 session.

This binding is to a **commit identity**, not to behaviour: as §2 records, the
implementation tree at `ff0713b1` is byte-identical to `6693e728`. The guard is
deliberately stricter than behavioural equivalence, and that strictness is not
questioned here.

### 14.2 Commit `ff0713b1` is described as an implementation fix but is not one

`ff0713b18b23c8ef28bb76dd594e955c35215972` carries the message
"fix head campaign runner implementation". Its diff against `6693e728` is exactly
one file — `docs/audits/051-stage2-head-campaign-runner-implementation.md` — with
no change to any file under `unmark/`, `tests/`, `docs/spec/`, `scripts/` or
`configs/`.

Recorded so that a later reader does not conclude the runner was modified after
its Audit-051 acceptance. **Git history is the author's to mutate; nothing is
proposed here.**

---

## 15. Final state

```
git status --short
 ?? docs/audits/052-stage2-real-clean-cache-materialization-pre-training-acceptance.md
```

`git diff --check` produced no output. `HEAD` is
`ff0713b18b23c8ef28bb76dd594e955c35215972`, unchanged.

This audit is **uncommitted and awaiting author review**. It adds exactly one
path — this file — and modifies none. Nothing has been staged, committed, pushed
or history-mutated.

The runtime evidence recorded above is about the scientific implementation commit
`6693e728ccaebc987e4786bd5cd5e0f5c16143f7`; this file is the record of it, not
part of what was tested.

---

## 16. What `READY_FOR_STAGE2_TRAINING=YES` authorises

It authorises **only** the already-frozen 10-head clean campaign:

| | |
|---|---|
| Arms | 2 — `UNMARK-A`, `UNMARK-B` |
| Seeds | 5 — `53148, 59945, 42941, 720, 9428` |
| Epochs | 30 |
| Learning rate | `0.01` |
| Batch size | `128` |
| Fitting data | clean `protocol-train` **only** |
| Within-head epoch/checkpoint selection | clean `protocol-dev` **only** |
| Early stopping | **none** |
| A/B selection | **none** |

It does **NOT** authorise:

* **degraded measurement-dev** — `P25`, `P50`, `P75`, `P100`, `STRIP_ALL` remain
  blocked while `STAGE2_MEASUREMENT_CORRUPTION_SEED_PINNED=False`;
* **choosing, promoting or dropping A vs B** — no winner rule exists and none may
  be added (D-S2-001 option (c), D-S1B-001 intact);
* **official TEST** — SEALED and structurally unnameable;
* **new tuning** of any kind;
* **new hyperparameters**;
* **inventing the measurement corruption seed.**

The flag names the campaign it opens and nothing beyond it. Head training is now
authorised because the inputs it consumes have verified identities — not because
the runner passed acceptance, which Audit 051 already established and which alone
was never sufficient.

---

## 17. Exact next allowed step

After author review and commit of this audit:

1. **Run the frozen 10-head Stage-2 campaign** over the already-verified four
   clean caches of §9, under the manifest of §10, passing `repository_head` =
   `6693e728ccaebc987e4786bd5cd5e0f5c16143f7` (§14.1).
2. **Persist every run and artifact** beneath `MyDrive/UNMARK/UNMARK-BACKUP`.
3. **STOP** before any measurement-dev extraction or scoring.

No measurement. No TEST. No A/B selection. No new value decided.

---

```
STAGE2_PROTOCOL_FROZEN=YES
AUDIT049_PROTOCOL_AUTHORITY=UNCHANGED
AUDIT050_RUNTIME_PATH=PASS
AUTHORITATIVE_RUNNER_TORCH_ACCEPTANCE=PASS

FROZEN_DERIVED_TRAIN_VERIFIED=YES
FROZEN_STAGE2_SPLIT_MATERIALIZED=YES
FROZEN_STAGE2_SPLIT_MATCHES_HISTORICAL=YES
FOUR_CLEAN_STAGE2_CACHES_EXTRACTED=YES
FOUR_CLEAN_STAGE2_CACHES_VERIFIED=YES
STAGE2_CAMPAIGN_MANIFEST_VERIFIED=YES

STAGE2_AB_SELECTION_IMPLEMENTED=NO
STAGE2_MEASUREMENT_CORRUPTION_SEED_PINNED=False
DOWNSTREAM_RESULTS_SEEN=NO
DOWNSTREAM_TEST=SEALED
STAGE2_TRAINING_STARTED=NO

READY_FOR_CLEAN_REPRESENTATION_EXTRACTION=COMPLETED
READY_FOR_STAGE2_TRAINING=YES
```

`READY_FOR_CLEAN_REPRESENTATION_EXTRACTION` moves from `YES` to `COMPLETED`: the
work that flag authorised is done, and it authorises nothing further.

`READY_FOR_STAGE2_TRAINING` moves from `NO` to `YES` under the condition Audit
051 §18 pre-specified — "only after step 3 passes" — and its scope is exactly §16
above.

`STAGE2_TRAINING_STARTED=NO` is unchanged and remains true: this audit trained
nothing. The next execution is the first that will change it.
