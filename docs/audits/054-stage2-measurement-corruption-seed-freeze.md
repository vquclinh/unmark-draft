# Audit 054 - Stage-2 Measurement Corruption Seed Freeze

**Scope:** resolve the one Stage-2 scientific value Audit 051 §10 surfaced and
refused to invent — the measurement corruption seed — **prospectively**, before
any `measurement-dev` row is read and before any degraded score exists.
**Date:** 2026-09-08
**Type:** **prospective scientific decision.** `D-S2-002` is appended to
`docs/spec/decisions.md`. The protocol artifact and the Stage-2 measurement
guards change with it: the frozen seed constant, the plan, protocol and cache-key
guards (§7), and the provenance-only collator and extraction checks (§7a). **No
numerical forward computation changes. No measurement is executed.**

---

## 1. Executive verdict

**PASS.**

`STAGE2_MEASUREMENT_CORRUPTION_SEED = 19225`, frozen by **D-S2-002**.

The value is **inherited, not minted**: it is the pre-existing Stage-1
validation-corruption seed, itself *derived* from a namespace tag under the
locked root-seed scheme rather than chosen. It therefore predates every Stage-2
artifact, and no degraded Stage-2 result was in existence — let alone inspected —
when it was fixed.

**The decision freezes the measurement realisation. It does not execute
measurement.**

| | |
|---|---|
| Seed frozen | **YES** — `19225`, §4 |
| Frozen prospectively | **YES** — nothing degraded existed, §3 |
| Seed invented or substituted | **NO** |
| `measurement-dev` read during this task | **NO** |
| Degraded cache or score created | **NO** |
| Official TEST | **SEALED**, not read |
| A/B selection, winner or ranking | **NONE** — unchanged |
| Audit 049 protocol authority | **UNCHANGED** |
| Head seeds, artifacts, epochs, clean caches | **UNTOUCHED** |
| Deferred state-schema cleanup | **NOT** attempted, §11 |

---

## 2. Starting state and preconditions

```
branch : main
HEAD   : ceb903832cb57fea03e9492e9af6660ecddad9e0
status : clean (git status --short produced no output)
```

All three preconditions were verified **before** any edit:

| Precondition | Result |
|---|---|
| `HEAD == ceb903832cb57fea03e9492e9af6660ecddad9e0` | **YES** |
| Working tree clean | **YES** — `git status --short` empty |
| No `D-S2-002` already exists | **CONFIRMED** — the only `D-S2-*` in `docs/spec/decisions.md` was `D-S2-001`; no collision, no silent renumbering |

---

## 3. The prospective boundary

This is the property that makes the decision defensible, so it is stated as a
before/after rather than asserted in prose.

### Before this decision

```
STAGE2_MEASUREMENT_CORRUPTION_SEED_PINNED=False
MEASUREMENT_DEV_READ=NO
DEGRADED_MEASUREMENT_CACHE_CREATED=NO
DEGRADED_MEASUREMENT_RESULTS_SEEN=NO
OFFICIAL_TEST_READ=NO
```

### After this decision

```
STAGE2_MEASUREMENT_CORRUPTION_SEED=19225
STAGE2_MEASUREMENT_CORRUPTION_SEED_PINNED=True
```

**but still, unchanged:**

```
MEASUREMENT_DEV_READ=NO
DEGRADED_MEASUREMENT_CACHE_CREATED=NO
DEGRADED_MEASUREMENT_RESULTS_SEEN=NO
OFFICIAL_TEST_READ=NO
A_B_SELECTION=NO
A_B_WINNER=NONE
A_B_RANKING=NONE
```

Exactly two prospective state declarations changed: the measurement corruption
seed acquired the frozen value `19225`, and its pinned gate moved from `False` to
`True`. `measurement-dev`, the degraded caches and scores, official TEST and
A/B selection are all unchanged — everything the seed could conceivably have been
chosen to flatter is still `NO`, because none of it exists yet.

---

## 4. The frozen value and its provenance

| | |
|---|---|
| Frozen seed | **`19225`** |
| Inherited from | the Stage-1 **validation-corruption** seed |
| Namespace tag | `UNMARK-STAGE1-v1\|validation-corruption` |
| Pinned by | **D-S1B-005** — "two additional pre-use determinism pins" |
| Also pinned in | `docs/spec/stage1-final-freeze.json` |
| Derived, not chosen | **YES** — `derive_seeds(VALIDATION_CORRUPTION_SEED_TAG, 1)[0]` in `unmark/stage1/protocol.py` |

The provenance was located in the authoritative record rather than assumed.
D-S1B-005 carries the seed table in which `UNMARK-STAGE1-v1|validation-corruption`
resolves to **19225**, and `unmark/stage1/protocol.py` computes it from that tag
string, so it is recomputable by anyone and was never a free parameter.

**Why inheriting matters.** A freshly minted seed invites one question that has
no good answer — *how many were tried before this one?* Even asked in good faith
it cannot be refuted after the fact. A value that was precommitted for a
different purpose, under a scheme frozen long before Stage 2 existed, forecloses
the question instead of answering it. That is the whole argument, and it is worth
more than the arbitrary novelty a fresh integer would have bought.

**No Stage-1 seed-role collision is created.** D-S1B-005 requires the *seven
Stage-1 role seeds* to be distinct and asserts it at import time in
`unmark/stage1/protocol.py`. This decision adds no Stage-1 role and alters no
Stage-1 seed, so that assertion is untouched and still passes. The Stage-2
measurement corruption runs over a different corpus and is keyed per `sample_id`,
never by row order, so sharing the integer couples no two realisations over the
same samples. The reuse is deliberate and recorded, not an accidental clash.

---

## 5. `FULL` structurally carries no corruption seed

The frozen seed applies **only to the degraded conditions**:

| Condition | Binds `19225`? |
|---|---|
| `FULL` | **NO** — clean, binds `corruption_seed=None` |
| `P25` | **YES** |
| `P50` | **YES** |
| `P75` | **YES** |
| `P100` | **YES** |
| `STRIP_ALL` | **YES** |

This is structural, not a convention. `Stage2RepresentationKey.__post_init__`
**refuses** a `FULL` key that carries any seed — "FULL is the clean condition and
must carry no corruption seed" — and equally refuses a degraded key that carries
none. `FULL` is additionally defined in `unmark/corruption/conditions.py` with
scope `NONE` and probability `0.0`, so a seed could not affect it even if one
reached it.

The four already-verified clean caches of Audit 052 are `FULL` and therefore
**unaffected by this decision**. Their keys bind `corruption_seed=None`, they are
not rewritten, and the ten head artifacts of Audit 053 are untouched.

---

## 6. Spec amendment

Amended minimally, in the existing `measurement` section, preserving the file's
schema style and protocol identity.

| Path | Classification | Value |
|---|---|---|
| `measurement.corruption_seed` | `scientific_identity` | `19225` |
| `measurement.corruption_seed_pinned` | `safety_gate` | `true` |

Both carry a `note` in the file's established style. They are inserted in the
section's existing alphabetical key order, between `corruption_determinism` and
`excluded_conditions`. **No unrelated protocol content was rewritten**, and
`schema_version` remains `stage2-dual-finalist-protocol-v1`.

**A schema misreading was checked and rejected.** The artifact's identity field
is the top-level **`schema_version`**, plus a nested `protocol.version`. There is
**no** top-level `protocol_version` key, and none was added: an exploratory
notebook calling `spec.get("protocol_version")` would print `null` and that is a
property of the query, not a missing field. `require_frozen_protocol_spec()`
reads `schema_version`, which is the correct field and is unchanged.

---

## 7. Implementation

Two modules, with a deliberate asymmetry between them:

* **`unmark/evaluation/stage2_head_campaign.py`** carries the whole measurement
  guard — the frozen constant, the plan guard, the protocol validator, the
  measurement-key rule and the extraction-time provenance check;
* **`unmark/evaluation/stage2_dual_finalist.py`** carries **one provenance-only
  change**: the collator emits the actual corruption seed as non-tensor batch
  metadata. It is documented in §7a and is the only edit to that module.

**No numerical forward computation changed.** The forward-tensor computation and
its numerical semantics are untouched in both modules; the sole addition is
non-tensor provenance metadata, which the forward allowlist ignores.

### Constants

```
STAGE2_MEASUREMENT_CORRUPTION_SEED = 19225
STAGE2_MEASUREMENT_CORRUPTION_SEED_PINNED = True
```

`STAGE2_MEASUREMENT_CORRUPTION_SEED` is new and exported in `__all__`;
`STAGE2_MEASUREMENT_CORRUPTION_SEED_PINNED` moved from `False` to `True`.

### Function signature — deliberately unchanged

```
def stage2_measurement_extraction_plan(*, corruption_seed: int) -> ...
```

Keyword-only, **no Python default**, exactly as before. Freezing the value did
not turn it into a default, and that is the point: a default would let an
execution use the frozen realisation without ever naming it, so a later reader
could not tell from the call site whether the seed had been considered at all.
The caller must state the seed, and may state only the frozen one.

### Fail-closed behaviour

| Input | Behaviour |
|---|---|
| omitted | `TypeError` — missing required keyword-only argument |
| `True` / `False` | `EvaluationContractViolation`, "must be an explicit integer" |
| `"19225"`, `19225.0`, `None`, list | `EvaluationContractViolation`, "must be an explicit integer" |
| `19224`, `19226`, `0`, `-19225`, any other int | `EvaluationContractViolation`, **"the Stage-2 measurement corruption seed is frozen to 19225 (D-S2-002) and may not be overridden"** |
| `19225` | **accepted** — 12 requests |

The wrong-integer message names the frozen value, cites the decision, and states
that it may not be overridden, then explains the consequence: substituting
another integer would produce a different degradation realisation and report it
as the frozen protocol's.

**This plan-level guard is necessary but not sufficient on its own.** It refuses
a wrong seed at `stage2_measurement_extraction_plan`, but the lower-level APIs —
`prepare_stage2_unmark_input`, `collate_stage2_unmark_batch`,
`Stage2RepresentationKey`, `extract_and_cache_stage2_representations` — can be
called directly, and a caller doing so would never reach the plan. The
end-to-end guarantee is §7a; the claim made here is deliberately the narrow one.

### Protocol/implementation agreement

`require_frozen_protocol_spec()` — the module's single existing protocol
validator, already called by both extraction plans and by `stage2_campaign.py` —
now validates the amendment's **value and its pinned state**, in order:

| Check | Refusal |
|---|---|
| `measurement` present and a mapping | "no `measurement` section" |
| `measurement.corruption_seed` present, carrying a `value` | "does not pin `measurement.corruption_seed` (D-S2-002); a missing frozen field is a drifted artifact, not an unset default" |
| that value is an `int`, not a `bool` | "is not an integer seed (D-S2-002)" |
| that value equals `STAGE2_MEASUREMENT_CORRUPTION_SEED` | "The artifact and the implementation must agree, or a cache key would bind a degradation realisation the protocol never froze." |
| `measurement.corruption_seed_pinned` present, carrying a `value` | "does not declare `measurement.corruption_seed_pinned` (D-S2-002); a seed value without its resolved-state declaration is a half-applied amendment" |
| that value `is True` | "declares `measurement.corruption_seed_pinned` as `...`, not True. D-S2-002 resolved the seed; an artifact that still reports it unresolved contradicts the value it pins." |

**Why the pinned flag needs its own check.** Independent review found the first
version of this amendment validated only the seed *value*. An artifact carrying
`corruption_seed = 19225` beside `corruption_seed_pinned = false` would have
passed while asserting two contradictory things — a value is frozen, and no value
is frozen — and which one a reader believes would depend on which field they
happened to read. Both are load-bearing now.

**No protocol-reading logic was duplicated.** There is still exactly one loader,
one path constant and one validator; the added checks live inside it.

---

## 7a. End-to-end cache provenance

Independent review found that the plan-level guard alone did **not** deliver the
D-S2-002 guarantee. The pre-existing data path carried the seed only as far as
preparation:

```
prepare_stage2_unmark_input(...)  -> corruption_metadata["corruption_seed"]
collate_stage2_unmark_batch(...)  -> kept sample_ids and conditions, DROPPED the seed
extract_and_cache_stage2_representations(...)
                                  -> checked arm, checkpoint and row count,
                                     but could not see the seed that made the batch
```

So a caller bypassing the plan could prepare degraded examples under `19224`,
build a cache key declaring `19225`, and store a tensor whose key **falsely
claimed the frozen realisation**. The key was a claim *about* the tensor rather
than a property *of* it, and nothing downstream could tell the difference.

Three changes close it.

### 1. The collator carries the seed as provenance

`collate_stage2_unmark_batch` now emits one additional non-tensor field:

```
batch["corruption_seeds"] = [item.corruption_metadata.get("corruption_seed")
                             for item in inputs]
```

**No model tensor, dtype, shape, numerical input, tokenisation, adapter path,
encoder path or corruption operation changed.** The forward path moves an
allowlist — `_stage2_forward_tensors_on_device` iterates
`STAGE2_FORWARD_TENSOR_KEYS` and ignores every other key — so the new field is
structurally incapable of reaching the encoder. A test asserts both halves: that
`corruption_seeds` is absent from the allowlist, and that the moved tensors are
`torch.equal` to those from a batch with the field stripped out.

### 2. A measurement degraded key must bind the frozen seed

`Stage2RepresentationKey.__post_init__` gained one rule, scoped to the
authoritative Stage-2 measurement role:

| Role | Condition | Required `corruption_seed` |
|---|---|---|
| `official-validation` | `FULL` | `None` — unchanged |
| `official-validation` | `P25`/`P50`/`P75`/`P100`/`STRIP_ALL` | **exactly `19225`** |
| `protocol-train`, `protocol-dev` | any | **unchanged** — no D-S2-002 rule applies |

The scoping is deliberate. Historical synthetic fixtures build degraded
`protocol-dev` keys with arbitrary seeds, and retro-fitting D-S2-002 onto them
would break coverage that has nothing to do with measurement. Two parametrised
tests assert those roles are left alone.

### 3. Extraction verifies the batch against the key

`require_batch_provenance(batches, key)` runs inside
`extract_and_cache_stage2_representations` **before the torch import and before
any cache write**, so a mislabelled extraction is refused on any machine and
leaves no artifact behind:

| Condition | Refused |
|---|---|
| batch has no `conditions` | "carries no `conditions` provenance" |
| batch mixes conditions | "mixes conditions […]; one cache holds exactly one condition" |
| batch condition != key condition | "was prepared for condition `X` but the cache key declares `Y`" |
| degraded batch has no `corruption_seeds` | "carries no `corruption_seeds` provenance" |
| degraded seed list shorter than the rows | "provenance must cover every row" |
| degraded batch has a `None` seed | "at least one row has no recorded corruption seed; missing provenance fails closed" |
| degraded batch mixes seeds | "mixes corruption seeds […]; one cache holds exactly one degradation realisation" |
| degraded seed != key seed | "was prepared with corruption seed `X` but the cache key declares `Y`. Saving it would store a tensor whose key falsely claims a realisation it was not produced under." |

### `FULL` is deliberately exempt from the seed half

`prepare_stage2_unmark_input` requires an integer seed even for the clean
condition, where corruption is a structural no-op — `CorruptionScope.NONE`,
probability `0.0` — so a clean batch may legitimately carry an API-only
placeholder. That placeholder is **not** scientific identity:

* a `FULL` cache key still binds `corruption_seed=None`, which
  `Stage2RepresentationKey` enforces independently;
* `require_batch_provenance` checks only the *condition* for `FULL`, never the
  seed, so a clean batch carrying `0` — or `19224` — is accepted and that integer
  never becomes cache identity;
* a `FULL` batch with no seed provenance at all is accepted.

**The four historical clean caches of Audit 052 are therefore untouched and still
valid.** They are `FULL` with `corruption_seed=None`; a test reconstructs all
four key shapes and runs them through the new provenance check.

### What may now be claimed

Before this repair, the audit could honestly claim only that the *plan* refuses a
wrong seed. After it, the claim is stronger and is earned: a Stage-2 measurement
cache cannot be written under a key whose declared realisation differs from the
one that actually produced the tensor, whichever API the caller enters through.

### Why `stage2_dual_finalist.py` had to change after all

Audit 054 originally recorded that module as deliberately untouched. That
scope-out is **lifted for exactly one change**: the collator now carries
`corruption_seeds` through as non-tensor metadata. It was necessary because the
seed exists only in preparation output and the extraction driver is the only
place that can compare it against a key — without carrying it across collation,
there is nothing to compare. It is a **provenance change, not a numerics change**:
the forward-tensor computation and numerical semantics are unchanged, and the
only addition is non-tensor provenance metadata ignored by the forward
allowlist — proven by the allowlist itself and by the `torch.equal` test above.
The deferred state-schema items of §11 remain deferred.

---

## 8. Tests

Two test files. Counts below are derived from the working-tree diff by parsing
each file's top-level `test_*` functions with `ast`, not from prose:

| File | test functions at `ceb90383` | now | added | replaced | net |
|---|---|---|---|---|---|
| `tests/test_stage2_head_campaign.py` | 58 | 88 | **31** | **1** | **+30** |
| `tests/test_stage2_dual_finalist_infra.py` | 36 | 39 | **3** | 0 | **+3** |
| **total** | **94** | **127** | **34** | **1** | **+33** |

The one replaced function is
`test_the_measurement_corruption_seed_has_no_default`, whose assertion
`STAGE2_MEASUREMENT_CORRUPTION_SEED_PINNED is False` became untrue under
D-S2-002; it was superseded by a stronger set rather than deleted.

| Proven | Test |
|---|---|
| `STAGE2_MEASUREMENT_CORRUPTION_SEED == 19225` | `..._is_frozen_to_the_inherited_value` |
| `..._PINNED is True` | same |
| No default for `corruption_seed` | `..._still_has_no_python_default` — inspects `Parameter.empty` and `KEYWORD_ONLY` |
| `bool` rejected | `..._non_integer_measurement_seed_is_refused[True/False]` |
| non-int rejected | same, for `"19225"`, `19225.0`, `None`, `[19225]` |
| `19225` accepted | `..._frozen_measurement_seed_is_accepted_and_bound_correctly` |
| `19224` rejected | `..._wrong_integer_measurement_seed_is_refused[19224]` |
| `19226` rejected | same, `[19226]` |
| Exactly 12 requests | `..._plan_covers_two_arms_by_six_conditions`, `..._accepted_and_bound_correctly` |
| Exactly two arms | `..._is_two_arms_by_six_frozen_conditions_each` |
| Exactly six conditions per arm | same |
| `FULL` is `None` for both arms | `..._accepted_and_bound_correctly` |
| Every degraded request is `19225` | same |
| No silent substitution **at the plan level** | the wrong-integer parametrisation, six cases |
| No silent substitution **end to end** | the §7a provenance tests: a batch prepared under 19224 with a key declaring 19225 is refused before forward and before write |
| Artifact agrees with implementation | `..._protocol_artifact_and_the_implementation_agree_on_the_seed` |
| A drifted artifact is refused | `..._drifted_protocol_seed_is_refused_by_the_protocol_validator` |
| Artifact `corruption_seed_pinned=false` refused | `test_corruption_seed_pinned_false_is_refused` |
| Missing `corruption_seed_pinned` refused | `test_a_missing_corruption_seed_pinned_field_is_refused` |
| Missing `corruption_seed` refused | `test_a_missing_protocol_seed_field_is_refused` |
| Non-integer artifact seed refused | `test_a_non_integer_protocol_seed_value_is_refused` |
| The real amended artifact passes | `test_the_real_amended_artifact_passes_the_validator` |
| Measurement degraded key with `19225` succeeds | `test_a_measurement_degraded_key_must_bind_the_frozen_seed` |
| Measurement degraded key with `19224`/`19226` refused | `test_a_measurement_degraded_key_refuses_another_realisation`, and per-condition |
| Non-measurement roles unaffected | `test_non_measurement_roles_are_not_subjected_to_the_measurement_seed` |
| Collator preserves raw seed metadata | `test_the_collator_preserves_each_inputs_raw_corruption_seed` |
| Batch 19225 + key 19225 passes | `test_a_degraded_batch_matching_its_key_passes_provenance` |
| Batch 19224 + key 19225 fails before forward/write | `test_a_degraded_batch_prepared_under_another_seed_is_refused`, `test_the_driver_refuses_a_mislabelled_batch_before_any_write` |
| Mixed seeds fail | `test_a_mixed_seed_batch_is_refused` |
| Batch condition != key condition fails | `test_a_batch_condition_that_is_not_the_keys_is_refused`, `test_a_mixed_condition_batch_is_refused` |
| Missing degraded seed provenance fails | `test_missing_degraded_seed_provenance_fails_closed` |
| `FULL` key stays `corruption_seed=None` | `test_a_measurement_full_key_still_binds_no_seed` |
| `FULL` placeholder integer never becomes identity | `test_full_provenance_ignores_an_api_only_placeholder_seed` |
| Existing clean-cache contracts unchanged | `test_the_four_historical_clean_cache_shapes_remain_valid` |
| Added metadata never reaches the forward pass | `test_the_added_seed_metadata_never_reaches_the_forward_pass`, `test_the_seed_provenance_field_is_not_a_forward_tensor` |
| Official TEST structurally unreachable | pre-existing `test_official_test_has_no_role_member`, `test_the_module_names_official_test_in_no_executable_position` |
| No A/B winner/ranking/selection introduced | pre-existing `test_the_module_declares_no_selection_machinery`, `test_no_function_in_the_module_names_a_winner_or_ranking`, `test_the_aggregate_report_names_no_winner` |

The last two rows are deliberately **not** new tests. Those invariants already
had AST-level coverage, and re-asserting them beside the change would have proved
only that the new test agrees with itself. They were re-run instead.

### Results

The two smallest relevant suites:

```
.venv/bin/python -m pytest -q \
  tests/test_stage2_head_campaign.py tests/test_stage2_dual_finalist_infra.py

139 passed, 23 skipped
```

The broader Stage-2 regression set — all four Stage-2 files plus the Audit-050
infrastructure suite:

```
170 passed, 62 skipped
```

Whole repository:

```
.venv/bin/python -m pytest -q

4343 passed, 170 skipped in 154.62s (0:02:34)
```

Zero failures, zero errors.

**Measured against the same commit.** The baseline is a worktree checked out at
`ceb90383`, not a figure remembered from an earlier audit:

| | baseline `ceb90383` | this tree | delta |
|---|---|---|---|
| whole repository | `4294 passed, 168 skipped` | `4343 passed, 170 skipped` | **+49 / +2** |

**How `+49 / +2` reconciles with the function counts.** The two change sets add
**34** test functions and replace **1**, a net **+33** (§8 intro). Parametrisation
expands those into more collected cases than functions:

| File | collected cases at `ceb90383` | now | delta |
|---|---|---|---|
| `tests/test_stage2_head_campaign.py` | 74 | 122 | **+48** |
| `tests/test_stage2_dual_finalist_infra.py` | 37 | 40 | **+3** |
| **total** | **111** | **162** | **+51** |

`+51` collected is exactly `+49` passed and `+2` skipped, and matches the
whole-repository collected delta (`4462 -> 4513`). Net `+33` functions becoming
`+51` cases is the parametrised sets — six non-integer inputs, six wrong
integers, four wrong measurement seeds, five degraded conditions and two
non-measurement roles.

**The `+2` skips are accounted for exactly.** Of the three tests added to
`test_stage2_dual_finalist_infra.py`, two are `@requires_torch` — the collator
provenance test and the forward-pass-isolation test — because
`collate_stage2_unmark_batch` imports torch. The third,
`test_the_seed_provenance_field_is_not_a_forward_tensor`, is torch-free and runs
here. Nothing previously running became skipped.

**The skips are honest.** Every one of the 62 Stage-2 skips is torch-gated —
`torch is not installed locally; tensor checks run where available` — and
`import torch` raises `ModuleNotFoundError` in this environment. Per the
repository's established workflow, torch was **not** installed to eliminate them;
they stay collected and visibly skipped, and their runtime acceptance is the
separate step in §12.

One caveat on how the whole-repository baseline was obtained, recorded so the
numbers can be reproduced. The baseline worktree has no `.resources-cache/` of
its own — it is git-ignored and lives only in the main checkout, the same
condition Audit 052 §14 describes — so the pinned inventory was symlinked in to
let collection proceed. That symlink makes
`test_linguistics_eligibility.py::test_cache_location_is_repo_local_and_gitignored`
fail, correctly, because the cache was then not repo-local. The raw baseline run
printed `1 failed, 4293 passed`; the `4294` above adds back that one test, which
passes in the main checkout and is unrelated to this change. The failure is an
artifact of the measurement setup, not of the repository.

---

## 9. What was not touched

| | |
|---|---|
| Stage-2 head seeds `53148, 59945, 42941, 720, 9428` | **unchanged** |
| Clean head campaign artifacts (all ten) | **unchanged** |
| Selected epochs | **unchanged** |
| The four clean representation caches | **unchanged** — `FULL`, `corruption_seed=None` |
| A/B policy | **unchanged** — D-S2-001 option (c), D-S1B-001 intact |
| Final adapter state | **unchanged** — none selected |
| Stage-1 finalist historical artifact | **unchanged** |
| Stage-1 `downstream_results_seen=false` historical claim | **unchanged** — see Audit 053 §16.4 |
| Corruption probabilities and conditions | **unchanged** |
| Corruption mechanism | **unchanged** |
| Measurement metrics | **unchanged** |
| Aggregation semantics | **unchanged** |
| TEST seal | **unchanged** |
| Audit-050 representation numerics | **unchanged** — §7a; only non-tensor provenance was added to the collated batch |
| `unmark/evaluation/stage2_dual_finalist.py` | **one provenance-only change** — §7a; see the note in §11 |

---

## 10. Files changed

| File | Change |
|---|---|
| `docs/spec/decisions.md` | **appended** `D-S2-002`; nothing rewritten |
| `docs/spec/stage2-dual-finalist-protocol.json` | **+2 fields** under `measurement` |
| `unmark/evaluation/stage2_head_campaign.py` | seed constant, pinned flag, plan guard, validator value+pinned checks, `require_batch_provenance`, measurement-key rule, docstring, `__all__` |
| `unmark/evaluation/stage2_dual_finalist.py` | **provenance only** — the collator emits `batch["corruption_seeds"]`; no numerics, §7a |
| `tests/test_stage2_head_campaign.py` | one test replaced, **31** added |
| `tests/test_stage2_dual_finalist_infra.py` | **3** added — collator provenance, forward-pass isolation, and the torch-free allowlist check |
| `docs/audits/054-stage2-measurement-corruption-seed-freeze.md` | **new** — this file |

Seven paths. No other file was modified.

---

## 11. Deferred by explicit scope decision

The following stale project-state readings were **deliberately not repaired**:

* `docs/spec/stage2-dual-finalist-protocol.json` — `state.stage2_started`,
  `state.downstream_results_seen`;
* `unmark/evaluation/stage2_dual_finalist.py` — `STAGE2_TRAINING_STARTED`,
  `STAGE2_HEAD_TRAINING_IMPLEMENTED`.

Audit 053 §16.1 and §16.2 already record them. They are orthogonal to pinning the
measurement corruption seed, each is pinned by a test, and correcting them is a
state-schema cleanup with its own author decision. Folding them into this patch
would have widened a narrow scientific freeze into a mixed change that is harder
to review and harder to revert.

**On `stage2_dual_finalist.py`.** The first version of this audit recorded that
module as untouched, and that was true then. Independent review then found the
provenance gap of §7a, and the scope-out was lifted for **exactly one change**:
the collator carries `corruption_seeds` through as non-tensor metadata. Nothing
else in that module changed: the forward-tensor computation and numerical
semantics are unchanged, and the only addition is non-tensor provenance metadata
ignored by the forward allowlist. The stale state-schema constants in it —
`STAGE2_TRAINING_STARTED`, `STAGE2_HEAD_TRAINING_IMPLEMENTED` — were **not**
touched and remain deferred.

---

## 12. Exact next allowed step

**Not** real measurement. In order:

1. **Authoritative runtime acceptance of the amended measurement-seed guards,
   using synthetic data only.** The 62 torch-gated Stage-2 skips must run where
   torch exists.
2. **Verify at runtime:** the exact pinned seed; the 12-request plan; refusal of
   a wrong integer; `FULL` binding `None`; every degraded request binding
   `19225`; both arms identical; TEST still sealed.
3. **Only after that acceptance passes** may a real `measurement-dev` extraction
   notebook be prepared — prepared, and separately reviewed, before it is run.

Explicitly not permitted before step 2 passes:

* reading any UIT-VSFC `measurement-dev` row;
* creating any degraded representation cache;
* producing any measurement score;
* opening official TEST, which remains SEALED;
* choosing, promoting, dropping or ranking A versus B;
* re-running any completed head.

**No UIT-VSFC `measurement-dev` row was read during this task.**

---

## 13. Inconsistencies found

Two fail-closed gaps found by independent review of the first version of this
change set, both repaired here, plus one resolved supersession.

### 13.1 The protocol validator checked the value but not the pinned state

The amended artifact carries both `measurement.corruption_seed = 19225` and
`measurement.corruption_seed_pinned = true`, but the first validator checked only
the first. An artifact could therefore have pinned a value while still declaring
the seed unresolved, and passed. **Repaired** — §7 now validates presence, type,
value and the pinned flag, each with its own refusal.

### 13.2 A prepared representation could be mislabelled at the cache boundary

The larger gap. `collate_stage2_unmark_batch` dropped the corruption seed, so
`extract_and_cache_stage2_representations` could check arm, checkpoint and row
count but **not** the realisation that actually produced the batch. A caller
entering below the plan could have prepared degraded examples under `19224`,
declared `19225` in the key, and stored a tensor whose identity was false. The
plan-level refusal did not close this, because the plan can be bypassed.
**Repaired** — §7a: the collator carries the seed as non-tensor provenance, a
measurement degraded key must bind `19225`, and extraction verifies batch against
key before the torch import and before any write.

Both were real. Neither had produced a wrong artifact — no degraded cache exists
at all — but the guarantee D-S2-002 states was not yet enforced end to end, and
an audit that claimed it would have been overclaiming. The claim in §7 was
narrowed accordingly.

### 13.3 A resolved supersession, not a defect

**Audit 051 §10 and Audit 053 §13 both record the seed as unresolved**, and
Audit 051 §17 item 2 lists it as an open limitation. All three were accurate when
written and are now discharged by D-S2-002. **Those audits are not edited** — the
project keeps its audit history append-only, and rewriting a superseded audit
would destroy the record of the value having once been open, which is precisely
the evidence that this freeze was prospective. The supersession is recorded here.

Nothing else disagreed. The seed table in D-S1B-005, `unmark/stage1/protocol.py`,
`docs/spec/stage1-final-freeze.json` and the amended Stage-2 protocol artifact all
carry `19225` consistently, and `require_frozen_protocol_spec()` now enforces the
last of those against the implementation on every call.

---

## 14. Final state

```
git status --short
 M docs/spec/decisions.md
 M docs/spec/stage2-dual-finalist-protocol.json
 M tests/test_stage2_dual_finalist_infra.py
 M tests/test_stage2_head_campaign.py
 M unmark/evaluation/stage2_dual_finalist.py
 M unmark/evaluation/stage2_head_campaign.py
?? docs/audits/054-stage2-measurement-corruption-seed-freeze.md
```

`git diff --check` produced no output. `HEAD` is
`ceb903832cb57fea03e9492e9af6660ecddad9e0`, unchanged.

Uncommitted and awaiting author review. Nothing staged, committed, pushed or
history-mutated.

---

```
STAGE2_PROTOCOL_FROZEN=YES
AUDIT049_PROTOCOL_AUTHORITY=UNCHANGED
AUTHORITATIVE_RUNNER_TORCH_ACCEPTANCE=PASS

STAGE2_TRAINING_COMPLETE=YES
STAGE2_HEAD_RUNS_COMPLETED=10_OF_10
CLEAN_PROTOCOL_DEV_RESULTS_SEEN=YES

STAGE2_MEASUREMENT_CORRUPTION_SEED=19225
STAGE2_MEASUREMENT_CORRUPTION_SEED_PINNED=True
STAGE2_MEASUREMENT_SEED_FROZEN_PROSPECTIVELY=YES
STAGE2_MEASUREMENT_SEED_INHERITED_FROM=D-S1B-005

MEASUREMENT_DEV_READ=NO
DEGRADED_MEASUREMENT_CACHE_CREATED=NO
DEGRADED_MEASUREMENT_RESULTS_SEEN=NO

STAGE2_AB_SELECTION_IMPLEMENTED=NO
A_B_SELECTION=NO
A_B_WINNER=NONE
A_B_RANKING=NONE

DOWNSTREAM_TEST=SEALED
OFFICIAL_TEST_READ=NO

READY_FOR_STAGE2_TRAINING=COMPLETED
READY_FOR_MEASUREMENT_GUARD_RUNTIME_ACCEPTANCE=YES
READY_FOR_STAGE2_MEASUREMENT=NO
```

`READY_FOR_STAGE2_MEASUREMENT` stays `NO`. The seed was the *scientific* blocker
and it is now resolved; what remains is the *engineering* one — runtime
acceptance of the guards on synthetic data (§12). A frozen realisation and a
verified guard are different things, and only the first exists.
