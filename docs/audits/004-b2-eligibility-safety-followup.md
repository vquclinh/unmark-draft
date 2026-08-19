# Audit 004 — B2 eligibility safety follow-up

| | |
|---|---|
| **Audit id** | 004 |
| **Date (UTC)** | 2026-08-19 |
| **Scope** | The full B2 state after the eligibility-safety refactor |
| **Repository state** | `HEAD = 04c32df`; B2 work uncommitted |
| **Supersedes in part** | [Audit 003](003-b2-deterministic-corruption.md) — its D-B2-003 classification only. Audit 003 is a point-in-time record and is **not** rewritten. |
| **Phase** | Phase 0 / B2 |

---

## A. Verdict

**PASS**

The deterministic engine is unchanged and safe to commit: identical scores, identical
outputs, all previous tests green. What changed is the framing of eligibility. Candidate
spans and eligible Vietnamese syllables are now separate concepts in the types, the field
names, the artifacts and the config; `Eligibility.UNDECIDED` is carried through and never
upgraded; `CorruptionResult.eligible_units` raises instead of returning the candidate
count; and `corrupt()` defaults to `purpose=SCIENTIFIC`, which refuses while GAP-2 is
open. Generating stage-1 training or evaluation data under the provisional denominator is
now impossible by accident — it requires explicitly asking for `SELF_CHECK` mode, and every
artifact that mode writes is stamped `provisional_eligibility: true`. The decision log
reclassifies D-B2-003 from `CLARIFIED` to `TEMPORARY IMPLEMENTATION FALLBACK` with B3 named
as resolution owner. 1304 tests pass offline.

---

## B. Files changed

**New**

| File | Purpose |
|---|---|
| `unmark/corruption/eligibility.py` | `EligibilityPolicy`, `CorruptionPurpose`, `EligibilityUnresolved`, `require_resolved_eligibility`, `is_resolved`, `ACTIVE_ELIGIBILITY_POLICY` |
| `docs/audits/004-b2-eligibility-safety-followup.md` | This audit |

**Modified**

| File | Change |
|---|---|
| `unmark/corruption/models.py` | `eligible_units`/`selected_units`/`modified_units` → `candidate_units`/`selected_candidates`/`modified_candidates`; `realized_probability`/`realized_modification_rate` → `candidate_selection_rate`/`candidate_modification_rate`; added `eligibility_policy`, `provisional_eligibility`, per-decision `eligibility`; `eligible_units` is now a property that **raises** |
| `unmark/corruption/corrupt.py` | `purpose` parameter defaulting to `SCIENTIFIC`; guard call; candidate-span comment marking the single filter point; provisional metadata |
| `unmark/corruption/__init__.py` | Exports the eligibility API; module docstring states the engine/policy split |
| `scripts/b2_corruption_self_check.py` | Runs in explicit `SELF_CHECK` mode; asserts the guard fires; stamps artifacts; report carries a provisional-eligibility warning |
| `tests/test_corruption.py` | Migrated to the new API; **+13 safety tests** |
| `configs/corruption/default.yaml` | `eligibility:` block recording the provisional policy, its consequences and its owner |
| `README.md` | "Eligibility is not resolved yet — and the code refuses to pretend" |
| `docs/spec/decisions.md` | D-B2-003 rewritten; status vocabulary extended; four-way categorisation table added |

`unmark-proposal.md` was **not** changed by this task, deliberately — see §G.

---

## C. Issue reclassification

**What audit 003 said.** D-B2-003 was recorded as `CLARIFIED`, with the English-denominator
and `café → cafe` effects listed as "documented consequences" and "arguably realistic". Its
§F stated "B2 is NOT blocked by it".

**What researcher review found.** That classification was too weak. The two effects do not
merely add footnotes — they change what a corruption rate *means*:

1. `P25`/`P50`/`P75` realized rates are fractions of alphabetic runs, not of Vietnamese
   syllables. On `toi dung Python va PyTorch` the denominator is 5, not 3.
2. `STRIP_ALL` modifies foreign spans whose spelling uses a codepoint that is also a
   Vietnamese tone mark.

A number that silently means something different from what the protocol specifies is a
scientific defect, not a clarification — and nothing in the previous implementation stopped
that number from flowing into stage-1 training.

**What is now recorded.** D-B2-003 is `TEMPORARY IMPLEMENTATION FALLBACK`, with resolution
owner **B3 / pre-training**. The decision log's status vocabulary gained that category, and
a table now sorts every entry into `INTENDED FINAL SPEC` / `TEMPORARY IMPLEMENTATION
FALLBACK` / `RESOLVED DECISION` / `KNOWN DEFERRED GAP`, so a later proposal-vs-repository
audit can tell them apart mechanically.

**What did not change.** The deterministic engine. Audit 003's findings about determinism,
`P100` vs `STRIP_ALL`, base invariance and H4 metadata all still hold and were re-verified.

---

## D. Final B2 engine contract

Unchanged from audit 003 §C, and re-verified byte-for-byte:

```text
canonical = canon(text)
identity  = sha256(canonical).hexdigest()
payload   = schema_version | seed | sample_id | identity | unit_index
score     = int(blake2b(payload, 8)) / 2**64
selected  = score < probability
```

`CORRUPTION_SCHEMA_VERSION = "b2-v1"` is unchanged: the engine's decisions are identical
before and after this refactor, so artifacts remain comparable across it. Conditions,
scopes, base invariance, syllable-count assertion, canonical output and H4 metadata are all
as audited in 003.

Regression evidence:

```text
P75  seed=42 id=s1 -> "Tôi đang nghiên cứu xư lý ngôn ngư tự nhiên."   (10 candidates, 6 selected)
P100 seed=42 id=s1 -> "Tôi đang nghiên cưu xư ly ngôn ngư tư nhiên."
STRIP_ALL          -> "Toi dang nghien cuu xu ly ngon ngu tu nhien."
unit_score(seed=42, id="s1", index=0) == 0.21396497977394088
```

---

## E. Temporary eligibility contract

| Concept | Meaning | Computable today |
|---|---|---|
| **candidate span** | maximal alphabetic run; structural, language-blind | yes |
| **eligible Vietnamese syllable** | a candidate whose stripped form is in the Vietnamese syllable inventory (§4.3) | **no** — GAP-2 |

While `ACTIVE_ELIGIBILITY_POLICY is EligibilityPolicy.UNRESOLVED`:

* counts are `candidate_units`, `selected_candidates`, `modified_candidates`;
* rates are `candidate_selection_rate`, `candidate_modification_rate` — `None`, never
  `0.0`, when there are no candidates;
* `eligible_units` raises `EligibilityUnresolved`, naming the candidate count, GAP-2 and B3;
* every `UnitDecision.eligibility` is `Eligibility.UNDECIDED`;
* `to_dict()` omits `eligible_units` and `realized_probability` entirely and includes
  `eligibility_policy` and `provisional_eligibility`;
* `result.metadata["eligibility_filter"]` spells out `PROVISIONAL`, GAP-2 and the owner.

`EligibilityPolicy.VIETNAMESE_SYLLABLE_INVENTORY` exists as a **name only** — no inventory,
no word list, no classifier ships with it.

---

## F. Pre-training guard evidence

```text
active policy: UNRESOLVED

corrupt() default        raises EligibilityUnresolved ✓
purpose=SCIENTIFIC       raises EligibilityUnresolved ✓
corrupt_batch default    raises EligibilityUnresolved ✓
purpose=SELF_CHECK       works, provisional=True      ✓
result.eligible_units    raises                        ✓
```

The default is the safe path: the unsafe one must be asked for by name. The error message
names GAP-2, the B3 resolution owner, the missing syllable inventory, both concrete
consequences, and how to opt into `SELF_CHECK` — asserted by test, not just by inspection.

The self-check itself verifies the guard fires, and stamps `config.json` and `summary.json`
with `eligibility_policy: UNRESOLVED`, `provisional_eligibility: true`, `purpose:
SELF_CHECK`; `report.md` opens with a provisional-eligibility warning naming GAP-2.

---

## G. Proposal / decision-log consistency

| Surface | State |
|---|---|
| `unmark-proposal.md` §4.3 | Already states the intended semantics correctly ("matches the Vietnamese syllable inventory after stripping"). **Unchanged** — no correction needed. |
| `unmark-proposal.md` §5.3 | The B2 edit describes canonicalisation and the `sample_id` key only. It makes **no** eligibility claim. **Unchanged.** |
| `docs/spec/decisions.md` D-B2-003 | Rewritten: `TEMPORARY IMPLEMENTATION FALLBACK`, owner B3, with original wording, temporary state, reason, affected metrics, affected files and closure requirements. |
| `docs/spec/decisions.md` header | Status vocabulary extended; four-way categorisation table added. |
| `configs/corruption/default.yaml` | `eligibility:` block records the provisional state. |
| `README.md` | Section states the fallback and the guard. |

The fallback was deliberately **not** written into the proposal. Doing so would make a
temporary state look normative, which is the failure this task exists to prevent.

Audit 003 is left intact as a point-in-time record; this audit references and corrects it.

---

## H. Blocking issues

`None` — for committing B2. GAP-2 is blocking for *training*, which is now enforced in code
rather than in prose.

---

## I. Non-blocking issues

```
ID: N1  unmark/corruption/models.py
    `eligible_units` is a property that raises. A caller doing getattr-style introspection
    over the dataclass will trip it. Intentional -- a silent provisional number is the
    failure mode being prevented -- but worth knowing before writing generic serialisers.

ID: N2  unmark/corruption/corrupt.py
    corrupt() now raises by default, so casual exploration needs purpose=SELF_CHECK. The
    error explains this. Accepted cost of making the unsafe path opt-in.

ID: N3  Audit 003 sections F and H
    Still describe D-B2-003 in the old, weaker terms. Not edited, by design: audits are
    point-in-time records. This audit is the correction, and audit 003's header now has a
    successor reference only through this file, not by modification.

ID: N4  unmark-proposal.pdf
    Still stale w.r.t. §4.2 (B1A) and §5.3 (B2). Unchanged by this task.
```

---

## J. Test evidence

| Command | Result |
|---|---|
| `.venv/bin/python -m pytest -q -p no:cacheprovider` | **1304 passed in 2.40s** |
| `pytest tests/test_corruption.py` | 517 passed (504 migrated + 13 new safety tests) |
| B1A suites (`test_orthography_*.py`) | green |
| G−1 suites (`test_restore_smoke_utils.py`, `test_orthography_signature.py`) | green |
| `scripts/b2_corruption_self_check.py` | 17 cases × 6 conditions = 102 records, **0 failures**, `UNRESOLVED` / provisional stamped |
| `pip list` in `.venv` | 7 packages, unchanged — nothing installed |
| `ls ~/.cache/huggingface/hub` | unchanged — nothing downloaded |

The ten required proofs are covered by name:

1. engine output/scores unchanged in provisional mode — `test_deterministic_output_is_unchanged_by_the_eligibility_refactor`, `test_scores_are_unchanged_by_the_eligibility_refactor`
2. `UNDECIDED` not presented as resolved — `test_undecided_eligibility_is_not_presented_as_resolved`
3. guard fails while unresolved — `test_scientific_purpose_is_the_default_and_fails_today`
4. message names GAP-2 / B3 — `test_guard_error_names_gap2_and_the_b3_resolution_owner`
5. English spans reported as provisional candidates — `test_the_documented_provisional_consequences_are_still_reproducible`
6. metric names unmisreadable — `test_metric_names_cannot_be_misread_as_a_vietnamese_syllable_rate`, `test_eligible_units_refuses_to_return_a_provisional_number`
7. no dictionary/classifier/tokenizer added — `test_no_dictionary_classifier_or_tokenizer_was_added`
8. previous B2 tests green — full suite
9. B1A and G−1 green — full suite
10. no network/dependency/download — `pip list` and HF cache unchanged

---

## K. Future B3 resolution requirement

To close GAP-2, B3 must:

1. **Enumerate the Vietnamese syllable inventory** (proposal §1.1: ~21 onsets, 155 finals,
   ~7000 syllables in practical use). Source and version must be recorded as a decision.
2. **Apply it to the stripped form**, so clean and corrupted input receive identical labels
   and the base grid stays invariant (§4.3). Deciding from the presence of diacritics would
   break that; a B1A test already forbids it.
3. **Filter `spans`** at the single marked place in `corrupt.py`. Nothing else in the
   operator changes.
4. **Set `ACTIVE_ELIGIBILITY_POLICY`** to `VIETNAMESE_SYLLABLE_INVENTORY`, which
   simultaneously unblocks `purpose=SCIENTIFIC` and makes `eligible_units` return a real
   number.
5. **Regenerate every corruption artifact.** Artifacts produced under `UNRESOLVED` are not
   comparable with resolved ones: the denominator changes, so the rates mean different
   things. `provisional_eligibility` in each artifact is what makes that detectable.
6. **Record the ambiguity error mode.** §4.3 accepts that ambiguous spans resolve towards
   Vietnamese and calls it "a known and deliberate error mode"; that acceptance should be
   restated once the rule is real.

Until all six are done, `corrupt()` refuses `SCIENTIFIC` purpose, which is the property this
audit certifies.

---

## L. Git state

* **Branch:** `main`
* **HEAD:** `04c32df` "implement B1A Vietnamese orthography core" — the researcher's commit;
  the reflog shows no commit from this session
* **Staged:** none (`git diff --cached --stat` empty)
* **Unstaged:** `README.md`, `unmark-proposal.md`
* **Untracked:** `configs/corruption/`, `docs/spec/decisions.md`,
  `docs/audits/003-b2-deterministic-corruption.md`,
  `docs/audits/004-b2-eligibility-safety-followup.md`, `results/b2/`,
  `scripts/b2_corruption_self_check.py`, `tests/test_corruption.py`, `unmark/corruption/`

No `add`, `commit`, `push`, `tag`, `stash`, `reset`, `checkout` or `restore` was run.

```text
AUDIT FILE WRITTEN: docs/audits/004-b2-eligibility-safety-followup.md
OTHER AUDIT MODIFICATIONS MADE: NO
COMMIT CREATED: NO
```
