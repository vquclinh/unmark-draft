# Audit 003 — B2 deterministic corruption engine

| | |
|---|---|
| **Audit id** | 003 |
| **Date (UTC)** | 2026-08-19 |
| **Scope** | The complete B2 corruption engine |
| **Repository state** | `HEAD = 04c32df` ("implement B1A Vietnamese orthography core"); B2 work uncommitted |
| **Predecessors** | [001](001-g-minus1-environment-policy.md), [002](002-b1a-orthography-core.md) (PASS) |
| **Phase** | Phase 0 / B2 |

---

## A. Verdict

**PASS**

B2 implements proposal §5.3 and §6.3 as written, with no deviations. Corruption is
`C(canon(x), condition, seed, sample_id)`, keyed by a BLAKE2b digest over
`(schema_version, seed, sample_id, sha256(canonical text), unit_index)` — no `random`
module, no global RNG, no Python `hash()`, and no sequential dependence between units.
Output is byte-identical across fresh processes and across `PYTHONHASHSEED` values;
dataset reordering changes nothing; placement variants corrupt identically. `P100` and
`STRIP_ALL` are correctly distinguished. Base invariance holds for every condition
including `STRIP_ALL`. The genuine-`ngang`-versus-stripped-tone ambiguity is preserved in
paired metadata while both read `UNMARKED` in the corrupted string. 1291 tests pass
offline, including all B1A and G−1 tests. Six clarifications and one deferral are recorded
in `docs/spec/decisions.md`; the two that touch scientific semantics were also written into
the editable proposal.

---

## B. Files changed

**New**

| File | Purpose |
|---|---|
| `unmark/corruption/conditions.py` | `CorruptionCondition`, `CorruptionScope`, the six conditions, `VARIANT` registered as unimplemented |
| `unmark/corruption/deterministic.py` | `CORRUPTION_SCHEMA_VERSION`, `text_identity`, `unit_score`, `is_selected` |
| `unmark/corruption/models.py` | `UnitDecision`, `CorruptionResult`, H4 oracle views |
| `unmark/corruption/corrupt.py` | `corrupt`, `corrupt_batch` |
| `unmark/corruption/__init__.py` | Package exports |
| `configs/corruption/default.yaml` | Schema version, condition definitions, seed policy, canonicalisation mode |
| `scripts/b2_corruption_self_check.py` | Offline self-check writing four artifacts |
| `tests/test_corruption.py` | 505 tests |
| `docs/spec/decisions.md` | Decision / deviation log (D-B2-001 … D-B2-006) |
| `results/b2/.gitkeep` | Run-artifact directory |

**Modified**

| File | Change |
|---|---|
| `unmark-proposal.md` | §5.3 now states that corruption operates on `canon(x)` and is keyed by an explicit `sample_id`, pointing at the decision log |
| `README.md` | "Deterministic corruption (B2)" section; layout updated |

`unmark-proposal.pdf` was **not** touched and no PDF was regenerated. It is stale with
respect to §4.2 (B1A) and §5.3 (B2).

---

## C. Corruption contract

**Signature**

```python
corrupt(text, condition, seed, sample_id, *, source_is_clean=True) -> CorruptionResult
```

**Conditions** (proposal §6.3, verbatim)

| Condition | Scope | p | Removes |
|---|---|---|---|
| `FULL` | `NONE` | 0.0 | nothing |
| `P25` / `P50` / `P75` | `TONE` | 0.25 / 0.50 / 0.75 | tone marks |
| `P100` | `TONE` | 1.0 | tone marks (letter diacritics survive) |
| `STRIP_ALL` | `TONE_AND_LETTER` | 1.0 | tone marks **and** `ă â ê ô ơ ư đ` |
| `VARIANT` | — | — | recognised, not implemented (D-B2-005) |

**Unit** — `SyllableSpan` (maximal alphabetic run), per §6.3's "% of syllables". No
characters, no words, no tokens.

**Algorithm**

```text
canonical = canon(text)                              # nucleus-based placement
identity  = sha256(canonical).hexdigest()
payload   = schema_version | seed | sample_id | identity | unit_index   (0x1F-separated)
score     = int(blake2b(payload, digest_size=8)) / 2**64                # in [0, 1)
selected  = score < probability
```

`p = 0.0` selects nothing (no score `< 0`); `p = 1.0` selects everything (max score is
`(2**64 − 1)/2**64 < 1`). `CORRUPTION_SCHEMA_VERSION = "b2-v1"` is inside the payload, so a
future algorithm change yields different decisions and old artifacts cannot be pooled
silently.

**Invariants**

```text
strip_to_base(canon(x)) == strip_to_base(corrupt(...).corrupted_text)
canon(corrupted_text) == corrupted_text
len(clean.syllables) == len(corrupted.syllables)          # asserted, not assumed
original_text is never mutated
realized_probability       = selected_units / eligible_units   (None when 0 eligible)
realized_modification_rate = modified_units / eligible_units   (None when 0 eligible)
```

---

## D. Proposal consistency

| Proposal statement | Implemented behavior | Status |
|---|---|---|
| §5.3 `C(x, p, s) → x̃`, "the same triple must always produce the same corrupted string" | Keyed digest over schema/seed/sample_id/text-identity/unit; verified across processes and `PYTHONHASHSEED` | **CLARIFIED** (D-B2-001) |
| §5.3 corruption is a deterministic function of *example* | `sample_id` makes example identity explicit, not row order | **CLARIFIED** (D-B2-001) |
| §5.3 input `x` (raw or canonical unstated) | Corruption operates on `canon(x)`; original preserved verbatim | **CLARIFIED** (D-B2-004) |
| §6.3 `P25/P50/P75` = "Tone marks removed from 25/50/75% of syllables" | Tone scope, syllable unit, independent per-unit selection | **CLARIFIED** (D-B2-002) |
| §6.3 `P100` = "All tone marks removed" | Tone scope, p = 1.0; `ă â ê ô ơ ư đ` survive | **MATCH** |
| §6.3 `STRIP_ALL` = "Tone **and** letter diacritics removed" | Tone + letter scope, p = 1.0 | **MATCH** |
| §6.3 `P100` and `STRIP_ALL` are distinct rows | Distinct scopes; test asserts different output | **MATCH** |
| §6.3 `VARIANT` = placement variants and NFC/NFD forms | Registered, raises with reason; needs `TonePlacement.TRADITIONAL` | **BLOCKED / DEFERRED** (D-B2-005) |
| §6.3 `FULL` = "Fully diacritized (upper bound)" | Returns `canon(x)` unchanged, 0 selections | **MATCH** |
| §4.3 "a *ngang* syllable is *invariant*; a toned syllable transitions to `UNMARKED`" | Selected `ngang` recorded `selected=True, modified=False`; both rates reported | **CLARIFIED** (D-B2-006) |
| §4.3 tone channel "encodes only what is observable"; `UNMARKED` ≠ `NGANG` | Separate `Tone` / `ObservedTone` enums carried through corruption; never relabelled | **MATCH** |
| §4.5 "`b(x) = b(x̃)` for every corruption rate" | Base invariance tested for all 6 conditions × 15 texts | **MATCH** |
| §4.3 non-Vietnamese spans are `N/A`, membership by syllable-inventory rule | No filter applied; every span eligible, GAP-2 deferred | **CLARIFIED** (D-B2-003) |
| §6.7 ORACLE needs genuine *ngang* vs `MISSING` | `oracle_tone_is_genuine_ngang` / `oracle_tone_is_missing` derivable from paired metadata | **MATCH** |
| §4.6 stage-1 channel dropout `p ~ U(0,1)` per example | Not implemented — that is stage-1 training sampling, not the B2 operator | Out of B2 scope |

---

## E. Decisions / deviations logged

All in `docs/spec/decisions.md`. **No `DEVIATED` entries.**

| ID | Status | Affected files |
|---|---|---|
| D-B2-001 | CLARIFIED | `deterministic.py`, `corrupt.py`, `unmark-proposal.md` §5.3 |
| D-B2-002 | CLARIFIED | `deterministic.py` |
| D-B2-003 | CLARIFIED (GAP-2 dependency) | `corrupt.py` |
| D-B2-004 | CLARIFIED | `corrupt.py`, `unmark-proposal.md` §5.3 |
| D-B2-005 | DEFERRED | `conditions.py` |
| D-B2-006 | CLARIFIED | `models.py` |

Explicitly recorded as **"No B2 deviation from proposal"**: `P100` vs `STRIP_ALL`, the
sampling unit, the determinism requirement, and the tone semantics.

---

## F. Spec gaps

```
GAP-2  DEFERRED (unchanged from audit 002) — Vietnamese-candidate eligibility.
B2 is NOT blocked by it: a non-Vietnamese word carries no Vietnamese tone mark, so
selecting it is a no-op and the corrupted string matches what a resolved rule would
produce. Two consequences are documented rather than hidden (D-B2-003): English words sit
in the realized_probability denominator, and a loanword written with a Vietnamese-codepoint
mark is stripped (café -> cafe), which is covered by a test.

VARIANT condition (D-B2-005) is deferred, not a new gap: the proposal specifies it fully;
implementing it requires adopting TonePlacement.TRADITIONAL, which B1A deliberately did
not implement.
```

---

## G. Blocking issues

`None`

---

## H. Non-blocking issues

```
ID: N1  unmark/corruption/corrupt.py
    realized_probability counts selections over ALL syllable spans, so a mixed
    Vietnamese/English sentence has English words in its denominator. Correct given GAP-2
    is deferred, and realized_modification_rate is reported alongside, but the number
    should be read as a selection rate rather than a damage rate.

ID: N2  unmark/corruption/corrupt.py
    STRIP_ALL strips an acute from a loanword (café -> cafe) because the acute is a
    Vietnamese tone codepoint. Arguably realistic for the typing behaviour the condition
    models, and identical to the documented base_signature behaviour, but it is a
    language-blind operation that a resolved GAP-2 would change.

ID: N3  scripts/b2_corruption_self_check.py
    The self-check asserts contracts and writes artifacts but always exits 0 on success;
    it has no "expected failure" fixture proving the failure path renders. The unit tests
    cover the failure-reporting code paths instead.

ID: N4  unmark-proposal.pdf
    Stale with respect to §4.2 (B1A tone placement) and §5.3 (B2 corruption key). The
    editable markdown is current. Regenerating the PDF was out of scope and would need a
    toolchain this repository does not have.
```

---

## I. Test evidence

| Command | Result |
|---|---|
| `.venv/bin/python -m pytest -q -p no:cacheprovider` | **1291 passed in 2.13s** |
| `pytest tests/test_corruption.py` | 505 passed |
| `pytest tests/test_orthography_decompose.py tests/test_orthography_signature.py` | B1A green |
| `pytest tests/test_restore_smoke_utils.py` | G−1 green |
| `scripts/b2_corruption_self_check.py` | 17 cases × 6 conditions = 102 records, **0 failures**, `B2_CORRUPTION_SELF_CHECK_PASS` |
| `pip list` in `.venv` | 7 packages, unchanged — nothing installed |
| `ls ~/.cache/huggingface/hub` | `CACHEDIR.TAG`, `datasets--AIGuruTinix--ViFinQA` — nothing downloaded |

All 40 required test areas are covered. AST-level tests assert that the corruption package
contains no `hash()` call, no `random`/`numpy` import, and no torch/transformers/tokenizer/
sentencepiece/datasets import.

---

## J. Determinism evidence

Selection bitmap for the first 40 of 80 syllables of a fixed generated sequence,
`P50`:

```text
same key (seed=1, id=a) : 0010110101100000111110010111001011111001   (identical, text equal)
different seed (seed=2)  : 1000010001001000101010000110110000000111
different id   (id=b)    : 1101001010001111011001000010111100000001
```

Fresh subprocess, three `PYTHONHASHSEED` values:

```text
PYTHONHASHSEED=0       -> 'Tôi đang nghiên cứu xử lý'
PYTHONHASHSEED=1       -> 'Tôi đang nghiên cứu xử lý'
PYTHONHASHSEED=random  -> 'Tôi đang nghiên cứu xử lý'
```

(identical scores and selections too, not only the text).

Row reordering: corrupting the 15 mixed texts forward, reversed and as a 6-element subset
produces byte-identical per-`sample_id` results in all three orders.

---

## K. Corruption evidence

Clean: `Tôi đang nghiên cứu xử lý ngôn ngữ tự nhiên.` (seed 42, `sample_id="s1"`)

| Condition | Output |
|---|---|
| `FULL` | `Tôi đang nghiên cứu xử lý ngôn ngữ tự nhiên.` |
| `P25` | `Tôi đang nghiên cứu xử lý ngôn ngữ tự nhiên.` |
| `P50` | `Tôi đang nghiên cứu xử lý ngôn ngữ tự nhiên.` |
| `P75` | `Tôi đang nghiên cứu xư lý ngôn ngư tự nhiên.` |
| `P100` | `Tôi đang nghiên cưu xư ly ngôn ngư tư nhiên.` |
| `STRIP_ALL` | `Toi dang nghien cuu xu ly ngon ngu tu nhien.` |

`P100` keeps `ô` and `ê`; `STRIP_ALL` removes them. `P25`/`P50` are unchanged here because
the selected syllables happened to be `ngang` — visible in the metadata as
`selected > 0, modified = 0`, which is exactly why both rates are reported.

---

## L. Ambiguity / H4 evidence

`corrupt("ma má mà mả mã mạ", "P100", seed=1, sample_id="amb")` → `ma ma ma ma ma ma`

| syllable | clean lexical | clean observed | corrupted observed | mark removed | ORACLE view |
|---|---|---|---|---|---|
| ma | `NGANG` | `UNMARKED` | `UNMARKED` | False | genuine `NGANG` |
| ma | `SAC` | `SAC` | `UNMARKED` | True | `MISSING` |
| ma | `HUYEN` | `HUYEN` | `UNMARKED` | True | `MISSING` |
| ma | `HOI` | `HOI` | `UNMARKED` | True | `MISSING` |
| ma | `NGA` | `NGA` | `UNMARKED` | True | `MISSING` |
| ma | `NANG` | `NANG` | `UNMARKED` | True | `MISSING` |

All six corrupted syllables are indistinguishable in the string and all read `UNMARKED`.
None is relabelled `NGANG` — the two enums are disjoint types. The paired metadata is the
only place the difference survives, which is what the ORACLE policy needs and what
`OBSERVABLE` must not have access to.

---

## M. Base-invariance evidence

`strip_to_base(clean) == strip_to_base(corrupted)` for 7 representative texts × all 6
conditions: **True** in every combination, and separately for all 15 mixed texts in the
unit tests.

```text
STRIP_ALL: 'đường ăn cân' -> 'duong an can'   base = 'duong an can'
```

Corruption never substituted a letter, deleted a consonant, inserted a word, reordered
characters, or altered punctuation, whitespace, digits, case, URLs, e-mail structure or
non-Vietnamese combining marks (`ü`, `ç` survive every condition). The syllable count is
asserted equal before and after; a mismatch raises rather than being absorbed.

---

## N. Future UNMARK compatibility

- **Stage-1 training pairs** — `corrupt` yields the clean/corrupted string pair plus both
  decompositions, which is what the alignment and clean-preservation losses consume. The
  §4.6 continuous `p ~ U(0,1)` sampler is a thin caller on top of this operator and is
  deliberately not implemented in B2.
- **B3 alignment** — the base stream is identical between clean and corrupted, so the token
  grid is invariant by construction; character offsets are on every unit and span.
- **H4** — `oracle_tone_is_missing` and `oracle_tone_is_genuine_ngang` give the ORACLE
  labels; `corrupted_observed_tone` gives `OBSERVABLE`; `FORCED-NGANG` is a reinterpretation
  of the same field. No embedding indices, no 7-slot table, no policy implemented.
- **Reproducibility** — every result carries `schema_version`, condition, seed, `sample_id`
  and text identity, so a run's noise is reconstructable from its artifacts alone.

---

## O. Git state

* **Branch:** `main`
* **HEAD:** `04c32df` "implement B1A Vietnamese orthography core" — committed by the
  researcher during this session (after B1A, before this audit was written). The reflog
  shows no commit from this session's work.
* **Staged:** none (`git diff --cached --stat` empty)
* **Unstaged:** `README.md`, `unmark-proposal.md`
* **Untracked:** `configs/corruption/`, `docs/spec/decisions.md`,
  `docs/audits/003-b2-deterministic-corruption.md`, `results/b2/`,
  `scripts/b2_corruption_self_check.py`, `tests/test_corruption.py`, `unmark/corruption/`

No `add`, `commit`, `push`, `tag`, `stash`, `reset`, `checkout` or `restore` was run.

```text
AUDIT FILE WRITTEN: docs/audits/003-b2-deterministic-corruption.md
OTHER AUDIT MODIFICATIONS MADE: NO
COMMIT CREATED: NO
```
