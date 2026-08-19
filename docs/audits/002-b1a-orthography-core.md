# Audit 002 — B1A Vietnamese orthography core (final state)

| | |
|---|---|
| **Audit id** | 002 |
| **Date (UTC)** | 2026-08-19 |
| **Scope** | The complete final B1A state, not only the GAP-1 closure diff |
| **Repository state** | `HEAD = 7f57ab2` ("build G-1 RESTORE smoke-test harness"); B1A work uncommitted |
| **Predecessor** | [Audit 001](001-g-minus1-environment-policy.md); the B1A audit returned `IMPLEMENTATION PARTIAL — SPEC DECISION REQUIRED` |
| **Phase** | Phase 0 / B1A |

---

## A. Verdict

**PASS**

GAP-1 is closed. The researcher fixed UNMARK's canonical tone placement as nucleus-based,
and `canon` now implements it as a deterministic rule over syllable structure — onset
digraphs (`qu-`, `gi-`), the medial glide, letter diacritics marking the nucleus, then coda
presence — not as a lookup table. `TonePlacement.MODERN` is the default for `canon()`;
`PRESERVE` survives as an explicit diagnostic mode; `TRADITIONAL` raises. Placement variants
collapse (`hòa` → `hoà`, `thúy` → `thuý`, `khỏe` → `khoẻ`, `qùa` → `quà`, `gía` → `giá`) while
`mùa`, `được`, `tiếng`, `moóc` and every other already-nucleus form are untouched.
Canonicalisation moves nothing but a tone mark, and only within the vowel cluster that
already holds it. 787 tests pass offline, including all 166 pre-existing G−1 tests. The
decision, its rationale and its explicit non-claim are recorded in
`docs/spec/orthography.md` and mirrored into the editable proposal source. GAP-2
(Vietnamese-candidate eligibility) remains deliberately deferred, which is the intended
state, not a defect.

---

## B. Files changed

**New**

| File | Purpose |
|---|---|
| `unmark/orthography/placement.py` | The nucleus-based rule; `apply_modern_placement`, `find_nucleus_index` |
| `unmark/orthography/units.py` | `split_units` / `join_units`, shared by `placement` and `decompose` so the two cannot disagree about letter boundaries |
| `unmark/orthography/marks.py` | Mark inventories; `Tone`, `ObservedTone`, `LetterDiacritic`, `Eligibility`, `Anomaly` |
| `unmark/orthography/models.py` | `CharacterUnit`, `SyllableSpan`, `DecomposedText` |
| `unmark/orthography/canonical.py` | `canon`, `TonePlacement`, `canonical_differences` |
| `unmark/orthography/decompose.py` | `decompose`, `recompose`, `strip_to_base` |
| `scripts/g0_orthography_check.py` | G0 round-trip checker |
| `tests/test_orthography_decompose.py` | 621 tests |
| `docs/spec/orthography.md` | D-001 placement decision, D-002 deferred eligibility, D-003 canon scope |
| `results/g0/.gitkeep` | run-artifact directory |

**Modified**

| File | Change |
|---|---|
| `unmark/orthography/__init__.py` | Exports the core alongside the G−1 signatures |
| `unmark/orthography/signature.py` | Mark tables imported from `marks.py` instead of redeclared; public API and behaviour unchanged |
| `tests/test_restore_smoke_utils.py` | `LIGHTWEIGHT_MODULES` extended with the six core modules |
| `README.md` | Orthography-core section, canonical-placement section, deferred-GAP-2 section |
| `unmark-proposal.md` | §4.2 now names the fixed nucleus-based convention and points at `docs/spec/orthography.md` |

`unmark-proposal.pdf` was **not** touched and no PDF was regenerated.

---

## C. Closed spec gap

**GAP-1 — canonical tone placement.**

*Before.* Proposal §4.2 required `canon` to apply "Unicode NFC and a fixed tone-placement
rule" and named the `hoà`/`hòa` ambiguity, but never said which convention was canonical.
No repository code encoded one. B1A implemented `PRESERVE` (NFC only) and raised on the
alternatives rather than invent a spelling standard.

*Decision.* The researcher fixed the convention as **nucleus-based**
(`TonePlacement.MODERN`).

*How it was resolved.*

1. `unmark/orthography/placement.py` implements the rule over syllable structure, applied
   to the contiguous vowel cluster that already carries the tone mark:
   1. **Onset digraphs** — in `qu-` the `u` belongs to the onset, and in `gi-` followed by
      another vowel the `i` does; neither can take the tone.
   2. **Glide** — a cluster-initial `o` before `a`/`e`/`ă`, or `u` before `y`, is the medial
      glide /w/ and is not the nucleus.
   3. **Letter diacritics mark the nucleus** — breve, circumflex or horn wins; with two
      (only `ươ`), the second.
   4. **Otherwise** — among the remaining plain vowels, coda present → last, absent → first.
2. `canonical.py` sets `DEFAULT_TONE_PLACEMENT = TonePlacement.MODERN` and routes `canon`
   through the rule. `PRESERVE` skips the placement step. `TRADITIONAL` raises
   `TonePlacementNotImplemented` (aliased as the former `TonePlacementUndecided`).
3. `units.py` was extracted so `placement` and `decompose` share one definition of a
   letter unit. The dependency graph stays acyclic: `units → placement → canonical → decompose`.
4. The decision is recorded in `docs/spec/orthography.md` (D-001) with purpose, rule,
   examples, scope and an explicit non-claim, and mirrored into `unmark-proposal.md`.

*Rejected approaches.* "First vowel", "last vowel" and "middle character" are each wrong
for a large class of syllables and none distinguishes a glide from a nucleus. A hard-coded
table of the specified examples would not generalise; a test canonicalises syllables that
appear nowhere in `placement.py` and asserts they are absent from its source.

---

## D. Orthography contract

**Canonicalisation**

```text
input -> Unicode normalisation -> fixed nucleus-based tone placement -> NFC canonical text
```

```text
canon(x)                            uses TonePlacement.MODERN by default
canon(canon(x)) == canon(x)
canon(NFC(x)) == canon(NFD(x))
canon(old_variant) == canon(nucleus_variant)
canon(x, PRESERVE)                  NFC only; explicit diagnostic mode
canon(x, TRADITIONAL)               raises TonePlacementNotImplemented
```

`canon` never alters letters, case, punctuation, whitespace, digits, URLs or e-mail
addresses. Letter-forming diacritics never leave their base letter. A tone mark never
crosses a consonant. A syllable with more than one tone mark is left exactly as found.

**Decomposition**

```text
recompose(decompose(x)) == canon(x)          exact, by construction
decompose(x).base_text                       no Vietnamese tone mark, letter mark or đ/Đ
decompose(x).base_text == base_signature(x, collapse_whitespace=False)
observed_tone_channel identical whether or not source_is_clean is asserted
```

**Channels** — base: character; tone: **syllable**; letter diacritic: **character**.

**States**

```text
Tone (lexical)        NGANG SAC HUYEN HOI NGA NANG
ObservedTone (deploy) UNMARKED SAC HUYEN HOI NGA NANG
LetterDiacritic       NONE BREVE CIRCUMFLEX HORN STROKE NA
Eligibility           NOT_APPLICABLE UNDECIDED
Anomaly               MULTIPLE_TONE_MARKS MULTIPLE_LETTER_DIACRITICS
                      UNSUPPORTED_COMBINING_MARK TONE_MARK_ON_NON_LETTER
TonePlacement         MODERN (default) PRESERVE (diagnostic) TRADITIONAL (raises)
```

`UNMARKED` is not `NGANG`: `lexical_tone` is `None` for an unmarked syllable unless the
caller asserts `source_is_clean=True`. A visible mark settles it either way.

---

## E. Remaining spec gaps

```
GAP-2  DEFERRED BY DESIGN — Vietnamese-candidate eligibility
Where:   proposal 4.3 — "an alphabetic span is treated as a Vietnamese candidate if it
         matches the Vietnamese syllable inventory after stripping".
Missing: the syllable inventory. Not enumerated in the proposal, not in this repository.
Status:  every alphabetic span reports Eligibility.UNDECIDED. No word list invented, no
         dictionary downloaded, no Vietnamese/English classifier built. Recorded as D-002
         in docs/spec/orthography.md and in the README.
Owner:   the B3 / input-policy stage.
Constraint carried forward: whatever rule is adopted must remain a pure function of the
         STRIPPED form, so clean and corrupted input get identical labels and the base grid
         stays invariant (proposal 4.3). A test asserts the code never decides from the
         presence of diacritics.
```

No new gap was exposed while implementing the placement rule. Two cases were considered
and resolved from structure rather than guessed: `ươ` (two diacritic vowels — the tone goes
on the second, verified against `người`, `được`, `rượu`, `mười`, `tưởng`) and plain vowel
pairs with a coda (`moóc`), where the coda clause applies.

---

## F. Blocking issues

`None`

---

## G. Non-blocking issues

```
ID: N1  unmark/orthography/placement.py
    Clause 4 ("plain vowel pair + coda -> last") is reachable in practice only for `oo`
    loanwords such as `moóc`; every other Vietnamese cluster is resolved earlier by the
    glide or diacritic clauses. It is exercised by a test but has a narrow natural
    footprint. Behaviour on an unattested plain-pair-plus-coda cluster is defined and
    deterministic, not a guess.

ID: N2  unmark/orthography/decompose.py
    The module and the `decompose` function share a name (proposal 8.1 fixes both), so
    `import unmark.orthography.decompose as d` binds the FUNCTION — the datetime.datetime
    shadowing. Documented in the module docstring and pinned by a test.

ID: N3  scripts/g0_orthography_check.py
    Reads one unit per line; sentence segmentation for a real corpus is not implemented.
    Not needed until a corpus exists.

ID: N4  unmark/orthography/models.py
    No embedding-slot indices are assigned to the 7-slot tone table. Deliberate: slot
    layout is B2/B3 policy and assigning it now would pre-commit the H4 design.

ID: N5  unmark/orthography/canonical.py
    `TonePlacementUndecided` is retained as an alias of the renamed
    `TonePlacementNotImplemented` so existing callers keep working. Two names for one
    exception; the old one can be dropped once nothing references it.
```

---

## H. Test evidence

| Command | Result |
|---|---|
| `.venv/bin/python -m pytest -q -p no:cacheprovider` | **787 passed in 1.49s** |
| `pytest tests/test_orthography_signature.py tests/test_restore_smoke_utils.py` | **166 passed** — no G−1 regression |
| `pytest tests/test_orthography_decompose.py` | 621 tests |
| `pytest -k lightweight` | 7 passed — all six core modules import with zero heavy modules in `sys.modules` |
| `scripts/g0_orthography_check.py --self-check` | 30 checked, 30 passed, 8 canonical-only differences of which 6 are tone-placement collapses; status `ORTHOGRAPHY_CORE_READY_FOR_G0_CORPUS_CHECK` |
| `pip list` in `.venv` | 7 packages, unchanged — nothing installed |
| `ls ~/.cache/huggingface/hub` | `CACHEDIR.TAG`, `datasets--AIGuruTinix--ViFinQA` — nothing downloaded |

The 20 required regression areas are all covered: placement-pair collapsing, modern-form
idempotence, traditional→modern conversion, NFC/NFD equivalence, uppercase, tone-only
vowels, letter-diacritic + tone, `oa`, `oe`, `uy`, `qu-`, `ia`/`iê`, `ua`/`uô`, `ưa`/`ươ`,
`yê`, surrounding punctuation, mixed Vietnamese/English, URL/e-mail preservation,
malformed multiple-tone input, and decomposition-channel equivalence across variants.

Two structural-invariance tests run over every curated text and every placement pair:
`base_signature` is unchanged by canonicalisation, and the letter-diacritic channel is
unchanged by canonicalisation.

---

## I. Canonicalisation evidence

| Input | `canon(input)` | Rule clause |
|---|---|---|
| `hòa` | `hoà` | glide `o` before `a` |
| `hoà` | `hoà` | already canonical |
| `hóa` | `hoá` | glide `o` before `a` |
| `khỏe` | `khoẻ` | glide `o` before `e` |
| `thúy` | `thuý` | glide `u` before `y` |
| `thủy` | `thuỷ` | glide `u` before `y` |
| `qùa` | `quà` | `qu-` onset |
| `gía` | `giá` | `gi-` onset |
| `gì` | `gì` | `gi` with no following vowel |
| `ủy` | `uỷ` | glide with no onset at all |
| `HÒA` | `HOÀ` | case preserved |
| `mùa` | `mùa` | `u` is not a glide before `a` |
| `tiếng` | `tiếng` | letter diacritic marks the nucleus |
| `được` | `được` | `ươ` — tone on the second |
| `rượu` | `rượu` | `ươ` with a following vowel |
| `moóc` | `moóc` | plain pair with a coda → last |
| `"hòa",` | `"hoà",` | punctuation untouched |
| `hòa với machine learning` | `hoà với machine learning` | English untouched |
| `https://example.edu.vn/a?id=42` | unchanged | URL untouched |
| `a` + acute + grave | unchanged | two tone marks: malformed, not repaired |

Decompositions of each variant pair expose identical base text, letter-diacritic channel,
observable-tone channel, lexical-tone channel and canonical text.

---

## J. Round-trip evidence

| Case | Input | Base | Syllable tone | `recompose == canon` |
|---|---|---|---|---|
| plain vowel | `ba` | `ba` | UNMARKED | ✓ |
| tone only | `bà` | `ba` | HUYEN | ✓ |
| letter diacritic only | `ăn` | `an` | UNMARKED | ✓ |
| tone + letter diacritic | `ắt` | `at` | SAC | ✓ |
| đ / Đ | `đi` / `Đi` | `di` / `Di` | UNMARKED | ✓ |
| NFC input | `Tiếng Việt` | `Tieng Viet` | SAC | ✓ |
| NFD input | `Tiếng Việt` (NFD) | `Tieng Viet` | SAC | ✓ |
| placement variant | `hòa` | `hoa` | HUYEN | ✓ (canon → `hoà`) |
| multi-diacritic syllable | `được` | `duoc` | NANG | ✓ |
| punctuation / mixed | `Toi dung Python, 2026! a@b.com 😄` | unchanged | — | ✓ |

Also verified over 298 generated letters (6 base vowels × 4 letter-mark options × 6
tone-mark options × 2 cases), in isolation and inside a syllable context, plus all 30
`ắằẳẵặ…ứừửữự` characters and their uppercase forms, empty and whitespace-only strings, a
combining mark with no base character, and a tone mark on a digit.

---

## K. Future UNMARK compatibility

Unchanged from the B1A audit, and strengthened by the closed gap:

- **Corruption (B2)** — `SyllableSpan.observed_tone` is the unit corruption acts on;
  `tone_unit_index` locates the mark to delete, and it is now *canonical* rather than
  input-dependent, so the same word corrupts identically however it was spelled.
  `source_is_clean=True` on a gold corpus yields `lexical_tone`, from which the `ORACLE`
  policy's genuine-*ngang*-vs-`MISSING` distinction is derivable without H4 being
  implemented here.
- **Base-token alignment (B3)** — every `CharacterUnit` and `SyllableSpan` carries both
  `canonical_start/end` and `base_start/end`. `Eligibility.UNDECIDED` is where the deferred
  N/A policy plugs in.
- **Tone embeddings** — `observed_tone_channel` is per syllable, aligned to `syllables`.
  Slot assignment intentionally left to B2.
- **Letter embeddings** — `letter_channel` is per character, aligned to `units`, ready for
  mean-pooling in embedding space (proposal §4.4 step 4). `NONE`/`NA` stay distinct so
  padding and "no diacritic" never collapse.

The B1A caveat is now gone: because placement variants collapse, the `VARIANT` evaluation
condition (proposal §6.3) has a well-defined reference form, and two corpora using
different conventions produce the same base grid.

---

## L. G0 readiness

Status reported: **`ORTHOGRAPHY_CORE_READY_FOR_G0_CORPUS_CHECK`**.

The checker evaluates the locked convention (`tone_placement: MODERN`,
`variant_collapsing_implemented: true`) and separates the two kinds of canonical-only
difference — Unicode normalisation and tone-placement collapsing — so neither hides inside
the other. Its self-check units now include placement variants.

It does **not** and must not claim G0 PASS. G0 requires ≥100K sentences of real Vietnamese;
no corpus ships with this repository, nothing here downloads one, and no natural corpus was
run. The string `G0 PASS` appears nowhere in the script's status values, and a test asserts
that.

What remains before G0 can be claimed: obtain a corpus meeting the §7 composition
requirements (ordinary Vietnamese, NFC and NFD forms, mixed script, digits, punctuation,
URLs, e-mail, emoji, đ/Đ, all of ă â ê ô ơ ư, all six tones, multi-modified-vowel syllables,
tone-placement variants, uppercase and title case, empty and whitespace-only strings), run
the checker on it, and read the enumerated differences.

---

## M. Git state

* **Branch:** `main`
* **HEAD:** `7f57ab2` "build G-1 RESTORE smoke-test harness" — committed by the researcher
  before this session; the reflog shows no commit from the implementation or the audit
* **Staged:** none (`git diff --cached --stat` empty)
* **Unstaged:** `README.md`, `unmark-proposal.md`, `tests/test_restore_smoke_utils.py`,
  `unmark/orthography/__init__.py`, `unmark/orthography/signature.py`
* **Untracked:** `docs/spec/`, `results/g0/`, `scripts/g0_orthography_check.py`,
  `tests/test_orthography_decompose.py`, `unmark/orthography/canonical.py`,
  `unmark/orthography/decompose.py`, `unmark/orthography/marks.py`,
  `unmark/orthography/models.py`, `unmark/orthography/placement.py`,
  `unmark/orthography/units.py`, and this audit file

No `add`, `commit`, `push`, `tag`, `stash`, `reset`, `checkout` or `restore` was run.
`__pycache__` byproducts were removed after testing; no `.pytest_cache` was created.

```text
AUDIT FILE WRITTEN: docs/audits/002-b1a-orthography-core.md
OTHER AUDIT MODIFICATIONS MADE: NO
COMMIT CREATED: NO
```
