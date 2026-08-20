# Audit 013 — B3B-1C alignment validation and channel projection

| | |
|---|---|
| **Audit id** | 013 |
| **Date (UTC)** | 2026-08-20 |
| **Scope** | Close manual-alignment validation; decide tone ownership; implement deterministic channel projection |
| **Repository state** | `HEAD = 826f641`; this work uncommitted |
| **Predecessors** | [011](011-b3b1a-input-path-and-alignment-preflight.md), [012](012-b3b1b-whitespace-chunk-alignment-repair.md) |
| **Phase** | Phase 0 / B3B-1C |
| **Evidence run** | `20260820T035339Z` — `B3B1_ALIGNMENT_PROBE_COMPLETE` |

---

## A. Verdict

**PASS**

Manual whitespace-chunk alignment is now a **validated component**, not a hypothesis. The
corrected probe aligned 2,489 / 2,489 sentences against the real pinned `PhobertTokenizer`
with 13/13 token-sequence consistency, 13/13 token-**id** consistency, 119/119 exact chunk
reconstructions, 42/42 curated sentences and **0** `UNDECIDED` eligibility labels. It is
described in the repository as a validated mechanism from here on.

Tone ownership is decided by **counting distinct Vietnamese candidate contributors**, and
deterministic channel projection is implemented as pure data in
[`unmark/alignment/channels.py`](../../unmark/alignment/channels.py). **1642 tests pass**
offline (baseline before this work: 1576).

---

## B. What the evidence settled

### B1. The mixed-piece question, closed on data rather than on caution

Of 191 piece overlays: 138 single-candidate, 51 no-candidate, **2 mixed**, 0 unresolved.
Both mixed pieces mix exactly one candidate with punctuation, and **zero** pieces span two
distinct candidates:

```text
piece "en-"   <- candidate "tuyen" + "-"     (inside a URL)
piece ".com"  <- "." + candidate "com"       (inside a URL)
```

The audit-012 rule ("any mixture ⇒ no tone") would have thrown away two tone labels that
were never ambiguous. Punctuation does not compete to own a tone. The replacement rule:

| Distinct candidates overlapping the piece | Tone |
|---|---|
| 0 | `NA` — `NOT_APPLICABLE` |
| exactly 1 | **that candidate's observed tone**, punctuation present or not |
| ≥ 2 | `NA` — `MULTI_CANDIDATE_AMBIGUOUS`, every contributor recorded |

Verified against the real observed case before the rule was written down:

```text
piece 0 'tuy'    SINGLE_CANDIDATE           region=0 candidates=(0,)
piece 1 'en-'    SINGLE_CANDIDATE           region=0 candidates=(0,)
piece 2 'sinh'   SINGLE_CANDIDATE           region=2 candidates=(2,)
```

### B2. What is never done

A multi-candidate piece is **never** resolved — not by majority overlap length, not by
first or last contributor, and never by averaging categorical tone ids. This holds **even
when the candidates carry the same tone**: sharing a value is not the same as having one
source, and a "they agree, so take it" rule has no principled extension to the disagreeing
case. Tested directly (`test_agreeing_candidates_are_still_ambiguous`).

No URL, e-mail, `tuyen`, `com` or literal piece is special-cased anywhere. Counting is over
**distinct region indices**, so a piece receiving several contribution records from one
region stays single-candidate.

---

## C. What was built

`unmark/alignment/channels.py` — pure data. No torch, no neural layers, no trainable
parameters, no model weights, no pooling arithmetic. Enforced by AST tests.

| Structure | Role |
|---|---|
| `TokenToneLabel` | Projects `ObservedTone`, plus a **token-level** `NA` no syllable can have |
| `CharacterContribution` | One base character: its index, letter label, eligibility, region |
| `ToneProjection` | Ownership, label, source region, all candidate contributors |
| `LetterProjection` | Every contributor, the applicable subset, the token summary, the pooling rule |
| `TokenOrthographyProjection` | Both channels for one authoritative subword |

Decisions worth naming:

* **`UNMARKED` ≠ `NA`.** `UNMARKED` says "a Vietnamese syllable with no readable mark";
  `NA` says "no Vietnamese syllable here". Collapsing them would tell the adapter a comma
  is a ngang-looking syllable. **Lexical `NGANG` is absent from `TokenToneLabel`** — the
  deploy pathway never reintroduces it.
* **`NONE` ≠ `NA` in the letter channel.** `NONE` is an applicable contributor — "a letter
  that could carry a Vietnamese letter diacritic and does not" — and is real information.
  Only `NA` is excluded. Zero applicable contributors ⇒ token letter channel `NA`.
* **Pooling is recorded, not implemented.** `LETTER_POOLING_RULE` persists the decided rule
  (arithmetic mean *in embedding space*, `NONE` included, `NA` excluded) so the adapter
  implements it rather than re-deciding it. A test asserts nothing here computes a mean.
* **Special tokens** carry `NA` in both channels and **no fabricated source range**
  (`global_start`/`global_end` stay `None`). **Whitespace never becomes a model token.**
* **One source of truth for Unicode.** Character labels are read from the canonical
  `unmark.orthography` decomposition, never re-derived from a BPE token string. An AST test
  forbids `unicodedata`, `NFD`/`NFC` and combining-codepoint literals in the alignment
  package.

### C1. Corruption invariance

`b(x)` is invariant under all six B2 conditions, verified against the **real** B2 engine
locally (no ML):

```text
FULL      base_invariant=True  Tôi học nghiên cứu đường phố.
P25       base_invariant=True  Tôi hoc nghiên cưu đường phố.
P50       base_invariant=True  Tôi hoc nghiên cưu đường phố.
P75       base_invariant=True  Tôi hoc nghiên cưu đường phố.
P100      base_invariant=True  Tôi hoc nghiên cưu đương phô.
STRIP_ALL base_invariant=True  Toi hoc nghien cuu duong pho.
```

One token grid and one set of character ranges therefore serve every condition; only the
channel **values** degrade. Tone coverage is monotonically non-increasing FULL → STRIP_ALL,
and STRIP_ALL leaves `UNMARKED` on real syllables rather than `NA`.

### C2. Tests

`tests/test_channel_projection.py` — 28 categories, 63 tests: label sourcing, multi-character
units, `NONE`/`NA`, tone-label coverage, no lexical `NGANG`, single-candidate propagation,
split syllables sharing a tone, candidate + punctuation, zero candidates, ineligible
syllables, multi-candidate ambiguity, non-resolution by length / position / averaging,
agreeing candidates, unresolved eligibility, special tokens (5 parametrised), fabricated
ranges, whitespace, applicable letters, punctuation exclusion, letter marks, empty letter
channel, pooling recorded, pooling not performed, contribution order and identity, inherited
ranges, OOV ids, six corruption-invariance categories, determinism, serialisation, and two
hygiene categories.

---

## D. Defects found and fixed during this work

**D1 — the B3B-1 probe would have crashed on Colab.** `scripts/b3b1_phobert_alignment_probe.py`
read `o["is_mixed"]`, a key the `ToneOwnership` rename removed. Updated to
`is_multi_candidate` along with its report columns and prose. Found by grep after the
rename, not by a test — the probe cannot execute locally.

**D2 — the new B3B-2 probe reported empty channels.** Caught by a local dry run against a
stub tokenizer: every tone came back `UNMARKED` and every letter `NONE`. Cause: the probe
decomposed `base_text` — the string with the marks **already stripped** — to obtain
syllables. Fixed to use a single canonical decomposition, whose unit and syllable offsets
are already indexed into `base_text`. Confirmed safe for eligibility: `decompose` feeds the
classifier `base_text` regardless ([decompose.py:254](../../unmark/orthography/decompose.py#L254)),
so classification behaviour is unchanged. After the fix:

```text
tones/FULL       tone: SAC HUYEN HOI NGA NANG UNMARKED
tones/STRIP_ALL  tone: UNMARKED ×6
letters/FULL   letter: STROKE HORN | HORN NONE NONE | NONE NONE CIRCUMFLEX | ...
```

`scripts/b3b2_channel_projection_probe.py` is **not run locally** and is not claimed to have
run against the real tokenizer. The dry run used a stub that slices chunks arbitrarily; its
multi-candidate count is a stub artifact, not evidence.

---

## E. Scope discipline

* The **backbone checkpoint is not locked**. This probe pinned a revision for
  reproducibility, which is a provenance requirement, not a modelling decision.
  [D-B3B0-002](../spec/decisions.md#d-b3b0-002) stays open.
* No model weights were loaded, nothing was trained, no adapter was implemented.
* The local `.venv` remains ML-free: `torch`, `transformers`, `sentencepiece`, `datasets`
  and `py_vncorenlp` are absent. No network access, no Java, no VnCoreNLP.
* Superseded evidence was retained, not rewritten: the 6/13 run and the A05C diagnostic are
  recorded as such in `docs/experiments/b3b1-manual-alignment-result.md`.

## F. Files

| File | Change |
|---|---|
| `unmark/alignment/channels.py` | **new** — channel projection |
| `unmark/alignment/manual.py` | `ToneOwnership` candidate-count rule; `candidate_region_indices` |
| `unmark/alignment/__init__.py` | exports |
| `tests/test_channel_projection.py` | **new** — 28 categories, 63 tests |
| `tests/test_manual_alignment.py` | updated for the new rule; added multi-candidate coverage |
| `scripts/b3b2_channel_projection_probe.py` | **new** — Colab only |
| `scripts/b3b1_phobert_alignment_probe.py` | D1 fix |
| `docs/experiments/b3b1-manual-alignment-result.md` | **new** — run evidence |
| `docs/spec/decisions.md` | D-B3B1C-001; D-B3B1B-002 mixed-piece paragraph marked closed |

**Proposal updated: NO.** §4.4 is satisfied as written; the offset mechanism is an
implementation detail already recorded in D-B3B1A-003. **PDF stale: YES** (unchanged from
B3B-1B).

Every change is left **unstaged**. No `add`, `commit`, `push`, `tag`, `stash`, `reset`,
`checkout` or `restore` was run.

```text
AUDIT FILE WRITTEN: docs/audits/013-b3b1c-alignment-validation-and-channel-projection.md
OTHER AUDIT MODIFICATIONS MADE: NO
COMMIT CREATED: NO
```
