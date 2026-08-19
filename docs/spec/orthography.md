# Orthography specification

Decisions governing `unmark/orthography/`. Each entry records what was decided,
why, and what is deliberately *not* claimed.

---

## D-001 — Canonical tone placement

**Status:** decided, 2026-08-19. Closes GAP-1 from
[audit 001](../audits/001-g-minus1-environment-policy.md) and the B1A audit.

### Decision

UNMARK's canonical tone placement is **nucleus-based**:

```python
TonePlacement.MODERN
```

`canon(text)` with no explicit override uses it. The canonicalisation pipeline is:

```text
input
  ↓
Unicode normalisation
  ↓
project fixed tone-placement canonicalisation   (nucleus-based)
  ↓
NFC canonical text
```

### Purpose

Deterministic canonicalisation and reproducible experiments. Vietnamese admits two
accepted positions for a tone mark over a vowel cluster, so the same word reaches the
pipeline spelled two ways. Without a fixed convention, `canon` is not a function of the
word, the G0 round-trip invariant cannot be stated, the `VARIANT` evaluation condition
(proposal §6.3) has no reference form, and the base token grid differs between corpora
that happen to use different conventions.

### Non-claim

> This is **UNMARK's fixed canonical tone-placement convention**, adopted for
> reproducibility.

It is **not** claimed to be the sole official, uniquely correct, or universally preferred
Vietnamese orthographic convention. Both conventions occur in real Vietnamese text and in
published material; text using the other convention is not an error. The project fixes one
so that its own pipeline is deterministic, and canonicalises the other onto it.

### Examples

| Input | Canonical form | Rule clause |
|---|---|---|
| `hòa` | `hoà` | `o` is a glide before `a` |
| `hoà` | `hoà` | already canonical |
| `hóa` | `hoá` | `o` is a glide before `a` |
| `khỏe` | `khoẻ` | `o` is a glide before `e` |
| `thúy` | `thuý` | `u` is a glide before `y` |
| `thủy` | `thuỷ` | `u` is a glide before `y` |
| `qùa` | `quà` | `qu-` is an onset; the `u` cannot take the tone |
| `gía` | `giá` | `gi-` is an onset when another vowel follows |
| `gì` | `gì` | `gi` with no following vowel: `i` is the nucleus |
| `mùa` | `mùa` | `u` is **not** a glide before `a` |
| `tiếng` | `tiếng` | a letter diacritic marks the nucleus |
| `được` | `được` | in `ươ` the tone goes on the second (`ơ`) |
| `moóc` | `moóc` | plain pair with a coda: tone on the last |

### The rule

Applied to the contiguous vowel cluster that already carries the tone mark, in order:

1. **Onset digraphs.** In `qu-` the `u` belongs to the onset; in `gi-` followed by another
   vowel the `i` does. Neither can carry the tone.
2. **Glide.** A cluster-initial `o` before `a`/`e`/`ă`, or `u` before `y`, is the medial
   glide /w/ and is not the nucleus.
3. **Letter diacritics mark the nucleus.** If a remaining vowel carries breve, circumflex
   or horn, the tone goes there. When two do — only `ươ` — it goes on the second.
4. **Otherwise**, among the remaining plain vowels: with a coda consonant the tone goes on
   the last, without one on the first.

Implemented in [`unmark/orthography/placement.py`](../../unmark/orthography/placement.py)
as a rule over syllable structure. It is not a lookup table of examples: a test
canonicalises syllables that appear nowhere in the source and asserts they are absent
from it.

"First vowel", "last vowel" and "middle character" are all rejected — each is wrong for a
large class of syllables, and none distinguishes a glide from a nucleus.

### Scope

Relocation happens strictly **within the vowel cluster that already holds the tone mark**.
Canonicalisation never changes lexical letters, case, punctuation, whitespace, digits, URLs
or e-mail addresses; letter-forming diacritics (`ă â ê ô ơ ư`) never move off their base
letter; a tone mark never crosses a consonant. Only tone-mark position may change.

A syllable carrying more than one tone mark is malformed and is left exactly as found, for
the anomaly reporter to surface.

### `PRESERVE` is diagnostic-only

`TonePlacement.PRESERVE` skips the placement step and returns Unicode NFC alone. It is
**not** the project pathway. Use it to inspect what an input actually contained — for
example to measure how far a corpus is from canonical form. It must be requested
explicitly.

`TonePlacement.TRADITIONAL` is **not implemented** and raises. The project needs one
canonical convention; implementing a second unused one would introduce a spelling standard
no part of the design requires.

### Invariants

```text
canon(x)                        uses MODERN placement by default
canon(canon(x)) == canon(x)
canon(NFC(x)) == canon(NFD(x))
canon(old_variant) == canon(nucleus_variant)
recompose(decompose(x)) == canon(x)
```

Decompositions of the two placement variants expose identical base, letter-diacritic,
observable-tone and lexical-tone channels.

---

## D-002 — Vietnamese-candidate eligibility

**Status:** DEFERRED to the B3 / input-policy stage. This is GAP-2.

Proposal §4.3 decides whether an alphabetic span is a Vietnamese candidate by matching
"the Vietnamese syllable inventory after stripping". That inventory is not enumerated in
the proposal and does not exist in this repository, so B1A does not decide eligibility:
every alphabetic span reports `Eligibility.UNDECIDED`.

Plain ASCII spans such as `ban`, `AI`, `machine`, `learning` are observationally ambiguous
at the orthography layer — `ban` is simultaneously a valid English word and a valid
undiacritized Vietnamese syllable. No word list is invented here and no dictionary is
downloaded.

Whatever rule is eventually adopted **must remain a pure function of the stripped form**,
so that clean and corrupted input receive identical labels and the base grid stays
invariant (proposal §4.3). Deciding from the presence of diacritics would break that;
the code never does, and a test enforces it.

---

## D-003 — What `canon` does not normalise

`canon` is a reconstruction target, not a comparison form. It does not collapse
whitespace, fold case, or rewrite punctuation.

This distinguishes it from `signature.base_signature` and `signature.rewrite_signature`
in the same package, which do normalise whitespace (and, for the latter, case and a
trailing sentence stop) because they serve the G−1 restorer diagnostic. The two layers
share one mark inventory (`marks.py`) but answer different questions.
