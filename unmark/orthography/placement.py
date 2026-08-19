"""UNMARK's canonical tone-placement rule: nucleus-based ("modern").

Vietnamese admits two accepted positions for a tone mark over a vowel cluster
(`hòa` versus `hoà`). Both occur in real text. UNMARK fixes one of them as its
**project canonicalisation convention** so that experiments are reproducible;
this is not a claim that the other convention is wrong, nor that this one is
the sole official Vietnamese orthography. See `docs/spec/orthography.md`.

The convention adopted is **nucleus-based**: the tone mark belongs on the vowel
nucleus, not on a glide.

The rule, applied to the vowel cluster containing the tone mark
--------------------------------------------------------------
1. **Onset digraphs.** In `qu-` the `u` belongs to the onset, and in `gi-`
   followed by another vowel the `i` does; neither can carry the tone.
   `qua` -> `quà`, `gia` -> `giá`, but `gi` alone keeps its vowel: `gì`.
2. **Glide.** A cluster-initial `o` before `a`/`e`/`ă`, or `u` before `y`, is
   the medial glide /w/ and is not the nucleus. `hoa` -> `hoà`,
   `khoe` -> `khoẻ`, `thuy` -> `thuý`.
3. **Letter diacritics mark the nucleus.** If any remaining vowel carries a
   Vietnamese letter diacritic (breve, circumflex, horn), the tone goes there.
   `tieng` -> `tiếng`, `muon` -> `muốn`, `hoac` -> `hoặc`. When two do -- only
   `ươ` -- the tone goes on the second: `nguoi` -> `người`, `duoc` -> `được`.
4. **Otherwise**, among the remaining plain vowels: with a coda consonant the
   tone goes on the last, without one on the first. `mua` -> `mùa`,
   `mai` -> `mài`, `cao` -> `cào`, `mooc` -> `moóc`.

Deliberately *not* implemented as "first vowel", "last vowel" or "middle
character": each of those is wrong for a large class of syllables, and none
distinguishes a glide from a nucleus.

Scope of the transformation
---------------------------
Relocation happens strictly **within the contiguous vowel cluster that already
carries the tone mark**. Letters, case, punctuation, whitespace, digits and
non-Vietnamese text are never touched, letter-forming diacritics never move off
their base letter, and a tone mark never crosses a consonant. A syllable
carrying more than one tone mark is left exactly as it is: that is malformed
input for the anomaly reporter to surface, not something to repair here.
"""

from __future__ import annotations

import unicodedata

from unmark.orthography.marks import LETTER_MARK_TO_STATE, TONE_MARK_TO_OBSERVED
from unmark.orthography.units import join_units, split_units

_TONE_MARK_SET = frozenset(TONE_MARK_TO_OBSERVED)
_LETTER_MARK_SET = frozenset(LETTER_MARK_TO_STATE)

BASE_VOWELS = frozenset("aeiouy")

# Glide contexts, expressed on *base* letters. `ă` bases to `a`, so `oă` is
# covered by the `o` + `a` entry.
_GLIDE_FOLLOWERS = {"o": frozenset("ae"), "u": frozenset("y")}

# Onset digraphs whose second letter is written as a vowel but is not one.
_ONSET_DIGRAPHS = {("q", "u"), ("g", "i")}


def _base_letter(base_cp: str, marks: tuple[str, ...]) -> str:
    """Lowercased base letter of a unit, ignoring every diacritic."""
    if not base_cp:
        return ""
    stripped = unicodedata.normalize("NFD", base_cp)[:1].lower()
    return {"đ": "d"}.get(stripped, stripped)


def _has_letter_diacritic(base_cp: str, marks: tuple[str, ...]) -> bool:
    return any(m in _LETTER_MARK_SET for m in marks)


def _tone_marks(marks: tuple[str, ...]) -> list[str]:
    return [m for m in marks if m in _TONE_MARK_SET]


def find_nucleus_index(
    units: list[tuple[str, tuple[str, ...]]],
    cluster: list[int],
    syllable: list[int],
) -> int:
    """Index (into `units`) of the vowel that should carry the tone.

    `cluster` is the contiguous run of vowel units containing the tone mark;
    `syllable` is the surrounding alphabetic run, used only to detect the onset
    digraph and whether a coda consonant exists.
    """
    letters = {i: _base_letter(*units[i]) for i in syllable}
    candidates = list(cluster)

    # 1. Onset digraph: qu-, and gi- when another vowel follows.
    first = candidates[0]
    position = syllable.index(first)
    if position > 0 and len(candidates) > 1:
        if (letters[syllable[position - 1]], letters[first]) in _ONSET_DIGRAPHS:
            if not _has_letter_diacritic(*units[first]):
                candidates = candidates[1:]

    # 2. Medial glide.
    if len(candidates) > 1:
        head = candidates[0]
        followers = _GLIDE_FOLLOWERS.get(letters[head], frozenset())
        if letters[candidates[1]] in followers and not _has_letter_diacritic(*units[head]):
            candidates = candidates[1:]

    # 3. A letter diacritic marks the nucleus; with two (only `ươ`), the second.
    marked = [i for i in candidates if _has_letter_diacritic(*units[i])]
    if marked:
        return marked[-1]

    # 4. Plain vowels: coda present -> last, otherwise first.
    if len(candidates) == 1:
        return candidates[0]
    has_coda = syllable[-1] != cluster[-1]
    return candidates[-1] if has_coda else candidates[0]


def _alphabetic_runs(units: list[tuple[str, tuple[str, ...]]]) -> list[list[int]]:
    """Maximal runs of letter units: the orthographic syllable candidates."""
    runs: list[list[int]] = []
    current: list[int] = []
    for index, (base_cp, marks) in enumerate(units):
        if base_cp and unicodedata.normalize("NFD", base_cp)[:1].isalpha():
            current.append(index)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


def _vowel_cluster_containing(
    units: list[tuple[str, tuple[str, ...]]], syllable: list[int], target: int
) -> list[int]:
    """The contiguous run of vowel units inside `syllable` that contains `target`."""
    is_vowel = {i: _base_letter(*units[i]) in BASE_VOWELS for i in syllable}
    if not is_vowel.get(target, False):
        return []
    position = syllable.index(target)
    start = position
    while start > 0 and is_vowel[syllable[start - 1]]:
        start -= 1
    end = position
    while end + 1 < len(syllable) and is_vowel[syllable[end + 1]]:
        end += 1
    return syllable[start : end + 1]


def apply_modern_placement(text: str) -> str:
    """Move every tone mark onto its syllable's nucleus. Returns NFC.

    Idempotent: a string already in nucleus placement is returned unchanged.
    """
    units = split_units(unicodedata.normalize("NFD", text))
    mutable: list[tuple[str, list[str]]] = [(base, list(marks)) for base, marks in units]

    for syllable in _alphabetic_runs(units):
        toned = [i for i in syllable if _tone_marks(units[i][1])]
        if len(toned) != 1:
            # No tone, or malformed multi-tone input: leave it exactly as it is.
            continue
        source = toned[0]
        if len(_tone_marks(units[source][1])) != 1:
            continue

        cluster = _vowel_cluster_containing(units, syllable, source)
        if not cluster:
            # Tone mark on a consonant: not a placement question. Leave it.
            continue

        target = find_nucleus_index(units, cluster, syllable)
        if target == source:
            continue

        tone = _tone_marks(units[source][1])[0]
        mutable[source][1].remove(tone)
        mutable[target][1].append(tone)

    rebuilt = join_units([(base, tuple(marks)) for base, marks in mutable])
    # NFC re-applies canonical combining-class ordering, so marks need not have
    # been appended in canonical order above.
    return unicodedata.normalize("NFC", rebuilt)
