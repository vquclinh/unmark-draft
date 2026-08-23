"""Composed, memoised Stage-1 length transforms. **Torch-free.**

Stage 6 fed every document through `canon`/`decompose` roughly **250 times its
own length**: the greedy chunker re-derives the *whole growing candidate* at
every segment, so a chunk spanning `k` segments canonicalises prefixes of total
length `O(k * chunk_length)`. On the real corpus that made pre-chunking unusable
(Audit 029 §R).

This module removes that blow-up **without touching the chunking algorithm**.
`chunking.py` still receives plain `str -> int` length functions and still makes
exactly the same `fits` queries in exactly the same order; only the cost of
producing the *transformed* text changes.

**What is optimised, and what is not.**

* Optimised: `canon(text)` and `decompose(canon(text)).base_text` are built from
  memoised whitespace segments and extended incrementally along a growing
  prefix. This rests on a lemma about *this repository's own code* (below).
* **Not optimised: tokenization.** The transformed candidate is handed to the
  tokenizer **whole**, through the **same API chain** the pre-optimisation
  implementation used. No property of the tokenizer is assumed, used, or needed.

**The composability lemma.** For a text `T` split into maximal whitespace and
non-whitespace runs `s1 … sn`::

    canon(T)                         == "".join(canon(si))
    decompose(canon(T)).base_text    == "".join(decompose(canon(si)).base_text)

because NFC never composes across a whitespace starter (combining class 0),
`apply_modern_placement` moves a tone mark only *within* a maximal alphabetic
run which whitespace terminates, and `base_text` is a per-character mapping.
Verified exhaustively in `tests/test_stage1_lengths.py`, and re-checked at
runtime by `ComposedTransforms.verify_remaining`.

**Removed in Revision 3a.** An earlier version also composed *token counts* per
non-whitespace run, citing D-B3B1B-001. The first real-tokenizer probe
**falsified** that on the pinned PhoBERT: `composed 5, exact 7`. The runtime
verifier caught it and refused to run. The shortcut is gone; only the
tokenizer-independent transform reuse remains. See Audit 029 §S.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from unmark.orthography import canon, decompose
from unmark.stage1.contracts import Stage1ContractViolation

_SEGMENT = re.compile(r"\s+|\S+")

DEFAULT_MAX_ENTRIES = 500_000
"""Memo ceiling. Vietnamese Wikipedia has a large but bounded word-form
vocabulary; this caps memory without changing any result -- an eviction only
costs a recomputation."""

DEFAULT_VERIFY_FIRST = 256
"""How many distinct queries are checked against the direct transform."""


@dataclass
class TransformCounters:
    """Algorithmic accounting. Counts only -- no text."""

    length_queries: int = 0
    segments_seen: int = 0
    canon_cache_hits: int = 0
    canon_calls: int = 0
    base_cache_hits: int = 0
    decompose_calls: int = 0
    characters_queried: int = 0
    characters_canonicalised: int = 0
    characters_decomposed: int = 0
    incremental_extensions: int = 0
    full_rescans: int = 0
    tokenizer_calls: int = 0
    verifications: int = 0

    def to_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


class ComposedTransforms:
    """`canon(text)` and `base(text)`, composed from memoised segments.

    Results are **bit-identical** to calling `canon` / `decompose` on the whole
    text -- that is the lemma above, and it is what makes this a pure
    performance change. `verify_remaining` re-checks that at runtime and fails
    closed, so the lemma is a checked precondition rather than a belief.
    """

    def __init__(
        self,
        max_entries: int = DEFAULT_MAX_ENTRIES,
        verify_first: int = DEFAULT_VERIFY_FIRST,
    ) -> None:
        self._canon: dict[str, str] = {}
        self._base: dict[str, str] = {}
        self._max_entries = max_entries
        self._verify_remaining = verify_first
        self._last_canon: tuple[str, str] = ("", "")
        self._last_base: tuple[str, str] = ("", "")
        self.counters = TransformCounters()

    # -- memoised per-segment transforms ----------------------------------
    def _segment_canon(self, segment: str) -> str:
        cached = self._canon.get(segment)
        if cached is not None:
            self.counters.canon_cache_hits += 1
            return cached
        self.counters.canon_calls += 1
        self.counters.characters_canonicalised += len(segment)
        value = canon(segment)
        if len(self._canon) < self._max_entries:
            self._canon[segment] = value
        return value

    def _segment_base(self, segment: str) -> str:
        cached = self._base.get(segment)
        if cached is not None:
            self.counters.base_cache_hits += 1
            return cached
        self.counters.decompose_calls += 1
        self.counters.characters_decomposed += len(segment)
        value = decompose(self._segment_canon(segment)).base_text
        if len(self._base) < self._max_entries:
            self._base[segment] = value
        return value

    def _compose(self, text: str, per_segment: Callable[[str], str]) -> str:
        pieces = [m.group(0) for m in _SEGMENT.finditer(text)]
        self.counters.segments_seen += len(pieces)
        return "".join(per_segment(p) for p in pieces)

    @staticmethod
    def _extendable(previous: str, text: str) -> bool:
        """Whether `text` extends `previous` at a junction outside any segment.

        The chunker's greedy scan asks about a growing candidate from a fixed
        start, so consecutive queries satisfy ``text.startswith(previous)``. The
        extension is taken only when the junction is whitespace-adjacent, which
        is exactly the condition the composability lemma needs. Every fast-path
        candidate satisfies it; the oversized-unit fallback cuts inside a run,
        the condition fails, and the full path runs. **Correctness never depends
        on the shortcut being taken.**
        """
        return (
            bool(previous)
            and len(text) > len(previous)
            and (previous[-1].isspace() or text[len(previous)].isspace())
            and text.startswith(previous)
        )

    def _transform(self, text, last, per_segment, direct) -> str:
        previous, previous_value = last
        if self._extendable(previous, text):
            self.counters.incremental_extensions += 1
            value = previous_value + self._compose(text[len(previous):], per_segment)
        else:
            self.counters.full_rescans += 1
            value = self._compose(text, per_segment)

        if self._verify_remaining > 0:
            self._verify_remaining -= 1
            self.counters.verifications += 1
            exact = direct(text)
            if value != exact:
                raise Stage1ContractViolation(
                    "composed transform disagreed with the direct transform. The "
                    "composability lemma (canon and base_text compose across "
                    "whitespace-run boundaries) does not hold for this input, so "
                    "Stage-1 refuses to chunk rather than emit lengths it cannot "
                    f"justify. Query length {len(text)}, composed {len(value)}, "
                    f"exact {len(exact)}."
                )
        return value

    # -- public API --------------------------------------------------------
    def canonical(self, text: str) -> str:
        """Identical to `canon(text)`."""
        self.counters.length_queries += 1
        self.counters.characters_queried += len(text)
        value = self._transform(text, self._last_canon, self._segment_canon, canon)
        self._last_canon = (text, value)
        return value

    def base(self, text: str) -> str:
        """Identical to `decompose(canon(text)).base_text`."""
        self.counters.length_queries += 1
        self.counters.characters_queried += len(text)
        value = self._transform(
            text, self._last_base, self._segment_base,
            lambda t: decompose(canon(t)).base_text,
        )
        self._last_base = (text, value)
        return value


def build_length_functions(
    tokenizer: object, transforms: ComposedTransforms | None = None
) -> tuple[Callable[[str], int], Callable[[str], int], ComposedTransforms]:
    """The two Stage-1 length functions, sharing one transform memo.

    **The length definition is the authoritative one, unchanged**::

        length(x) = len(build_inputs_with_special_tokens(
                        convert_tokens_to_ids(tokenize(transform(x)))))

    Same API chain, same whole-string tokenization, same special-token
    accounting as the pre-optimisation implementation. Only `transform` is
    computed more cheaply, so `optimized_length(x) == authoritative_length(x)`
    holds **by construction** rather than by argument.
    """
    transforms = transforms or ComposedTransforms()

    def _tokenized_length(transformed: str) -> int:
        transforms.counters.tokenizer_calls += 1
        ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(transformed))
        return len(tokenizer.build_inputs_with_special_tokens(list(ids)))

    def reference_length(text: str) -> int:
        return _tokenized_length(transforms.canonical(text))

    def base_length(text: str) -> int:
        return _tokenized_length(transforms.base(text))

    return reference_length, base_length, transforms
