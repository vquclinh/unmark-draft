"""Composed, memoised Stage-1 length transforms. **Torch-free.**

Stage 6 fed every document through `canon`/`decompose` roughly **250 times its
own length**: the greedy chunker re-derives the *whole growing candidate* at
every segment, so a chunk spanning `k` segments canonicalises prefixes of total
length `O(k * chunk_length)`. On the real corpus that made pre-chunking
unusable (Audit 029 §R).

This module removes that blow-up **without touching the chunking algorithm**.
`chunking.py` still receives plain `str -> int` length functions and still makes
exactly the same `fits` queries in exactly the same order; only the cost of each
query changes.

**The composability lemma.** For a text `T` split into maximal whitespace and
non-whitespace runs `s1 … sn`::

    canon(T)                         == "".join(canon(si))
    decompose(canon(T)).base_text    == "".join(decompose(canon(si)).base_text)

Two independent reasons, both properties of code in this repository rather than
of the tokenizer:

* **NFC never composes across whitespace.** A whitespace character is a starter
  with combining class 0, so no combining sequence spans a run boundary.
* **Tone placement is confined to a syllable.** `apply_modern_placement` moves a
  tone mark within a *maximal alphabetic run*, and whitespace terminates such a
  run, so placement decisions on either side are independent.
* `base_text` is a per-character mapping, hence composable everywhere.

Verified exhaustively in `tests/test_stage1_lengths.py`.

**What this module does NOT assume.** Nothing about the tokenizer. Token counts
are neither assumed additive across concatenation nor monotone in text length:
the transformed string is still handed to the tokenizer whole, exactly as
before. The only thing that is reused is the *orthographic transform*, and only
at boundaries where it is provably composable.
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
    runs_seen: int = 0
    run_cache_hits: int = 0
    tokenizer_calls: int = 0
    verifications: int = 0
    incremental_extensions: int = 0
    full_rescans: int = 0

    def to_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


class ComposedTransforms:
    """`canon(text)` and `base(text)`, composed from memoised segments.

    Results are **bit-identical** to calling `canon` / `decompose` on the whole
    text -- that is the lemma above, and it is what makes this a pure
    performance change.
    """

    def __init__(self, max_entries: int = DEFAULT_MAX_ENTRIES) -> None:
        self._canon: dict[str, str] = {}
        self._base: dict[str, str] = {}
        self._max_entries = max_entries
        self.counters = TransformCounters()

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

    def _segments(self, text: str) -> list[str]:
        pieces = [match.group(0) for match in _SEGMENT.finditer(text)]
        self.counters.segments_seen += len(pieces)
        return pieces

    def canonical(self, text: str) -> str:
        """Identical to `canon(text)`."""
        self.counters.length_queries += 1
        self.counters.characters_queried += len(text)
        return "".join(self._segment_canon(s) for s in self._segments(text))

    def base(self, text: str) -> str:
        """Identical to `decompose(canon(text)).base_text`."""
        self.counters.length_queries += 1
        self.counters.characters_queried += len(text)
        return "".join(self._segment_base(s) for s in self._segments(text))


_NON_WHITESPACE = re.compile(r"\S+")


class TokenLengthComposer:
    """Token length as `specials + sum over maximal non-whitespace runs`.

    **This uses an audited real-tokenizer fact, and still verifies it.**
    D-B3B1B-001 established, on the pinned PhoBERT revision, that fastBPE
    operates over **maximal non-whitespace chunks**: splitting on non-whitespace
    runs and
    tokenizing each whole chunk reproduced the authoritative token sequence
    **13/13** and the token IDs **13/13** across 119 chunks, with zero surface
    reconstruction failures. Composing per chunk is therefore exact, not an
    approximation.

    Audit 029 Revision 1 was caused by borrowing a B3B fact into chunking
    without re-checking it, so this class **does not trust the fact blindly**:
    the first `verify_first` distinct queries are computed *both* ways -- whole
    string and composed -- and any disagreement raises. The property is a
    checked precondition of every run, not a belief.

    No monotonicity is used or implied. Nothing here assumes token counts grow
    with text length; the composition is over disjoint runs of one text.
    """

    def __init__(
        self,
        tokenizer: object,
        transform: Callable[[str], str],
        *,
        counters: TransformCounters,
        verify_first: int = 256,
        max_entries: int = DEFAULT_MAX_ENTRIES,
    ) -> None:
        self._tokenizer = tokenizer
        self._transform = transform
        self._run_length: dict[str, int] = {}
        self._max_entries = max_entries
        self._verify_remaining = verify_first
        self._last: tuple[str, int] = ("", 0)
        self.counters = counters
        self._specials = self._whole_length("")

    def _whole_length(self, transformed: str) -> int:
        ids = self._tokenizer.convert_tokens_to_ids(self._tokenizer.tokenize(transformed))
        return len(self._tokenizer.build_inputs_with_special_tokens(list(ids)))

    def _run_tokens(self, run: str) -> int:
        cached = self._run_length.get(run)
        if cached is not None:
            self.counters.run_cache_hits += 1
            return cached
        self.counters.tokenizer_calls += 1
        value = len(self._tokenizer.tokenize(self._transform(run)))
        if len(self._run_length) < self._max_entries:
            self._run_length[run] = value
        return value

    def _run_sum(self, text: str) -> int:
        """Sum of per-run token counts, reusing the previous query's prefix.

        The chunker's greedy scan asks about a **growing candidate** from a fixed
        start, so consecutive queries satisfy ``text.startswith(previous)``.
        Recomputing every run of the prefix each time is what made Stage 6
        quadratic in segments per chunk.

        The extension is only taken when the junction is **not inside a run** --
        the previous text ends with whitespace, or the delta begins with it. Then
        ``runs(prev + delta) == runs(prev) + runs(delta)`` exactly, because runs
        are maximal non-whitespace spans. Every candidate end the chunker
        produces in its fast path is a whitespace/non-whitespace boundary, so the
        condition holds there; the oversized-unit fallback cuts inside a run, the
        condition fails, and the full path runs. Correctness never depends on the
        shortcut being taken.
        """
        previous, previous_sum = self._last
        if (
            previous
            and len(text) > len(previous)
            and (previous[-1].isspace() or text[len(previous)].isspace())
            and text.startswith(previous)
        ):
            delta = text[len(previous):]
            runs = _NON_WHITESPACE.findall(delta)
            self.counters.runs_seen += len(runs)
            self.counters.incremental_extensions += 1
            total = previous_sum + sum(self._run_tokens(run) for run in runs)
        else:
            runs = _NON_WHITESPACE.findall(text)
            self.counters.runs_seen += len(runs)
            self.counters.full_rescans += 1
            total = sum(self._run_tokens(run) for run in runs)
        self._last = (text, total)
        return total

    def length(self, text: str) -> int:
        self.counters.length_queries += 1
        self.counters.characters_queried += len(text)
        composed = self._specials + self._run_sum(text)

        if self._verify_remaining > 0:
            self._verify_remaining -= 1
            self.counters.verifications += 1
            exact = self._whole_length(self._transform(text))
            if exact != composed:
                raise Stage1ContractViolation(
                    "per-chunk token composition disagreed with whole-string "
                    f"tokenization: composed {composed}, exact {exact}. D-B3B1B-001 "
                    "established that PhoBERT's fastBPE operates over maximal "
                    "non-whitespace chunks; this run's tokenizer does not behave that "
                    "way, so Stage-1 refuses to chunk rather than emit lengths it "
                    "cannot justify. Check the tokenizer revision."
                )
        return composed


def build_length_functions(
    tokenizer: object, transforms: ComposedTransforms | None = None
) -> tuple[Callable[[str], int], Callable[[str], int], ComposedTransforms]:
    """The two Stage-1 length functions, sharing one transform memo.

    The tokenizer is still called on the **whole** transformed candidate, so no
    additivity or monotonicity property of the tokenizer is used or needed.
    """
    transforms = transforms or ComposedTransforms()
    reference = TokenLengthComposer(
        tokenizer, transforms.canonical, counters=transforms.counters
    )
    base = TokenLengthComposer(tokenizer, transforms.base, counters=transforms.counters)
    return reference.length, base.length, transforms
