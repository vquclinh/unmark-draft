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
* Optimised: token counts are summed over the tokenizer's **own** run unit
  (`PHOBERT_RUN`), each run memoised. Every run is counted with the **public
  wrapper** `tokenizer.tokenize(run)`, so added/special-token handling is the
  tokenizer's, not ours.
* **Never optimised: added-token semantics.** A direct `tokenizer.bpe(run)`
  fast path was implemented in Revision 3c and **removed** in the 3c hardening:
  it bypasses the wrapper's added-token split and silently miscounts any run
  containing e.g. `<mask>` (Audit 029 §V). Composition itself is additionally
  gated on `_composition_is_safe()`.

**The composability lemma.** For a text `T` split into maximal whitespace and
non-whitespace runs `s1 … sn`::

    canon(T)                         == "".join(canon(si))
    decompose(canon(T)).base_text    == "".join(decompose(canon(si)).base_text)

because NFC never composes across a whitespace starter (combining class 0),
`apply_modern_placement` moves a tone mark only *within* a maximal alphabetic
run which whitespace terminates, and `base_text` is a per-character mapping.
Verified exhaustively in `tests/test_stage1_lengths.py`, and re-checked at
runtime by `ComposedTransforms.verify_remaining`.

**Revision 3a / 3b history.** Revision 3 composed token counts over ``\\S+``
runs and failed the real probe with ``composed 5, exact 7``; Revision 3a removed
the composition entirely, which was correct but left the tokenizer cost intact
and the real run at ~0.45 docs/s. Revision 3b restores composition over the
**exact unit the pinned tokenizer itself uses** -- ``\\S+\\n?`` -- which is what
bb50823 got wrong. See `RunLengthComposer` and Audit 029 §T.
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
    authoritative_queries: int = 0
    run_cache_hits: int = 0
    run_cache_misses: int = 0
    bpe_run_evaluations: int = 0
    incremental_appends: int = 0
    last_run_recomputations: int = 0
    full_fallbacks: int = 0
    run_cache_evictions: int = 0
    run_cache_entries: int = 0
    run_cache_max_entries: int = 0

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


PHOBERT_RUN = re.compile(r"\S+\n?")
r"""**The pinned tokenizer's own decomposition unit.**

`PhobertTokenizer._tokenize` decomposes with ``re.findall`` over this same
pattern and calls ``bpe`` on each resulting run independently.

The trailing newline is **part of the run**, so ``bpe("gamma\n")`` is not
``bpe("gamma")`` -- BPE's end-of-word marker lands on a different final
character. Composing over plain ``\S+`` instead is exactly the defect that
produced ``composed 5, exact 7`` at bb50823, and it failed 1708 of 1920 real
slice cases. This regex is the contract; ``\S+`` must never be used for it.
"""


class RunLengthComposer:
    """Token length as `specials + sum over the tokenizer's own runs`.

    Exact rather than approximate: the runs are `PHOBERT_RUN`, the same
    decomposition `_tokenize` performs, so the sum of per-run token counts is
    the whole-string token count by construction of the tokenizer itself.

    **Incremental.** The chunker's greedy scan asks about a growing candidate,
    so only the *final* run can change: appended characters may extend it, or
    turn ``"gamma"`` into ``"gamma\n"``. Everything before the last run's start
    offset is therefore stable, and only the tail is recomputed. That is what
    makes the work proportional to **new and changed runs** rather than to the
    sum of all growing prefix lengths.

    **No monotonicity is assumed.** Nothing here says token counts grow with
    text; the composition is over disjoint runs of one string, and the greedy
    scan still evaluates every candidate in order.

    Fail-closed: the first `verify_first` distinct queries are also computed
    through the **authoritative** whole-string chain, and any disagreement
    raises.
    """

    def __init__(
        self,
        tokenizer: object,
        *,
        counters: TransformCounters,
        verify_first: int = DEFAULT_VERIFY_FIRST,
        max_entries: int = DEFAULT_MAX_ENTRIES,
    ) -> None:
        self._tokenizer = tokenizer
        self._runs: dict[str, int] = {}
        self._max_entries = max_entries
        self._verify_remaining = verify_first
        self.counters = counters
        # Derived through the authoritative API, never a hard-coded "+2".
        self._specials = self.authoritative_length("")
        self._last: tuple[str, int, int] = ("", 0, 0)
        self._added_tokens = self._added_token_strings()
        self._compose_runs = self._composition_is_safe()

    def _added_token_strings(self) -> tuple[str, ...] | None:
        """The tokenizer's OWN added/special tokens. Never hard-coded here.

        `PreTrainedTokenizer.tokenize` splits the text on these **before**
        `_tokenize` ever runs, so they are the one thing that can make the
        wrapper disagree with a per-run decomposition.
        """
        collected: set[str] = set()
        for attribute in ("get_added_vocab", "all_special_tokens"):
            value = getattr(self._tokenizer, attribute, None)
            try:
                value = value() if callable(value) else value
            except Exception:  # noqa: BLE001 - a tokenizer that cannot answer
                return None   # unknown added tokens -> composition is not provable
            if isinstance(value, dict):
                collected.update(str(k) for k in value)
            elif isinstance(value, (list, tuple, set)):
                collected.update(str(v) for v in value)
        encoder = getattr(self._tokenizer, "added_tokens_encoder", None)
        if isinstance(encoder, dict):
            collected.update(str(k) for k in encoder)
        return tuple(sorted(t for t in collected if t))

    def _composition_is_safe(self) -> bool:
        """May token counts be summed over `PHOBERT_RUN` runs at all?

        Per-run composition is exact **only** while every added token lies
        wholly inside one run. `tokenize` matches added tokens as literal
        substrings anywhere in the text, so an added token that *contains
        whitespace* would be lifted out across a run boundary by the wrapper
        and split in two by the composition -- a real divergence, demonstrated
        in Audit 029 §V.

        Checked once, from the tokenizer's authoritative collection. When the
        collection cannot be read, or any token contains whitespace, every query
        takes the authoritative whole-string path: slower, always correct.
        **False negatives only cost speed; a false positive would change
        scientific output**, so the unknown case is treated as unsafe.
        """
        if self._added_tokens is None:
            return False
        return not any(
            token != token.strip() or any(c.isspace() for c in token)
            for token in self._added_tokens
        )

    @property
    def composition_enabled(self) -> bool:
        return self._compose_runs

    def authoritative_length(self, transformed: str) -> int:
        """The unchanged pre-optimisation chain. The definition of truth."""
        self.counters.authoritative_queries += 1
        ids = self._tokenizer.convert_tokens_to_ids(self._tokenizer.tokenize(transformed))
        return len(self._tokenizer.build_inputs_with_special_tokens(list(ids)))

    def _run_tokens(self, run: str) -> int:
        cached = self._runs.get(run)
        if cached is not None:
            self.counters.run_cache_hits += 1
            return cached
        self.counters.run_cache_misses += 1
        self.counters.bpe_run_evaluations += 1
        value = len(self._tokenizer.tokenize(run))
        if len(self._runs) < self._max_entries:
            self._runs[run] = value
            self.counters.run_cache_entries = len(self._runs)
            self.counters.run_cache_max_entries = max(
                self.counters.run_cache_max_entries, len(self._runs)
            )
        else:
            # At the ceiling: this run is recomputed every time it recurs. Counted
            # so a cap that is actually costing work is visible rather than
            # guessed at.
            self.counters.run_cache_evictions += 1
        return value

    def _sum_runs(self, transformed: str) -> tuple[int, int]:
        """`(total tokens, offset where the last run starts)` for `transformed`."""
        total = 0
        last_start = len(transformed)
        for match in PHOBERT_RUN.finditer(transformed):
            total += self._run_tokens(match.group(0))
            last_start = match.start()
        return total, last_start

    def length(self, transformed: str) -> int:
        """Composed length of an already-transformed string.

        Falls back to the authoritative whole-string chain whenever per-run
        composition cannot be shown safe for this tokenizer.
        """
        if not self._compose_runs:
            self.counters.full_fallbacks += 1
            return self.authoritative_length(transformed)
        previous, previous_total, last_start = self._last
        if previous and len(transformed) > len(previous) and transformed.startswith(previous):
            # Only the final run of `previous` can be affected by appended text
            # -- it may be extended, or gain the trailing newline. Recompute
            # from its start; everything before it is stable.
            self.counters.incremental_appends += 1
            stable = previous_total
            tail_total = 0
            new_last = len(transformed)
            for match in PHOBERT_RUN.finditer(previous[last_start:]):
                stable -= self._run_tokens(match.group(0))
                self.counters.last_run_recomputations += 1
            for match in PHOBERT_RUN.finditer(transformed[last_start:]):
                tail_total += self._run_tokens(match.group(0))
                new_last = last_start + match.start()
            total, last_start = stable + tail_total, new_last
        else:
            self.counters.full_fallbacks += 1
            total, last_start = self._sum_runs(transformed)
        composed = self._specials + total
        self._last = (transformed, total, last_start)

        if self._verify_remaining > 0:
            self._verify_remaining -= 1
            self.counters.verifications += 1
            exact = self.authoritative_length(transformed)
            if exact != composed:
                raise Stage1ContractViolation(
                    "run composition disagreed with the authoritative whole-string "
                    f"length: composed {composed}, exact {exact}, specials "
                    f"{self._specials}, query length {len(transformed)}, runs "
                    f"{len(PHOBERT_RUN.findall(transformed))}. The pinned tokenizer "
                    "does not decompose as PHOBERT_RUN describes, so Stage-1 refuses "
                    "to chunk rather than emit lengths it cannot justify."
                )
        return composed


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
    reference_runs = RunLengthComposer(tokenizer, counters=transforms.counters)
    base_runs = RunLengthComposer(tokenizer, counters=transforms.counters)

    def reference_length(text: str) -> int:
        return reference_runs.length(transforms.canonical(text))

    def base_length(text: str) -> int:
        return base_runs.length(transforms.base(text))

    return reference_length, base_length, transforms
