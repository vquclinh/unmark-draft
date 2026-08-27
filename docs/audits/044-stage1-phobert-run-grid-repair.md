# Audit 044 — Stage-1 PhoBERT Run-Grid Repair

**Scope:** the production repair for the Audit-043 diagnosis. **One production
file changed.** Audits 001–043 untouched.
**Date:** 2026-08-27

---

## 1. Diagnosis Input

Audit 043 closed the diagnosis. Its final, complete evidence:

| | |
|---|---|
| diagnostic HEAD | `f218d822005ef3b769faa6d9a3a94f37dc973ee8` |
| scientific baseline before repair | `e495f7417fe41ac97aaaf9c2ea6aba0e89afb3e9` |
| real-tokenizer equivalence | `streamed 100 000`, `compared 1204`, `mismatch_count 0` |
| full TRAIN scan | `scanned 2 621 624`, `within 2 621 615`, **`over 9`** |
| overflow histogram | `257 -> 8`, `259 -> 1` |
| max Stage-1 realised | 259 · max Stage-6 authoritative seen 256 |
| all 9 offenders | **newline-bearing** |
| known offender | `Mô_đun:Inflation/data#572`, `7d99f2dba18e45c0`, Stage-6 256 → Stage-1 257 |
| largest offender | `Mô_đun:Easter#3`, Stage-6 256 → Stage-1 259 (delta +3) |
| scan runtime | ~98.2 s at 26 689 rows/s, fastest at 16 workers (5 317.5 rows/s) |

Classification: **CLASS A — Stage-1 implementation bug.**

## 2. Old Semantics vs Repaired Semantics

`PhobertTokenizer._tokenize` is, verbatim:

```python
words = re.findall(r"\S+\n?", text)
for token in words:
    split_tokens.extend(list(self.bpe(token).split(" ")))
```

| | before | after |
|---|---|---|
| Stage-6 `lengths.py::PHOBERT_RUN` | `\S+\n?` | `\S+\n?` (unchanged) |
| Stage-1 `alignment/manual.py::_CHUNK_PATTERN` | **`\S+`** | **`\S+\n?`** |

A run **owns its trailing newline**. `bpe` appends `</w>` to the run's *last*
character, so with the newline present the end-of-word marker lands on the
newline rather than on the final letter, and `bpe("gamma\n")` is a different
piece sequence from `bpe("gamma")` — different pieces, and a different count.

Stage-1 was therefore building its base content-id grid over a decomposition
**the model does not use**, on every newline-bearing chunk.

## 3. Why the Unit Change Is Safe — Proven, Not Assumed

The brief forbade a blind regex swap, and the alignment machinery is
surface-exact: `align_chunk` requires `reconstruct_surface(tokens) ==
chunk.text` and then tiles character ranges by piece-surface length. Including
the newline in a run is only sound if the pieces still reconstruct the run.

Read from the real `tokenization_phobert.py`, `bpe` is:

```python
word = tuple(token)                                   # every char, newline included
word = tuple(list(word[:-1]) + [word[-1] + "</w>"])   # marker on the LAST char
...merges of adjacent pairs...
word = "@@ ".join(word)
word = word[:-4]                                      # strip the trailing </w>
```

Merges only concatenate adjacent elements, so the tuple's concatenation is
always `token + "</w>"`. Joining with `"@@ "`, stripping `</w>`, splitting on
spaces and removing the `@@` suffix therefore recovers **exactly** `token`.

**The pieces are a pure character partition of the run.** Surface reconstruction
is exact for any input, newline-bearing runs included. Verified by transcribing
the real algorithm and checking reconstruction on `gamma\n`, `gamma`, `a`,
`a\n`, `xin\n`, `đọc\n`, `mot\n\n`, `abcdef\n` — all exact.

Consequently `align_chunk`, `_range_problem`, `reconstruct_surface`,
`piece_surface` and the `CONTINUATION_MARKER` guard **need no special case**.
They simply now see runs that end in a newline. The newline character falls in a
non-syllable gap region of `_regions`, so it carries no tone or letter
metadata — which is correct, a newline has no orthography.

## 4. Exact Files Changed

**Production — one file:**

* `unmark/alignment/manual.py` — `_CHUNK_PATTERN` is now `\S+\n?`, with the
  contract, the evidence and the reconstruction argument documented on it and on
  `whitespace_chunks`.

**Tests:**

* `tests/test_stage1_phobert_run_grid_repair.py` (new) — the §I matrix.
* `tests/test_manual_alignment.py` — the run-unit contract test, which asserted
  the old behaviour.
* `tests/test_stage1_length_contract_scanner.py` — its stub tokenizer replaced
  with a **faithful** transcription of the real `bpe`; the pre-repair
  divergence test inverted into an impossibility test.

**Diagnostic:**

* `scripts/stage1_length_contract_scan.py` — docstrings only; it now describes
  itself as the post-repair acceptance gate.

**Unchanged:** `unmark/stage1/protocol.py`, `configs/`, `requirements/`,
`docs/spec/`, `unmark-proposal.md`, and every frozen scientific value.

## 5. Why This Is a Bug Repair, Not a Protocol Change

Nothing in the frozen scientific contract changed, and the repair does not touch
any decision the protocol makes. `MAX_LENGTH` is still 256, truncation still
OFF, overflow still FAIL. No chunk is skipped, filtered, truncated, or dropped;
the prepared corpus is untouched and remains authoritative; no overflow is
caught and continued.

What changed is that Stage-1 now tokenizes the way the pinned tokenizer
tokenizes. The repository already held that as its contract —
`unmark/stage1/lengths.py` says of the plain form, with measurement behind it
(*"it failed 1708 of 1920 real slice cases"*): **"This regex is the contract;
`\S+` must never be used for it."** Stage-1 was violating an existing contract,
not implementing a different one. The repair makes the two halves agree.

The 9 overflowing chunks are not a corpus defect: Stage-6 admitted them at
exactly 256 under the authoritative grid, and under the repaired Stage-1 grid
they measure 256 too.

## 6. Invariants Proven

Proven against a faithful transcription of `PhobertTokenizer.bpe`, over 19
newline/whitespace shapes (§I matrix):

| # | invariant | status |
|---|---|---|
| 1 | `project_text(...).content_ids == convert_tokens_to_ids(tokenize(base_text))` | PASS, all cases |
| 2 | Stage-1 realised base length == Stage-6 authoritative base length | PASS, all cases |
| 3 | `len(content_ids) == len(projections)` — ids and channel metadata 1:1 | PASS, all cases |
| 4 | `reconstruct_surface` round-trips; pieces tile each run contiguously | PASS, all cases |
| 5 | a newline is owned by exactly one piece — not dropped, duplicated or misplaced | PASS |
| 6 | corruption cannot change the base grid (`b(C(x)) == b(x)`, equal base ids) | PASS, all cases |
| 7 | non-newline behaviour unchanged | PASS — old and new units are identical where no newline follows a run |

A **mutation check** recomputes the grid over the old plain-`\S+` unit and
requires it to disagree, so invariant 1 cannot be passing vacuously.

The former overflow class is covered directly: a synthetic newline-bearing text
whose Stage-6 authoritative length is exactly 256 now measures exactly 256 under
the repaired Stage-1 path, and the real `TruncationPolicy(max_length=256,
on_overflow=FAIL)` accepts it — the exact gate that aborted the real run.

Defensive shapes are covered too: empty and whitespace-only text, a final
continuation marker, a surface mismatch, and OOV ids all behave as before, and
the two fail-closed guards still fail closed.

## 7. Tests and Results

```
$ .venv/bin/python -B -m pytest -q -p no:cacheprovider \
      tests/test_stage1_phobert_run_grid_repair.py
132 passed

$ .venv/bin/python -B -m pytest -q -p no:cacheprovider \
      tests/test_stage1_phobert_run_grid_repair.py \
      tests/test_stage1_length_contract_scanner.py tests/test_manual_alignment.py
261 passed

$ .venv/bin/python -B -m pytest -q -p no:cacheprovider
3999 passed, 106 skipped, 0 failed
```

The full suite covers the Stage-1 freeze, optimizer, resume, checkpoint,
corruption, validation, selection, telemetry, run-independence and
no-TEST-access suites; all pass.

**Three tests failed on the first run after the change, and all three had
encoded the old behaviour:**

| test | why it failed | resolution |
|---|---|---|
| `test_tabs_and_newlines_are_whitespace` | asserted a newline belongs to no chunk | rewritten as the `\S+\n?` ownership contract, plus a test that the two halves share one unit |
| two scanner tests using `plus_one_offender()` | **the helper can no longer construct a 256→257 text** | inverted into `test_the_plus_one_divergence_mechanism_is_gone`; the reporting path now uses an explicit stub |

The second row is the repair, stated as an impossibility: the divergence
mechanism no longer exists to be constructed.

One further fixture defect was found and fixed rather than worked around: the
scanner's stub tokenizer **stripped the newline from its pieces**, so alignment
correctly refused newline-bearing runs. It is now a faithful transcription of
the real `bpe`, which makes the whole scanner suite trustworthy.

## 8. Real-Corpus Acceptance — STILL REQUIRED

**Local unit tests do not authorise training.** The repaired path has not run
against the real tokenizer or the real corpus. Required, in order, in §10.

The full TRAIN post-repair gate must show:

```
over_max_length                 = 0
max_stage1_realised_base_length <= 256
```

and for every deterministic spot check, `realised == authoritative`. All nine
former offenders must come out at exactly 256 — including
`7d99f2dba18e45c0` (`Mô_đun:Inflation/data#572`) and `Mô_đun:Easter#3`.

Since the scanner completes the corpus in ~98 s, there is no reason to accept
the repair without this gate.

## 9. Standing Facts, Unchanged

* Prepared corpus **unmodified**; `chunks.jsonl` never written.
* Frozen science unchanged — 19 locked values re-read, zero mismatches;
  `TRUNCATION: max_length=256 on_overflow=FAIL`.
* Official UIT-VSFC TEST: **SEALED / UNUSED** — not opened, inspected, mounted,
  searched, tokenized, scanned or evaluated.
* The historical aborted `lr-pilot` remains **250 telemetry-confirmed optimizer
  updates, 0 checkpoints, 0 validations, empty output directory, NOT
  RESUMABLE**. This repair does not make it resumable, and it must never be
  presented as a completed candidate.
* No training, no optimizer step, no backward was executed by this repair.

## 10. Required Acceptance Before Training

1. **Full TRAIN post-repair length acceptance** on the real corpus (§8).
2. **Real PhoBERT no-update smoke.**
3. **CUDA telemetry scientific-equivalence acceptance.**
4. **A clean committed final HEAD.**

Only all four together authorise Stage-1 training.

---

**Status: REPAIR IMPLEMENTED — LOCAL ACCEPTANCE PASS — REAL COLAB/CUDA
ACCEPTANCE REQUIRED.**

*End of Audit 044.*
