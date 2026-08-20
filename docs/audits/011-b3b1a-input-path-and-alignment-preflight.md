# Audit 011 — B3B-1A input path locked, alignment preflight

| | |
|---|---|
| **Audit id** | 011 |
| **Date (UTC)** | 2026-08-20 |
| **Scope** | Persist the RAW_BASE decision; repair the eligibility integration bug; build the manual-alignment core and Colab probe |
| **Repository state** | `HEAD = 48c44cd`; this work uncommitted |
| **Predecessors** | [006](006-b3b0-phobert-input-contract.md)–[010](010-b3b0-tokenizer-revision-verification.md) |
| **Phase** | Phase 0 / B3B-1A |

---

## A. Verdict

**PASS — COLAB ALIGNMENT PROBE REQUIRED**

The scientifically usable B3B-0 run settled the preprocessing question: **`RAW_BASE` is the
selected main UNMARK base path**, and **D-B3B0-001 is CLOSED**. The all-`UNDECIDED`
eligibility defect was traced to an actual cause — reproduced locally, not assumed — and
repaired at the library level. A deterministic manual-alignment core is implemented for the
slow tokenizer's fastBPE pieces, with an explicit failure policy, and a Colab probe is ready
to test it. 1572 tests pass offline.

Stated explicitly:

* **`RAW_BASE` is the selected main UNMARK base path.**
* **D-B3B0-001 is CLOSED.**
* **D-B3B0-002 remains OPEN** — the backbone checkpoint is not locked.
* **`OBSERVED_SEGMENT_THEN_BASE` is rejected** (breaks grid invariance, 9/18).
* **`CLEAN_SEGMENT_THEN_BASE` is diagnostic only** — non-deployable.
* **`BASE_THEN_SEGMENT` is not selected for the main method.**
* **The standard PhoBERT word-segmentation mismatch is explicitly documented**, in the
  proposal, the decision log and the README.
* **The B3A eligibility diagnostic bug is repaired.**
* **The slow PhoBERT tokenizer remains the token-ID authority.**
* **Manual alignment is not validated** until the Colab probe runs.
* **No model weights were loaded. Nothing was trained.**

---

## B. Files changed

**New**

| File | Purpose |
|---|---|
| `unmark/alignment/manual.py` | `align_span`, `piece_surface`, `reconstruct_surface`, `compare_sequences`, `summarize_alignments`, failure taxonomy |
| `scripts/b3b1_phobert_alignment_probe.py` | Colab manual-alignment probe (tokenizer only) |
| `tests/test_manual_alignment.py` | 39 mock-BPE tests |
| `docs/experiments/b3b0-input-contract-result.md` | Persistent record of the scientific B3B-0 run |
| `results/b3b1/.gitkeep` | Run-artifact directory |

**Modified**

| File | Change |
|---|---|
| `unmark/linguistics/inventory.py` | `DEFAULT_MANIFEST` anchored to the repository — the eligibility bug fix |
| `unmark/alignment/__init__.py` | Exports the manual-alignment API |
| `unmark-proposal.md` | §4.4 states the PhoBERT branch is literal `T(b(x))`, with the trade-off |
| `docs/spec/decisions.md` | D-B3B0-001 CLOSED; D-B3B1A-001/002/003 added |
| `README.md` | "Input path: RAW_BASE (locked)" |

---

## C. Scientific B3B-0 evidence

Run `20260820T031644Z`, repository HEAD `48c44cdc597614eb06abd52c4fe16e8ab5235c07`,
`scientifically_usable: true`. PhoBERT `vinai/phobert-base` at
`01daacda68afe13d83023d16ec647239e344a1e6`, requested == observed, `revision_verified: true`,
`PhobertTokenizer`, `is_fast: false`, `model_weights_loaded: false`. VnCoreNLP at
`62bbc58fe5d113c898eae112656be97dcf50b3a0`, revision and hashes verified, `pinned: true`.

18 cases × 6 conditions = 108 observations per path:

| Path | Grid invariant | Mean fragmentation | Unknown | Offsets |
|---|---|---:|---:|---|
| `RAW_BASE` | 18/18 | 1.5674165421972441 | 12 | ABSENT |
| `BASE_THEN_SEGMENT` | 18/18 | 1.5424165421972438 | 12 | ABSENT |
| `CLEAN_SEGMENT_THEN_BASE` | 18/18 | 1.6191500426149548 | 12 | ABSENT |
| `OBSERVED_SEGMENT_THEN_BASE` | 9/18 | 1.5762836172923893 | 12 | ABSENT |

Observed-segment failures: `vi_research`, `vi_multisyllable`, `vi_city`, `vi_proper_names`,
`vi_uppercase`, `email`, `emoji`, `hyphenated`, `long_sentence`.

Researcher analysis of all 432 observations, transcribed into
`docs/experiments/b3b0-input-contract-result.md`: the clean-segment path is not deployable;
the observed-segment path breaks invariance; base-then-segment recovers little segmentation
(**8 underscores versus 39** — labelled a *post-hoc diagnostic count, not a formal
word-segmentation accuracy metric*) and produces different merges
(`Truong Dai_hoc Khoa_hoc_Tu_nhien` vs `Truong_Dai hoc Khoa hoc Tu nhien`); the segmenter
mangled supplementary-plane emoji in at least one diagnostic (recorded, not generalised);
and the fragmentation gain is small with an identical unknown count.

The invalid first Colab run remains excluded.

---

## D. D-B3B0-001 closure

**CLOSED** by D-B3B1A-001. The entry records the original proposal wording, the discovered
PhoBERT segmentation requirement, the probe evidence, the selected policy, the status of
each rejected or deferred alternative, the reasons, the affected components, and that the
proposal source was updated (YES) and the compiled PDF is stale (YES).

---

## E. RAW_BASE contract

```text
MAIN UNMARK BASE PATH = RAW_BASE
tokenizer_input = b(x)
token_grid      = T(b(x))
```

No VnCoreNLP segmentation between `b(x)` and `T` on the main UNMARK base stream. The
proposal's notation is now literal for the PhoBERT branch.

Reasons: deployable; needs no clean text at inference; exactly corruption-invariant; adds no
hidden restoration or clean-segmentation side information; minimal preprocessing; free of
post-strip segmenter side effects; and no compelling empirical benefit from the alternative.

---

## F. Rejected / diagnostic paths

| Path | Status |
|---|---|
| `RAW_BASE` | **SELECTED** — main UNMARK base/deployment path |
| `BASE_THEN_SEGMENT` | **NOT SELECTED FOR MAIN METHOD**; retained only as a possible later ablation/diagnostic, no compute spent now |
| `CLEAN_SEGMENT_THEN_BASE` | **DIAGNOSTIC ONLY** — non-deployable, requires clean text; may serve as an upper-bound preprocessing diagnostic, never as a system |
| `OBSERVED_SEGMENT_THEN_BASE` | **REJECTED** — violates base-token-grid invariance |

---

## G. PhoBERT preprocessing trade-off

Recorded in the proposal, the decision log and the README, not buried:

* standard PhoBERT usage expects **pre-word-segmented** Vietnamese;
* UNMARK **intentionally departs** from that on its base branch, because every clean or
  observed segmentation alternative conflicts with deployability, invariance, or the probe;
* this is a deliberate experiment-design choice and **a possible source of distribution
  shift**;
* clean-reference and baseline preprocessing is a **separate** issue, to be locked when
  those pathways are implemented.

**`RAW_BASE` is not claimed to match PhoBERT's pretraining preprocessing.**

---

## H. B3A eligibility integration bug

**Observed.** All 432 artifact observations recorded `Eligibility.UNDECIDED` — 3960 labels,
zero `VIETNAMESE_CANDIDATE`, zero `NOT_APPLICABLE`.

**Actual cause, reproduced rather than assumed.** `DEFAULT_MANIFEST` in
`unmark/linguistics/inventory.py` was the relative string
`configs/linguistics/vietnamese_syllables.yaml`. `py_vncorenlp.VnCoreNLP()` `chdir()`s into
its resource directory *before* the probe's case loop, so `try_load_inventory()` resolved
the manifest against the wrong directory, returned `None`, and no classifier was injected.
Reproduced locally:

```text
cwd == repo root : inventory loads True
after chdir      : inventory loads False
```

This is the same class of defect as audit 007's output-path drift — fixed there for output
paths, still latent in the library's own default.

**A consequence I must not gloss over.** The same lookup failure means the probe's B2
corruption also ran under the **provisional candidate-span policy**, not the resolved one.
Scope: only `OBSERVED_SEGMENT_THEN_BASE` consumes corrupted text; the other three paths
tokenize `base_text` or clean text, which is invariant under both corruption and the
eligibility policy. So the path decision stands, `RAW_BASE`'s numbers are unaffected, and
the observed-segment rejection holds under either policy — but the **eligibility labels in
that artifact carry no information and must not be read**. This is recorded in the
experiment record and in D-B3B1A-002.

**Fix.** `DEFAULT_MANIFEST` is anchored via `__file__`, so inventory loading — and with it
`corrupt()`'s policy resolution — no longer depends on the caller's cwd. Verified after a
`chdir` that eligibility resolves to `[VIETNAMESE_CANDIDATE, VIETNAMESE_CANDIDATE,
NOT_APPLICABLE]` for `toi dung Python`, and that `corrupt()` runs scientifically. The B3B-1
probe additionally uses `load_inventory()` rather than `try_load_inventory()`, so a missing
inventory fails loudly instead of degrading to `UNDECIDED`.

**B3A's scientific eligibility semantics are unchanged.**

---

## I. Offset finding

`offset_availability = ABSENT` for every path. The authoritative tokenizer is
`PhobertTokenizer` with `is_fast = false` and returns no `offset_mapping`, so proposal §4.4
step 2 — "tracking character offsets through tokenization" — is not implementable as written.

**The token-grid authority does not move.** The frozen ids from the pinned slow tokenizer
remain authoritative; no scientific path switches to a fast implementation merely because it
offers offsets. An optional diagnostic may compare `tokenizer.json` ids against the slow
ones, but it is explicitly non-authoritative and its offsets are never adopted.

---

## J. Manual alignment hypothesis

Tokenize each base span independently with the same frozen slow tokenizer, no special
tokens, then strip the fastBPE `@@` continuation marker and reconstruct:

```text
nghien -> ["ngh@@", "ien"] -> ngh + ien == nghien -> ranges [0,3) [3,6)
```

Implemented in `unmark/alignment/manual.py`. The invariants the Colab probe must test:

| | Invariant |
|---|---|
| A | per-span composition equals the authoritative full-sequence tokenization |
| B | surface reconstruction equals the exact base syllable |
| C | no eligible Vietnamese syllable produces `<unk>` (enumerated if any do) |
| D | each piece has an exact half-open character range in the span |
| E | special tokens, punctuation and non-Vietnamese spans are `N/A` |

**None of these is assumed.** The module implements the hypothesis; the probe measures it.
Nothing in this task declares manual alignment validated.

---

## K. Alignment failure policy

Nothing is ever labelled on a guess.

| Situation | Outcome |
|---|---|
| pieces reconstruct the span exactly | `ALIGNED`, ranges emitted |
| `<unk>` among the pieces | `ALIGNMENT_FAILURE` / `UNKNOWN_TOKEN` |
| reconstruction ≠ span | `ALIGNMENT_FAILURE` / `SURFACE_MISMATCH` |
| final piece still carries `@@` | `ALIGNMENT_FAILURE` / `MALFORMED_CONTINUATION` |
| no pieces | `ALIGNMENT_FAILURE` / `NO_TOKENS` |
| eligibility `UNDECIDED` | `ALIGNMENT_FAILURE` / `UNRESOLVED_ELIGIBILITY` |
| special / punctuation / non-Vietnamese | `NOT_APPLICABLE` in both channels |

A failed alignment exposes **no** ranges, and `carries_channels` is true only for an
aligned, resolved `VIETNAMESE_CANDIDATE`.

---

## L. Colab probe design

`scripts/b3b1_phobert_alignment_probe.py`, Colab CPU, tokenizer only:

* `vinai/phobert-base` at `01daacda68afe13d83023d16ec647239e344a1e6` — the same **probe**
  revision as B3B-0, for comparability — verified post-load by the audit-010 mechanism, with
  a full-SHA requirement and a fail-closed mismatch (exit 3);
* `use_fast=False`: the slow tokenizer is the id authority;
* **every unique stripped form** in the B3A inventory is aligned, reporting total forms,
  tokenizable forms, `<unk>` forms, surface-reconstruction failures, and mean/max subwords
  per syllable;
* curated coverage of all six tones, every letter diacritic, uppercase, and NFC/NFD;
* the 13 representative sentences: mixed script, punctuation, URL, e-mail, emoji,
  hyphenation, long sentence;
* full-sequence consistency compares the authoritative tokenization against a composition
  over **every region** of the input — punctuation and non-candidate spans included, not
  just the Vietnamese spans — so any unexplained token is reported;
* the eligibility histogram is reported per case, and `UNDECIDED` must be zero;
* optional `--fast-diagnostic` compares `tokenizer.json` ids against the slow ones, recorded
  as a diagnostic and never promoted to authority;
* artifacts: `config.json`, `summary.json`, `cases.jsonl`, `inventory_failures.jsonl`,
  `report.md`.

---

## M. Proposal / decision-log consistency

| Surface | State |
|---|---|
| `unmark-proposal.md` §4.4 | **Updated (YES).** States that `T(b(x))` is literal for the PhoBERT branch, with no post-strip segmenter, and records the deliberate trade-off and its distribution-shift risk. General method notation preserved; no unrelated section touched. |
| Compiled PDF | **Stale (YES).** Not regenerated. |
| D-B3B0-001 | **CLOSED** by D-B3B1A-001; prior status preserved in the entry. |
| D-B3B0-002 | **OPEN.** `01daacda…` is a verified *probe* revision, not the paper backbone lock. |
| D-B3B1A-001 | New — RAW_BASE selection with full evidence. |
| D-B3B1A-002 | New — eligibility integration repair. |
| D-B3B1A-003 | New — offsets absent; manual alignment OPEN pending Colab. |

---

## N. Test evidence

| Command | Result |
|---|---|
| `.venv/bin/python -m pytest -q -p no:cacheprovider` | **1572 passed in 8.54s** |
| `tests/test_manual_alignment.py` | 39 passed |
| All previous suites (G−1, B1A, B2, B3A, B3B-0) | green |
| `pip list` | 7 packages, unchanged — nothing installed |

Local tests require no network, transformers, torch, Java or VnCoreNLP; alignment is
exercised against **mock BPE sequences**, and nothing claims correctness against the real
tokenizer from them.

Covered: `@@` reconstruction, one-piece and multi-piece syllables, exact half-open ranges,
repeated substrings (`toitoi` yields distinct ranges), uppercase, punctuation adjacency,
special tokens, non-Vietnamese `N/A`, unknown-token failure, malformed continuation,
surface mismatch, per-span/full-sequence mismatch and reordering, deterministic
serialisation, resolved B3A eligibility propagation, and that `UNDECIDED` cannot silently
enter a resolved scientific alignment. Two tests reproduce the cwd bug directly by
`chdir`-ing to `/` and asserting the inventory still loads and eligibility still resolves.

---

## O. Blocking issues

`None` for B3B-1A as scoped.

D-B3B1A-003 blocks **B3B proper**: manual alignment must be validated on the real tokenizer
before channel propagation is implemented.

---

## P. Non-blocking issues

```
ID: N1  Eligibility labels in the B3B-0 artifact are uninformative
    All UNDECIDED because of the cwd bug. The path decision does not depend on them, and
    the experiment record says so explicitly, but the artifact cannot be mined for
    eligibility statistics. A rerun would be needed for that, and is not required for the
    RAW_BASE decision.

ID: N2  The probe's B2 corruption ran provisionally
    Same root cause. Affects only OBSERVED_SEGMENT_THEN_BASE's specific outputs; its
    rejection reason holds under either policy. Recorded rather than glossed.

ID: N3  MALFORMED_CONTINUATION is a judgement call
    A final piece carrying `@@` is treated as a failure, on the grounds that the span's
    tokenization is not self-contained. If the real tokenizer legitimately emits this,
    the policy needs revisiting — the Colab probe will show it.

ID: N4  compare_sequences uses multiset removal for the unexplained list
    Order-sensitive mismatches are correctly reported as inconsistent, but the
    "unexplained tokens" list is computed by removal and can be imprecise when the same
    token appears several times. The boolean verdict is exact; the list is diagnostic.

ID: N5  unmark-proposal.pdf
    Stale, now also with respect to the 4.4 edit. Unchanged by this task.
```

---

## Q. Git state

* **Branch:** `main`
* **HEAD:** `48c44cd` — the researcher's commit; the reflog shows no commit from this session
* **Staged:** none (`git diff --cached --stat` empty)
* **Unstaged:** `README.md`, `docs/spec/decisions.md`, `unmark-proposal.md`,
  `unmark/alignment/__init__.py`, `unmark/linguistics/inventory.py`
* **Untracked:** `docs/audits/011-b3b1a-input-path-and-alignment-preflight.md`,
  `docs/experiments/`, `results/b3b1/`, `scripts/b3b1_phobert_alignment_probe.py`,
  `tests/test_manual_alignment.py`, `unmark/alignment/manual.py`
* **Ignored by design:** `.vncorenlp/`, `.probe_*`, `.hf-cache/`, `.resources-cache/`

No `add`, `commit`, `push`, `tag`, `stash`, `reset`, `checkout` or `restore` was run.

```text
AUDIT FILE WRITTEN: docs/audits/011-b3b1a-input-path-and-alignment-preflight.md
OTHER AUDIT MODIFICATIONS MADE: NO
COMMIT CREATED: NO
```
