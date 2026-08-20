# Audit 012 — B3B-1B whitespace-chunk alignment repair

| | |
|---|---|
| **Audit id** | 012 |
| **Date (UTC)** | 2026-08-20 |
| **Scope** | Repair the manual-alignment hypothesis using real slow-tokenizer evidence |
| **Repository state** | `HEAD = 8e8f3a7`; this work uncommitted |
| **Predecessors** | [010](010-b3b0-tokenizer-revision-verification.md), [011](011-b3b1a-input-path-and-alignment-preflight.md) |
| **Phase** | Phase 0 / B3B-1B |

---

## A. Verdict

**PASS — CORRECTED COLAB ALIGNMENT PROBE REQUIRED**

The first real B3B-1 run refuted the B3B-1A hypothesis at span granularity, and the
researcher's follow-up diagnosis identified why. Both defects are repaired: alignment now
runs over **maximal non-whitespace chunks**, the units PhoBERT's fastBPE actually operates
on, and **vocabulary OOV is no longer conflated with surface irrecoverability**. 1576 tests
pass offline against mock raw-BPE sequences reproducing the observed tokenizer behaviour.

Stated explicitly:

* **The previous 6/13 run is superseded as final alignment evidence** — retained, not
  deleted or rewritten.
* **Its failure came from wrong span granularity**, not from an inability to reconstruct
  PhoBERT's tokenization.
* **13/13 whole-whitespace-chunk agreement was observed** in the researcher's diagnostic,
  on both tokens and ids, with 119/119 chunk surfaces reconstructed.
* **`khut` is raw-surface recoverable despite an unknown vocabulary id.**
* **An unknown id is no longer automatically an alignment failure.**
* **`RAW_BASE` remains locked** (D-B3B0-001 CLOSED).
* **The slow PhoBERT tokenizer remains the token-ID authority.**
* **Manual alignment is still not finally validated** until the corrected probe reruns.
* **No model weights were loaded. Nothing was trained.**

---

## B. Files changed

| File | Change |
|---|---|
| `unmark/alignment/manual.py` | Rewritten: `whitespace_chunks`, `align_chunk`, `overlay_orthography`, `compose`, `verify_token_grid`, `summarize_chunk_alignments`; `PieceAlignment` gains local **and** global ranges plus `has_unknown_token_id`; `UNKNOWN_TOKEN` removed from the failure taxonomy; `RANGE_ERROR` added; `ToneOwnership` added |
| `unmark/alignment/__init__.py` | Exports the chunk API; span-level API withdrawn |
| `scripts/b3b1_phobert_alignment_probe.py` | Chunk-based probe; raw `tokenize()` instead of an id round trip; token **and** id grid verification; orthographic overlay; unknown-id forms enumerated; new validation criteria |
| `tests/test_manual_alignment.py` | Rewritten — 44 tests over the observed tokenizer shapes |
| `docs/spec/decisions.md` | D-B3B1A-003 marked SUPERSEDED; D-B3B1B-001 and D-B3B1B-002 added |
| `unmark-proposal.md` | §4.4 step 2 states offsets are reconstructed over the tokenizer's own units, not read |

`unmark-proposal.pdf` untouched.

---

## C. First B3B-1 real run

Provenance was sound: `vinai/phobert-base` at `01daacda68afe13d83023d16ec647239e344a1e6`,
`PhobertTokenizer`, `is_fast: false`, no model weights. Results:

| | |
|---|---:|
| inventory unique stripped forms | 2489 |
| "aligned" under the old implementation | 2488 |
| failed | 1 (`khut` → `UNKNOWN_TOKEN`) |
| full-sentence sequence consistency | **6/13** |
| `UNDECIDED` | 0 |

Correctly marked `B3B1_ALIGNMENT_PROBE_INCOMPLETE`.

**What was valid and is retained:** the eligibility result (`UNDECIDED = 0`, confirming
audit 011's cwd repair worked) and the tokenizer provenance. **What was wrong:** the
alignment granularity, and the `khut` failure classification. The run is an informative but
**superseded** diagnostic; it is not the final B3B alignment evidence and has not been
deleted or rewritten.

---

## D. Root cause of the 6/13 consistency

PhoBERT's authoritative slow tokenizer applies BPE over **maximal non-whitespace chunks**,
not over the smaller linguistic spans produced by the B1A/B3A scanner. Punctuation,
hyphens, URLs and e-mail addresses therefore change the segmentation of the entire chunk:

| Chunk | Authoritative | Span-composed |
|---|---|---|
| `nhien.` | `["nhi@@", "en@@", "."]` | `["nh@@", "ien"]` + `["."]` |
| `VNU-HCM` | `["VN@@", "U-@@", "HCM"]` | — |
| `(VAT` | `["(@@", "VAT"]` | — |
| `Viet-Nam` | `["Viet@@", "-@@", "Nam"]` | — |

URLs and e-mail addresses reconstructed **exactly** from their raw pieces when taken as
whole chunks. Every span still reconstructed its own surface perfectly — the surfaces were
never the problem. The composition simply was not over the units the tokenizer uses.

---

## E. Whitespace-chunk diagnostic evidence

The researcher split every sentence's `base_text` on `\S+` and tokenized each whole chunk
with the same authoritative tokenizer, across all 13 representative cases
(`vi_research`, `vi_multisyllable`, `vi_city`, `vi_proper_names`, `vi_uppercase`,
`mixed_en`, `mixed_ml`, `punctuation`, `url`, `email`, `emoji`, `hyphenated`,
`long_sentence`):

| | |
|---|---:|
| whitespace composition token matches | **13/13** |
| whitespace composition token-ID matches | **13/13** |
| total non-whitespace chunks | 119 |
| chunk surface reconstruction failures | **0** |

This is the evidence the repaired contract is built on.

---

## F. `khut` / unknown-id finding

```text
tokenizer.tokenize("khut")   -> ["khut"]     raw surface recoverable
tokenizer.bpe("khut")        -> "khut"
raw token ids                -> [3]          the unknown id
convert_ids_to_tokens([3])   -> ["<unk>"]    the round trip destroys the surface
direct encode ids            -> [3]
surface reconstruction exact -> TRUE
```

`khut` is **not** a surface-alignment failure. The surface is recoverable *before*
vocabulary lookup; B3B-1A read it *after*, and so lost it.

The two concepts are now separate. A piece may have `token_id == unk_token_id` while its
raw surface and character range are exact — that is `ALIGNED`, with
`has_unknown_token_id = True` recorded, and it may carry orthography channels when the
intersecting orthographic region is resolved and valid.

This is a **general policy**: the literal string `khut` appears nowhere in the
implementation, and `AlignmentFailureReason` no longer has an `UNKNOWN_TOKEN` member — a
test asserts it cannot return.

---

## G. Repaired authoritative alignment contract

```text
authoritative token grid = T(b(x))      from the pinned slow tokenizer, always
auxiliary character map  = chunk-level raw-BPE reconstruction + orthographic overlay
```

1. take exact `b(x)`;
2. maximal non-whitespace chunks, preserving global half-open ranges;
3. tokenize each **whole chunk**, no special tokens;
4. use **raw** BPE strings, before any id→token round trip;
5. reconstruct by stripping the real continuation marker;
6. require `reconstructed_chunk == original_chunk`;
7. derive an exact local range per piece;
8. translate to global ranges in `b(x)`;
9. retain the authoritative full-sequence tokens and ids;
10. verify chunk composition equals both, exactly;
11. overlay B1A/B3A spans onto the global ranges.

**The alignment subsystem never defines the token grid.** A test asserts `align_span` — the
span-level retokenization primitive — cannot reappear in the module.

---

## H. Character-range / orthographic-overlay contract

Every piece exposes `local_start/local_end` within its chunk and
`global_start/global_end` within `b(x)`. Ranges are verified monotonic and tiling: a gap,
overlap or short/long cover is a `RANGE_ERROR`, not a silent acceptance.

Attribution is by **character-range overlap**, because BPE and linguistic boundaries do not
coincide. `overlay_orthography` records **every** contributing region with its exact
overlap range and assigns `ToneOwnership`:

| Ownership | Meaning |
|---|---|
| `VIETNAMESE` | every contributing character comes from one Vietnamese candidate span |
| `NOT_APPLICABLE` | no contributing character comes from a Vietnamese candidate span |
| `MIXED` | the piece straddles a Vietnamese candidate **and** something else |
| `UNRESOLVED` | a contributing span's eligibility is `UNDECIDED` |

A `MIXED` piece is **not** claimed to be Vietnamese and carries no tone. Letter-diacritic
pooling will later draw only from the exact characters assigned to a piece, which the global
ranges make possible. **No trainable tables, no pooling layers, no embeddings** were
implemented — this task produces alignment metadata only.

---

## I. True alignment failure policy

| Situation | Outcome |
|---|---|
| raw surface reconstructs, ranges tile the chunk | `ALIGNED` |
| unknown vocabulary id, surface exact | `ALIGNED` + `has_unknown_token_id` |
| reconstruction ≠ chunk | `ALIGNMENT_FAILURE` / `SURFACE_MISMATCH` |
| final piece carries the marker | `ALIGNMENT_FAILURE` / `MALFORMED_CONTINUATION` |
| non-monotonic / non-tiling ranges | `ALIGNMENT_FAILURE` / `RANGE_ERROR` |
| no pieces | `ALIGNMENT_FAILURE` / `NO_TOKENS` |
| unexplained authoritative token | reported by `verify_token_grid`; invalidates the run |
| contributing span `UNDECIDED` | `ToneOwnership.UNRESOLVED`; no channels |

A failed alignment exposes **no** ranges.

---

## J. Inventory probe contract

All 2,489 unique stripped B3A forms are re-run under the corrected policy, reporting
separately:

`total_unique_stripped_forms`, `raw_surface_reconstructable_forms`,
`forms_with_unknown_token_id`, `raw_surface_reconstruction_failures`, `range_failures`,
`mean_subwords_per_form`, `max_subwords_per_form` — plus an explicit
`unknown_token_id_forms.jsonl` listing every affected form.

An unknown vocabulary id is **not** counted as a surface failure. The diagnostic suggests
2489/2489 reconstructable with one unknown-id form, but **nothing is hard-coded**: the
Colab rerun must establish it.

Curated coverage is retained: all six tones, every Vietnamese letter-diacritic category,
uppercase, and NFC/NFD inputs. The NFC/NFD cases are recorded as measured; no claim is made
that normalisation changes PhoBERT input after canonical base generation.

---

## K. Full-sequence invariants

For every representative sentence the corrected probe verifies:

```text
chunk-composed raw tokens == authoritative full-sequence raw tokens
chunk-composed ids        == authoritative full-sequence token ids
```

Both must hold, and no authoritative token may be unexplained. The target from the
diagnostic is 13/13 on each. **If the repaired probe does not reproduce it, the run reports
`B3B1_ALIGNMENT_PROBE_INCOMPLETE`** — status is computed from tokens matching, ids matching,
zero inventory failures and zero unexplained tokens. Unknown vocabulary ids are reported and
never enter that decision.

---

## L. Proposal / decision-log consistency

| Surface | State |
|---|---|
| `unmark-proposal.md` §4.4 step 2 | **Updated (YES).** States that PhoBERT offsets are reconstructed from raw fastBPE pieces over the tokenizer's own maximal non-whitespace units and overlaid with orthographic spans; records that span-level reconstruction was tried and refuted (6/13); notes that the authoritative grid always comes from `T(b(x))` and that spans are metadata, not tokenization boundaries. Narrow edit; no unrelated section touched; no implementation variable names in the prose. |
| Compiled PDF | **Stale (YES).** |
| D-B3B0-001 | **CLOSED** — `RAW_BASE`. |
| D-B3B0-002 | **OPEN** — final backbone lock. `01daacda…` remains a probe revision. |
| D-B3B1A-003 | **SUPERSEDED** — the hypothesis was tested and refuted at span granularity. |
| D-B3B1B-001 | New — chunk-level contract with the full evidence chain. |
| D-B3B1B-002 | New — OOV policy RESOLVED; mixed-contributor tone assignment **OPEN**. |

---

## M. Test evidence

| Command | Result |
|---|---|
| `.venv/bin/python -m pytest -q -p no:cacheprovider` | **1576 passed in 7.41s** |
| `tests/test_manual_alignment.py` | 44 passed |
| All previous suites (G−1, B1A, B2, B3A, B3B-0) | green |
| `pip list` | 7 packages, unchanged — nothing installed |

No network, transformers, torch, Java or VnCoreNLP. Mock pieces reproduce sequences observed
from the real tokenizer (`nhien.`, `(VAT`, `Viet-Nam`, `VNU-HCM`, URL, e-mail), so the logic
is exercised against real behaviour — while claiming nothing about PhoBERT itself.

Covered: chunk extraction with exact global ranges; multiple spaces; tabs and newlines;
leading/trailing whitespace; attached and leading punctuation; hyphenation; acronyms; URL
and e-mail chunks; raw reconstruction; local ranges; global translation; monotonic tiling;
repeated substrings; token composition; id composition; token and id mismatch failure;
length mismatch; unknown id staying `ALIGNED`; unknown id reported separately; `UNKNOWN_TOKEN`
absent from the taxonomy; genuine surface mismatch; malformed continuation; overlay
attribution; a piece crossing two regions recorded as `MIXED` rather than collapsed;
`UNDECIDED` unable to carry channels; exact overlap ranges; deterministic serialisation; no
model loading; no fast-tokenizer authority; raw `tokenize()` rather than an id round trip.

---

## N. Colab rerun command

```bash
cd unmark-draft
pip install "transformers==4.57.6"
export HF_HOME="$PWD/.hf-cache"

python scripts/fetch_vietnamese_syllable_inventory.py

python scripts/b3b1_phobert_alignment_probe.py \
    --checkpoint vinai/phobert-base \
    --revision 01daacda68afe13d83023d16ec647239e344a1e6
```

Writes a **new** run directory under `<repo>/results/b3b1/<run_id>/` —
`config.json`, `summary.json`, `cases.jsonl`, `inventory_failures.jsonl`,
`unknown_token_id_forms.jsonl`, `report.md`. The previous run is not overwritten.

Optional: `--fast-diagnostic` compares `tokenizer.json` ids against the slow ones. It is
recorded as a diagnostic and is never promoted to authority.

Read `status`: `B3B1_ALIGNMENT_PROBE_COMPLETE` requires tokens **and** ids matching on every
sentence, zero inventory failures and no unexplained token.

---

## O. Blocking issues

`None` for B3B-1B as scoped.

Real-tokenizer validation blocks **B3B proper**: channel propagation must not be implemented
until the corrected probe passes.

---

## P. Non-blocking issues

```
ID: N1  Mixed-contributor tone assignment is OPEN
    A BPE piece straddling a Vietnamese candidate and punctuation is recorded with its
    contributors and marked MIXED. No deterministic assignment rule is proposed, because
    the evidence does not yet support one. The corrected probe reports the frequency;
    B3B decides. D-B3B1B-002.

ID: N2  Chunk-level probing costs 2,489 extra tokenizer calls
    One per inventory form, plus one per chunk per sentence. Trivial on CPU, but the run
    is slower than the span-level version it replaces.

ID: N3  MALFORMED_CONTINUATION remains a judgement call
    A final piece carrying `@@` is treated as failure, on the grounds that the chunk's
    tokenization is not self-contained. Carried over from audit 011 N3; the corrected
    probe will show whether the real tokenizer ever emits it.

ID: N4  The superseded 6/13 run stays on disk
    Deliberately retained as an informative diagnostic. Its alignment numbers must not be
    quoted as B3B evidence; its eligibility and provenance results remain valid.

ID: N5  unmark-proposal.pdf
    Stale, now also with respect to the 4.4 step 2 edit.
```

---

## Q. Git state

* **Branch:** `main`
* **HEAD:** `8e8f3a7` — the researcher's commit; the reflog shows no commit from this session
* **Staged:** none (`git diff --cached --stat` empty)
* **Unstaged:** `docs/spec/decisions.md`, `unmark-proposal.md`,
  `unmark/alignment/__init__.py`, `unmark/alignment/manual.py`,
  `scripts/b3b1_phobert_alignment_probe.py`, `tests/test_manual_alignment.py`
* **Untracked:** `docs/audits/012-b3b1b-whitespace-chunk-alignment-repair.md`
* **Ignored by design:** `.vncorenlp/`, `.probe_*`, `.hf-cache/`, `.resources-cache/`

No `add`, `commit`, `push`, `tag`, `stash`, `reset`, `checkout` or `restore` was run.

```text
AUDIT FILE WRITTEN: docs/audits/012-b3b1b-whitespace-chunk-alignment-repair.md
OTHER AUDIT MODIFICATIONS MADE: NO
COMMIT CREATED: NO
```
