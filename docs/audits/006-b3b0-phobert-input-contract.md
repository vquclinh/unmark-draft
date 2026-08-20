# Audit 006 — B3B-0 PhoBERT input contract and token-grid feasibility

| | |
|---|---|
| **Audit id** | 006 |
| **Date (UTC)** | 2026-08-19 |
| **Scope** | B3B-0 preparation only: input-contract structures, Colab probe, local mock tests |
| **Repository state** | `HEAD = 912329d`; B3B-0 work uncommitted |
| **Predecessors** | [002](002-b1a-orthography-core.md), [003](003-b2-deterministic-corruption.md), [004](004-b2-eligibility-safety-followup.md), [005](005-b3a-vietnamese-syllable-eligibility.md) |
| **Phase** | Phase 0 / B3B-0 |

---

## A. Verdict

**PASS — COLAB PROBE REQUIRED**

A real specification issue was identified before any code was written against it: the
proposal writes the token grid as `T(b(x))` while PhoBERT's published contract expects
word-segmented input, i.e. `T(S(b(x)))`. That is recorded as **OPEN** in the decision log
with five candidate pipelines enumerated and no policy chosen. A second, related gap is
recorded: the backbone checkpoint is named but pinned nowhere and is not even listed among
§5's open items. The local package stays ML-free — no transformers, torch, tokenizer,
Java or VnCoreNLP — and the 1428-test suite runs offline against mock tokenizer output.
The probe script exists, loads the tokenizer only, and refuses to run locally. No
alignment, no `inputs_embeds`, no training. B3B is **not** complete; the probe must run on
Colab before any policy is locked.

---

## B. Files changed

**New**

| File | Purpose |
|---|---|
| `unmark/alignment/contracts.py` | `PreprocessingPath`, `PathAvailability`, `OffsetAvailability`, `AlignmentStatus`, `TokenizerContract`, `SegmenterContract` |
| `unmark/alignment/spans.py` | `TokenSpan`, `validate_offsets`, `character_coverage`, `syllable_token_map`, `alignment_status` |
| `unmark/alignment/probe_models.py` | `PathObservation`, `grid_invariance`, `path_summary`, `compare_paths` |
| `unmark/alignment/__init__.py` | Package exports |
| `scripts/b3b0_phobert_input_probe.py` | Colab probe (tokenizer only) |
| `tests/test_alignment_contracts.py` | 46 mock-based tests |
| `results/b3b0/.gitkeep` | Run-artifact directory |

**Modified**

| File | Change |
|---|---|
| `docs/spec/decisions.md` | D-B3B0-001 (OPEN, segmentation) and D-B3B0-002 (OPEN, checkpoint) |
| `README.md` | "PhoBERT input contract (B3B-0) — open question" |

`unmark-proposal.md` **unchanged** — deliberately; see §D.

---

## C. Discovered input-contract issue

**The proposal.** §4.4: "The **base stream** defines the token grid. All positions are
indexed by `T(b(x))`." §4.3: base is "fully stripped letters; tokenized by the frozen
tokenizer". §5.1 locks "stripped text, frozen tokenizer, frozen embedding table". §4.4
step 2 propagates channel labels "by tracking character offsets through tokenization".
**No section mentions word segmentation.**

**PhoBERT.** Its usage contract expects Vietnamese word-segmented input — underscore-joined
compounds such as `nghiên_cứu` — from the VnCoreNLP/RDRSegmenter preprocessing used in
pretraining. §6.1 does describe PhoBERT-base as "word/syllable-level BPE", so the
granularity is acknowledged without saying where segmentation happens.

Operationally: `T(S(b(x)))`, not `T(b(x))`.

**Why it is scientific, not cosmetic.** `S` sits between two things the design depends on,
and four constraints pull against each other:

| Constraint | Source |
|---|---|
| Deployability — inference cannot require clean text | §1.3 |
| Corruption invariance — identical base token ids across conditions | §4.4, §4.5 |
| PhoBERT compatibility — match the pretraining distribution | model card |
| No hidden restoration — segmentation must not become a diacritic restorer | §3.2, the `RESTORE` comparison |
| Reproducibility — segmenter and model pinned | §5.3 |

Two further assumptions are exposed and will be measured rather than assumed: that the
tokenizer provides offsets at all, and that those offsets mean what §4.4 step 2 needs.

**Second finding — the backbone is unpinned.** §6.1 names "PhoBERT-base" but no repository,
no checkpoint revision, no tokenizer revision. Unlike the `RESTORE` checkpoint it is also
**absent from §5's open-items table**, so it is neither locked nor tracked as open.
Recorded as D-B3B0-002.

---

## D. Proposal consistency

| Proposal statement | B3B-0 treatment | Status |
|---|---|---|
| §4.4 "All positions are indexed by `T(b(x))`" | Represented as `PreprocessingPath.RAW_BASE`, one candidate among five | **OPEN** — not contradicted, not confirmed |
| §4.4 step 2 "tracking character offsets through tokenization" | `OffsetAvailability` measures whether this is implementable | **OPEN** |
| §4.5 "`b(x) = b(x̃)` for every corruption rate […] the same token grid" | `grid_invariance()` tests it at three levels: base text, tokenizer input, token ids | **TO BE MEASURED** |
| §4.4 "corrupting the input changes the tone labels but never the base ids" | The decisive probe column | **TO BE MEASURED** |
| §6.1 "PhoBERT-base" | Probe default `vinai/phobert-base`, revision unset and recorded as such | **OPEN** (D-B3B0-002) |
| §5.1 "frozen tokenizer, frozen embedding table" | Respected: tokenizer only, no weights, nothing trained | MATCH |

**The proposal was not rewritten.** No sentence was edited. Writing any candidate path into
§4.4 before the probe returns would make an unmeasured guess normative — the failure mode
audit 004 was created to prevent. Both decision entries record "Proposal source updated:
**NO**".

---

## E. Probe design

`scripts/b3b0_phobert_input_probe.py`, Colab-only:

* prints Python, platform, `transformers` version, tokenizer class, `is_fast`, vocab size,
  UNK, special tokens, checkpoint and revision;
* uses the repo-local `.hf-cache` via `HF_HOME`, as G−1 established;
* loads **tokenizer only** — an AST test asserts no `*Model.from_pretrained` call and no
  `Model` import from `transformers`;
* drives the **real B2 corruption engine** and **B3A eligibility**, so the probe measures
  the actual pipeline rather than a mock of it;
* runs 18 cases × 6 conditions × 4 paths;
* per observation records tokens, ids, offsets, offset availability with a stated reason,
  character coverage, syllable→token mapping, straddling tokens, fragmentation, unknown
  tokens, eligibility and observed tones;
* writes `config.json`, `environment.json`, `cases.jsonl`, `summary.json`, `report.md`;
* reports `decision: NOT_MADE` and says so in the report.

Cases cover ordinary Vietnamese, multi-syllable compounds (`nghiên cứu`, `xử lý`,
`ngôn ngữ`, `tự nhiên`, `thành phố`, `đại học`, `công nghệ`), proper names, mixed
Vietnamese/English, ASCII-ambiguous spans, uppercase, punctuation, numbers, dates, URLs,
e-mail, emoji, hyphenation, pre-underscored input, `đ/Đ`, `ă â ê ô ơ ư`, all six tones and
a long sentence. **No expected tokenizer output is hard-coded anywhere.**

If VnCoreNLP is unavailable, the three segmentation paths are recorded
`UNAVAILABLE_SEGMENTER` — never faked — and the run reports `B3B0_PROBE_PARTIAL`.

---

## F. Paths to be compared

| Path | Pipeline | Principal risk |
|---|---|---|
| `RAW_BASE` | `T(b(x))` | ignores PhoBERT's contract; distribution mismatch, fragmentation |
| `CLEAN_SEGMENT_THEN_BASE` | segment clean → strip | **not deployable**; hidden-restoration risk |
| `BASE_THEN_SEGMENT` | strip → segment base | segmenter out of distribution |
| `OBSERVED_SEGMENT_THEN_BASE` | segment observed → strip | segmentation may vary with corruption, breaking grid invariance |
| `PRESEGMENTED_DATASET` | dataset-supplied | enumerated in `contracts.py`; not probed (no dataset is pinned yet) |

`grid_invariance()` separates three failure levels so a break is attributable:
`base_text_invariant` (B2's guarantee), `tokenizer_input_invariant` (did preprocessing
break it?), `token_ids_invariant` (what §4.4 requires).

---

## G. Local test evidence

| Command | Result |
|---|---|
| `.venv/bin/python -m pytest -q -p no:cacheprovider` | **1428 passed in 6.07s** |
| `tests/test_alignment_contracts.py` | 46 passed |
| Previous suites (G−1, B1A, B2, B3A) | all green, unchanged |
| `pip list` | 7 packages, unchanged — nothing installed |
| `ls ~/.cache/huggingface/hub` | unchanged — no model downloaded |
| local `b3b0_phobert_input_probe.py` | refuses with Colab instructions, exit 2 |

Mock coverage: token spans, half-open overlap, special tokens, zero-width and offsetless
spans, exact/inexact/malformed/absent offsets, BPE continuation markers (`@@`, `##`, `▁`),
out-of-range and reversed offsets, overlapping offsets, offsets that cannot be reconciled
with token strings, character coverage and gaps, syllable→token mapping, fragmentation,
unmapped syllables, straddling tokens, grid invariance at all three levels, error
exclusion, path summaries, JSON serialisation, underscores, punctuation and mixed script.

**These tests prove the analysis logic, not PhoBERT's behaviour**, and the module docstring
says so. No claim about the real tokenizer is made from mocks.

---

## H. Colab requirements

```bash
pip install "transformers==4.57.6"
pip install py_vncorenlp          # optional; needs a JVM, which Colab provides
export HF_HOME="$PWD/.hf-cache"
python scripts/fetch_vietnamese_syllable_inventory.py   # B3A eligibility
python scripts/b3b0_phobert_input_probe.py --checkpoint vinai/phobert-base
```

* Java runtime — supplied by Colab; required only for the segmentation paths.
* `py_vncorenlp` downloads its model into a repo-local `.vncorenlp/`.
* **Reproducibility risk, reported not hidden:** `py_vncorenlp.download_model()` is not
  revision-pinned, so the segmentation model may change between runs. The probe records
  `pinned: false` and the jar filenames, and the report prints a warning. This must be
  pinned before any result depends on segmentation.
* No model weights are downloaded; the tokenizer files are all that is fetched.

---

## I. Decision-log state

| Entry | Status |
|---|---|
| D-B3B0-001 — PhoBERT word segmentation vs `T(b(x))` | **OPEN — empirical feasibility probe required**; owner B3B-0 → B3B; proposal updated **NO** |
| D-B3B0-002 — backbone checkpoint not locked | **OPEN — SPEC LOCK ITEM**; owner B3B / spec lock; proposal updated **NO** |

Both carry original proposal wording, the external requirement, the candidate choices, the
scientific risks, affected areas, owner, and what closes them. The four-way category table
gained an `OPEN — EMPIRICAL PROBE REQUIRED` row, so the later pre-training audit can still
separate intended final spec, temporary fallback, resolved decision and open gap.

---

## J. Blocking issues

`None` for B3B-0 as scoped. D-B3B0-001 blocks **B3B**, which is the point: the probe must
run before alignment is implemented.

---

## K. Non-blocking issues

```
ID: N1  py_vncorenlp model is not revision-pinned
    download_model() fetches current upstream. Reported as pinned:false with jar names.
    Must be pinned before any result depends on segmentation.

ID: N2  PRESEGMENTED_DATASET path is enumerated but not probed
    No dataset is pinned yet (§5 open item), so there is nothing to read segmentation from.

ID: N3  Probe default checkpoint could be mistaken for a lock
    `--checkpoint vinai/phobert-base` is a convenience. The probe records what it actually
    loaded, and D-B3B0-002 states it is not a lock, but a reader skimming the script might
    assume otherwise.

ID: N4  Straddling tokens are detected but no policy is proposed
    A token covering two syllables makes its tone label ambiguous. Measured and flagged;
    resolving it is B3B's problem.

ID: N5  unmark-proposal.pdf
    Still stale from the §4.2 and §5.3 edits. Unchanged by this task.
```

---

## L. What the Colab result must decide

1. **Does any path satisfy grid invariance?** `token_ids_invariant` across `FULL`…
   `STRIP_ALL`. A path that fails is unusable regardless of distribution match — §4.5's
   central claim depends on it.
2. **Are offsets usable?** If `ABSENT` or `NATIVE_MALFORMED`, §4.4 step 2 is not
   implementable as written and B3B must design a deterministic manual alignment.
3. **How much does segmentation change fragmentation?** §6.8's subwords-per-syllable
   diagnostic, per path and per condition.
4. **Does segmentation leak restoration?** Compare `CLEAN_SEGMENT_THEN_BASE` against
   `BASE_THEN_SEGMENT` and `OBSERVED_SEGMENT_THEN_BASE`: if segmenting clean text produces
   materially different groupings, segmentation carries diacritic information and cannot be
   used on a path claiming no restoration.
5. **Is the deployable path acceptable?** Only paths reading observed-or-stripped text
   survive §1.3.
6. **Which checkpoint, at which revision?** D-B3B0-002 must be closed before any number is
   produced.

Only after those are answered should §4.4 be amended and B3B implemented.

---

## M. Git state

* **Branch:** `main`
* **HEAD:** `912329d` — the researcher's commit; the reflog shows no commit from this
  session
* **Staged:** none (`git diff --cached --stat` empty)
* **Unstaged:** `README.md`, `docs/spec/decisions.md`
* **Untracked:** `docs/audits/006-b3b0-phobert-input-contract.md`, `results/b3b0/`,
  `scripts/b3b0_phobert_input_probe.py`, `tests/test_alignment_contracts.py`,
  `unmark/alignment/`

No `add`, `commit`, `push`, `tag`, `stash`, `reset`, `checkout` or `restore` was run.

```text
AUDIT FILE WRITTEN: docs/audits/006-b3b0-phobert-input-contract.md
OTHER AUDIT MODIFICATIONS MADE: NO
COMMIT CREATED: NO
```
