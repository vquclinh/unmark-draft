# Audit 007 — B3B-0 Colab probe reproducibility repair

| | |
|---|---|
| **Audit id** | 007 |
| **Date (UTC)** | 2026-08-19 |
| **Scope** | Probe-infrastructure repair only: VnCoreNLP provisioning, output-path resolution, revision pinning |
| **Repository state** | `HEAD = 7b17322` (B3B-0 probe committed by the researcher mid-task); this repair uncommitted |
| **Predecessor** | [Audit 006](006-b3b0-phobert-input-contract.md) — its probe design stands; two implementation bugs are fixed here |
| **Phase** | Phase 0 / B3B-0 |

---

## A. Verdict

**PASS — COLAB PROBE RERUN REQUIRED**

Both bugs the first real Colab run exposed are fixed. The probe no longer downloads
VnCoreNLP at all: resources must be provisioned externally, required files are checked,
their SHA-256 recorded, and `pinned=true` claimed only when this run verified every file
against hashes supplied to it. Every path is resolved absolutely at process start, before
any dependency runs, so `py_vncorenlp`'s `chdir()` can no longer relocate artifacts — and
the cwd change is now reported rather than hidden. `--revision` is required, failing closed
on a floating tokenizer revision. 1452 tests pass offline with no transformers, torch, Java
or VnCoreNLP.

**No scientific conclusion is drawn from the first run.** It is recorded as an invalid
probe run: its segmentation-resource provenance was not guaranteed, so any measurement
depending on segmentation is unattributable. D-B3B0-001 — whether PhoBERT's word
segmentation belongs in the pipeline — remains **OPEN** and requires a rerun.

---

## B. Files changed

| File | Change |
|---|---|
| `scripts/b3b0_phobert_input_probe.py` | `download_model()` removed; `--vncorenlp-dir` / `--vncorenlp-revision` / `--vncorenlp-hashes`; SHA-256 verification; absolute path resolution before any dependency; cwd diagnostics; `--revision` required with `--allow-floating-revision` escape |
| `unmark/alignment/contracts.py` | `SegmenterContract` gains `jar_name` and `resource_hashes`; `pinned` documented as verification-only |
| `tests/test_alignment_contracts.py` | +24 repair tests, including a simulated cwd mutation |
| `.gitignore` | `.vncorenlp/` |
| `docs/spec/decisions.md` | D-B3B0-003 (implementation repair); D-B3B0-001 left OPEN |
| `README.md` | Corrected Colab command and the two reproducibility rules |

`unmark-proposal.md` **unchanged**.

---

## C. First Colab run findings

**Bug 1 — unpinned VnCoreNLP download.** The script called
`py_vncorenlp.download_model(save_dir=…)` before constructing the segmenter, so the
segmentation model was whatever upstream served that day. The researcher had already
provisioned a pinned VnCoreNLP v1.2 checkout in `<repo>/.vncorenlp/` with recorded Git
revision and SHA-256 for `VnCoreNLP-1.2.jar`, `models/wordsegmenter/vi-vocab` and
`models/wordsegmenter/wordsegmenter.rdr`. The probe ignored it and fetched its own.

Audit 006 had flagged the unpinned download as non-blocking issue N1. That was too
lenient: for a probe whose output feeds a preprocessing decision, an unpinned segmenter is
disqualifying, not a caveat.

**Bug 2 — relative output-root drift.** `py_vncorenlp.VnCoreNLP(save_dir=…)` `chdir()`s
into its resource directory. The run directory was built from a *relative*
`--output-root` **after** that call, so artifacts landed in
`.vncorenlp/results/b3b0/<run_id>/` instead of `<repo>/results/b3b0/<run_id>/`.

Neither was catchable by the local mock tests, which never invoke the real dependency. The
repair adds a test that simulates the cwd mutation directly.

---

## D. Invalidated run

The first Colab run is an **INVALID PROBE RUN for scientific decision-making**, because
the segmentation resource provenance was not guaranteed. Any per-path measurement that
depends on segmentation is unattributable; the `RAW_BASE` rows are unaffected by bug 1 but
the run as a whole is not a basis for choosing a preprocessing policy.

**No conclusion is drawn from it.** D-B3B0-001 stays OPEN.

The researcher's artifacts under `.vncorenlp/results/b3b0/` were **not deleted or
modified** — they are their run to keep. `.vncorenlp/` is now git-ignored, so neither the
runtime directory nor those stray artifacts can be committed; verified for both
`.vncorenlp/` and `.vncorenlp/results/b3b0/x/report.md`.

---

## E. VnCoreNLP pinning contract

```bash
--vncorenlp-dir       <path>   # ALREADY-PROVISIONED checkout; relative resolves from repo root
--vncorenlp-revision  <sha>    # externally supplied revision, recorded
--vncorenlp-hashes    <json>   # {"revision": ..., "files": {relpath: sha256}}
```

The probe:

1. never downloads — `download_model()` is absent, and an AST test asserts no call to it
   can reappear;
2. requires the directory to exist, and does **not** create it;
3. requires `VnCoreNLP-*.jar`, `models/wordsegmenter/vi-vocab` and
   `models/wordsegmenter/wordsegmenter.rdr`, naming any that are missing;
4. computes SHA-256 for every required file and records them;
5. **refuses to load** on any mismatch against supplied hashes;
6. records `pinned=true` **only** when every observed file was verified against a supplied
   hash. Partial verification, or no hashes at all, yields `pinned=false` with the reason.

Provenance is recorded on **every** return path, including import and initialisation
failure: what the checkout contains is a separate fact from whether the library loaded.
No absolute Colab path is hard-coded.

---

## F. Output-path contract

At process start, before any third-party code runs:

* `REPO_ROOT` is derived from `__file__`;
* `--output-root` is resolved absolutely — a relative value resolves against the
  **repository root**, never the cwd, so the default `results/b3b0` always means
  `<repo>/results/b3b0`;
* `--vncorenlp-dir` and `--vncorenlp-hashes` are resolved absolutely the same way.

Recorded in `config.json` and `environment.json`: `repository_root`,
`resolved_output_root`, `resolved_vncorenlp_dir`, `cwd_at_start`,
`cwd_after_segmenter_initialization`, `cwd_changed_by_dependency`. A cwd change is printed
and called out in the report rather than silently absorbed.

No `chdir()` is used anywhere — an AST test forbids it. A successful run produces exactly
one directory `<repo>/results/b3b0/<run_id>/` containing `config.json`,
`environment.json`, `cases.jsonl`, `summary.json`, `report.md`.

---

## G. PhoBERT revision contract

`--revision` is **required**. Without it the probe exits 2 and explains that a floating
revision makes the result unattributable. `--allow-floating-revision` runs anyway but
stamps `revision_pinned: false` and `scientifically_usable: false`, and the report opens
with a "NOT SCIENTIFICALLY USABLE" warning.

`--checkpoint` keeps its convenience default `vinai/phobert-base`. That is still **not a
lock** — D-B3B0-002 records that the backbone checkpoint is unpinned in the proposal, and
this task did not change §6.1.

`scientifically_usable` is `revision_pinned AND segmenter.pinned`, so a run is only marked
usable when both the tokenizer and the segmenter are provenance-verified.

---

## H. Probe network contract

| Operation | Allowed |
|---|---|
| Hugging Face tokenizer via `from_pretrained(revision=…)` | **yes** — expected for B3B-0 |
| PhoBERT model weights | **no** — AST test asserts no `*Model.from_pretrained` |
| VnCoreNLP download | **no** — AST test asserts no `download_model` call |
| Anything else | not performed |

Network APIs are not banned globally: the tokenizer legitimately needs Hugging Face
access. The ban is specific to segmenter resource fetching.

---

## I. Proposal / decision-log consistency

| Surface | State |
|---|---|
| `unmark-proposal.md` | **Unchanged.** This was an infrastructure repair; no scientific semantics were touched. |
| D-B3B0-001 (segmentation vs `T(b(x))`) | **Still OPEN.** Not resolved, not narrowed. |
| D-B3B0-002 (backbone checkpoint unpinned) | Still OPEN. |
| D-B3B0-003 (this repair) | New: `IMPLEMENTATION REPAIR`, with both observed bugs, the invalidation, and the resolution. States "Proposal source updated: NO". |
| Audit 006 | Left intact as a point-in-time record; its N1 is superseded by this audit's stricter treatment. |

---

## J. Test evidence

| Command | Result |
|---|---|
| `.venv/bin/python -m pytest -q -p no:cacheprovider` | **1452 passed in 6.27s** |
| `tests/test_alignment_contracts.py` | 70 passed (46 + 24 repair tests) |
| All previous suites (G−1, B1A, B2, B3A, B3B-0) | green |
| `pip list` | 7 packages, unchanged — nothing installed |
| local probe without `--revision` | exit 2, "--revision is required" |
| local probe with `--revision` | refuses with Colab instructions (no transformers) |

The fifteen required proofs, by test:

| # | Proof | Test |
|---|---|---|
| 1 | never calls `download_model` | `test_probe_never_calls_download_model`, `test_probe_source_mentions_download_model_only_to_forbid_it` |
| 2 | directory must already exist | `test_missing_segmenter_directory_is_reported_not_created` |
| 3–4 | required files must exist, failing clearly | `test_missing_required_files_fail_clearly`, `test_missing_jar_fails_clearly` |
| 5 | relative `--vncorenlp-dir` resolves from repo root | `test_probe_resolves_paths_before_loading_the_segmenter` |
| 6 | relative output root resolves from repo root | `test_relative_output_root_resolves_against_the_repository_root` |
| 7 | simulated cwd mutation does not move artifacts | `test_output_artifacts_survive_a_dependency_changing_the_cwd` |
| 8 | all five artifacts land in the run directory | same test |
| 9 | SHA-256 recorded | `test_resource_hashes_are_recorded` |
| 10 | pinned not fabricated | `test_pinned_is_false_when_no_hashes_were_supplied`, `…_when_some_files_were_unverified`, `test_hash_mismatch_refuses_to_load`, `test_verified_checkout_is_marked_pinned_even_if_the_library_is_absent` |
| 11 | explicit revision supported | `test_probe_still_supports_an_explicit_revision` |
| 12 | missing revision fails closed | `test_probe_fails_closed_without_a_revision` |
| 13 | no model loading introduced | `test_no_model_loading_call_was_introduced_by_the_repair` |
| 14 | tests stay dependency-free | `test_probe_imports_cleanly_without_any_ml_dependency` |
| 15 | existing tests green | full suite |

Plus `test_probe_does_not_use_chdir_to_fix_paths` and
`test_vncorenlp_runtime_directory_is_gitignored`.

---

## K. Colab rerun command

```bash
cd unmark-draft
pip install "transformers==4.57.6"
pip install py_vncorenlp
export HF_HOME="$PWD/.hf-cache"

python scripts/fetch_vietnamese_syllable_inventory.py     # B3A eligibility

# .vncorenlp/ must already contain the pinned v1.2 checkout — the probe never
# downloads it. Supply the recorded hashes so the run can be marked pinned.
python scripts/b3b0_phobert_input_probe.py \
    --checkpoint vinai/phobert-base \
    --revision <FULL_TOKENIZER_SHA> \
    --vncorenlp-dir .vncorenlp \
    --vncorenlp-revision <VNCORENLP_GIT_SHA> \
    --vncorenlp-hashes <path/to/hashes.json>
```

`hashes.json`:

```json
{
  "revision": "<VNCORENLP_GIT_SHA>",
  "files": {
    "VnCoreNLP-1.2.jar": "<sha256>",
    "models/wordsegmenter/vi-vocab": "<sha256>",
    "models/wordsegmenter/wordsegmenter.rdr": "<sha256>"
  }
}
```

Artifacts land in `<repo>/results/b3b0/<run_id>/`. Check `scientifically_usable: true` in
`config.json` before reading any number: it is false unless both the tokenizer revision
and the segmenter resources were verified.

---

## L. Blocking issues

`None` for the repair. D-B3B0-001 blocks **B3B**, unchanged and by design.

---

## M. Non-blocking issues

```
ID: N1  The hashes file is caller-supplied, not committed
    The project has no committed VnCoreNLP manifest yet, unlike the B3A syllable
    inventory. Consider configs/linguistics/vncorenlp_v1.2.yaml so the pin lives in git
    rather than in a Colab-side file.

ID: N2  Only the first matching VnCoreNLP-*.jar is used
    A directory containing several versioned jars would silently use the lexicographically
    first. Recorded as jar_name, so it is visible, but not disambiguated.

ID: N3  --allow-floating-revision exists
    Deliberate: exploratory runs remain possible. Its artifacts are stamped
    scientifically_usable=false, but a reader skimming a report could still miss it.

ID: N4  PRESEGMENTED_DATASET still unprobed
    Unchanged from audit 006; no dataset is pinned yet.

ID: N5  unmark-proposal.pdf
    Still stale from the earlier 4.2 and 5.3 edits. Unchanged by this task.
```

---

## N. Git state

* **Branch:** `main`
* **HEAD:** `7b17322` — the researcher's commit of the original B3B-0 probe, made during
  this session; the reflog shows no commit from this repair
* **Staged:** none (`git diff --cached --stat` empty)
* **Unstaged:** `.gitignore`, `README.md`, `docs/spec/decisions.md`,
  `scripts/b3b0_phobert_input_probe.py`, `tests/test_alignment_contracts.py`,
  `unmark/alignment/contracts.py`
* **Untracked:** `docs/audits/007-b3b0-colab-probe-repair.md`
* **Ignored by design:** `.vncorenlp/`, including the invalid run's stray artifacts

No `add`, `commit`, `push`, `tag`, `stash`, `reset`, `checkout` or `restore` was run.

```text
AUDIT FILE WRITTEN: docs/audits/007-b3b0-colab-probe-repair.md
OTHER AUDIT MODIFICATIONS MADE: NO
COMMIT CREATED: NO
```
