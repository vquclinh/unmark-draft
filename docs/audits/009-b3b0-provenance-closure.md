# Audit 009 — B3B-0 provenance closure

| | |
|---|---|
| **Audit id** | 009 |
| **Date (UTC)** | 2026-08-19 |
| **Scope** | Provenance closure only: complete the VnCoreNLP pin, verify checkout revision, tighten `scientifically_usable` |
| **Repository state** | `HEAD = 7b17322`; audits 007–009 and this work uncommitted |
| **Predecessors** | [006](006-b3b0-phobert-input-contract.md), [007](007-b3b0-colab-probe-repair.md), [008](008-b3b0-vncorenlp-manifest-hardening.md) |
| **Phase** | Phase 0 / B3B-0 |

---

## A. Verdict

**PASS — READY FOR SCIENTIFIC COLAB RERUN**

Audit 008's researcher-provenance blocker is **closed**: the committed manifest now carries
the exact release tag, Git revision and three SHA-256 digests extracted from the
provisioned checkout, with `status: PINNED` and no placeholder remaining. Audit 008's N1
revision-verification gap is **closed**: the probe reads the checkout's `git rev-parse HEAD`
and refuses to load on any disagreement with the pin, and a checkout without `.git` has its
digests verified but is never marked pinned. `scientifically_usable` is now the conjunction
of tokenizer-revision pinning, checkout-revision verification and full digest verification.
1491 tests pass offline with no transformers, torch, Java, VnCoreNLP or network.

Explicitly, and unchanged by this task:

* **D-B3B0-001 remains OPEN** — whether PhoBERT's word segmentation belongs in the pipeline.
* **D-B3B0-002 remains OPEN** — the backbone checkpoint is named but not pinned.
* **No preprocessing policy has been selected.** This work makes the evidence trustworthy;
  it does not produce or interpret any.
* **No scientific conclusion is drawn from the invalid first Colab run** (audit 007 §D).

---

## B. Files changed

| File | Change |
|---|---|
| `configs/linguistics/vncorenlp_v1.2.json` | Placeholders replaced with the exact provenance; `status: PINNED`; repository URL corrected to the `.git` form supplied |
| `scripts/b3b0_phobert_input_probe.py` | `git_head_revision`, `git_tags_at_head`; revision verification with fail-closed refusal; `pinned` is now hashes ∧ revision; provenance block in `config.json` and the report |
| `unmark/alignment/contracts.py` | `SegmenterContract` gains `manifest_revision`, `observed_revision`, `revision_verified`, `observed_tags_at_head`, `expected_hashes`, `hashes_verified` |
| `tests/test_alignment_contracts.py` | +11 closure tests; 6 audit-008-era tests updated to the completed pin |
| `docs/spec/decisions.md` | D-B3B0-005; D-B3B0-004 marked CLOSED |
| `README.md` | Canonical rerun procedure; incomplete-pin warning replaced |
| `docs/audits/009-b3b0-provenance-closure.md` | This audit |

`unmark-proposal.md` **unchanged**.

---

## C. Researcher-provided provenance

Used exactly as supplied; nothing was regenerated, shortened or replaced.

| Field | Value |
|---|---|
| source repository | `https://github.com/vncorenlp/VnCoreNLP.git` |
| release tag at HEAD | `v1.2` |
| revision | `62bbc58fe5d113c898eae112656be97dcf50b3a0` |
| required jar | `VnCoreNLP-1.2.jar` |
| `VnCoreNLP-1.2.jar` | `9e2811cdbc2ddfc71d04be5dc36e185c88dcd1ad4d5d69e4ff2e1369dccf7793` |
| `models/wordsegmenter/vi-vocab` | `0a47c5b55bbce163029d37730a67b9479740388695c29c106c112b815613eaa5` |
| `models/wordsegmenter/wordsegmenter.rdr` | `9e62f96bd93e37a24f364238e8d8ae986fa5dad6dbc9f4eae622ab3651b7fa06` |

Obtained by the researcher with `git rev-parse HEAD`, `git tag --points-at HEAD` and
SHA-256 over the three required resources. The values are hard-coded in a test, so a silent
edit to the manifest fails the suite.

---

## D. Final committed VnCoreNLP manifest

`configs/linguistics/vncorenlp_v1.2.json`, `schema_version: vncorenlp-pin-v1`,
`status: PINNED`. It carries the source, repository, release tag, exact revision, the
exact `required_jar`, and one SHA-256 per required resource. No floating URL and no
"current"/"latest" semantics: it is a fixed pin.

The loader still rejects any manifest missing a required key, missing a required resource,
naming a `required_jar` absent from `files`, carrying a non-64-hex digest, or containing
invalid JSON. Changing the release tag, revision, jar, vocabulary, RDR model or any digest
is an **experiment dependency change** and must be recorded in `docs/spec/decisions.md`
(D-B3B0-005).

---

## E. Git revision verification

Closes audit 008 N1.

| Situation | Behaviour |
|---|---|
| `HEAD` == pinned revision | `revision_verified: true` |
| `HEAD` != pinned revision | **FAIL CLOSED** — refuses to load, naming both revisions |
| `.git` absent or `git` unusable | `observed_revision: null`, `revision_verified: false`, digests still verified, `pinned: false` |

`git rev-parse HEAD` runs as a **local subprocess** — no network. Verification is never
fabricated: absence of metadata yields `false`, never an assumed match.

`git tag --points-at HEAD` is recorded as `observed_tags_at_head`, **diagnostic only**. A
test plants the correct `v1.2` tag on a checkout whose revision does *not* match the pin
and asserts the load is still refused — tag text alone is never sufficient. Conversely,
tags being unavailable does not invalidate a revision that does match.

---

## F. Resource hash verification

Every required file is hashed at run time and compared against the pin.
`hashes_verified` is true only when **every** observed digest was checked against a pinned
one and matched; a file with no pinned digest leaves it false. Any mismatch **refuses to
load**, naming the offending files.

Exact-jar selection from audit 008 is retained: the jar is read from `required_jar`, extra
`VnCoreNLP-*.jar` files are recorded in `other_jars_present` and never substituted, and a
missing required jar fails closed.

---

## G. `scientifically_usable` contract

```text
scientifically_usable = revision_pinned(tokenizer) AND segmenter.pinned
segmenter.pinned      = hashes_verified AND revision_verified
```

So a run is scientifically usable only when **all** of these hold: an explicit PhoBERT
`--revision` was supplied; the manifest was valid and complete; the exact required jar was
selected; the checkout's Git revision equalled the pinned revision; all three digests
matched; and no downloader was used.

Fail-closed, none downgraded to a warning: hash mismatch, revision mismatch, missing
required resource, incomplete manifest. Missing `.git` does not refuse the load but does
make the run not scientifically usable, which is the behaviour requested for this probe.

---

## H. Run artifact provenance

`config.json` gains a `vncorenlp_provenance` block carrying `source`,
`source_repository` (via the manifest path), `release_tag`, `manifest_revision`,
`observed_revision`, `revision_verified`, `observed_tags_at_head`, `required_jar`,
`jar_name` (the jar actually loaded), `other_jars_present`, `expected_hashes`,
`resource_hashes`, `hashes_verified`, `manifest_path` and `pinned`. `environment.json`
retains the same segmenter block.

The report renders a per-resource **expected vs observed vs match** table, so the
verification can be reconstructed from the artifact alone.

Audit 007's path diagnostics are retained unchanged: `cwd_at_start`,
`cwd_after_segmenter_initialization`, `cwd_changed_by_dependency`, `repository_root`,
`resolved_output_root`, `resolved_vncorenlp_dir`.

---

## I. Scratch-file independence

The probe reads none of `.probe_phobert_revision`, `.probe_vncorenlp_revision`,
`.probe_vncorenlp_hashes.txt` or `/content/vncorenlp-provenance.json` — asserted by a test
that greps the source for all four, including the `/content` path.

Scientific configuration comes from exactly two places: the committed manifest for
VnCoreNLP, and an explicit `--revision` flag for the tokenizer. `.probe_*` remains
git-ignored. No files in any other runtime were deleted or modified.

---

## J. Proposal / decision-log consistency

| Surface | State |
|---|---|
| `unmark-proposal.md` | **Unchanged.** Provenance closure is not a method change. |
| D-B3B0-001 | **OPEN** — segmentation vs `T(b(x))`. Not resolved, not narrowed. |
| D-B3B0-002 | **OPEN** — backbone checkpoint not pinned. |
| D-B3B0-003 | Unchanged (probe repair). |
| D-B3B0-004 | Marked **CLOSED** by D-B3B0-005; prior status preserved in the entry. |
| D-B3B0-005 | New: original state, the exact researcher provenance, final implementation, reason, affected areas, and the two items that stay open. "Proposal updated: NO". |

---

## K. Test evidence

| Command | Result |
|---|---|
| `.venv/bin/python -m pytest -q -p no:cacheprovider` | **1491 passed in 6.53s** |
| `tests/test_alignment_contracts.py` | 111 passed |
| All previous suites (G−1, B1A, B2, B3A) | green |
| `pip list` | 7 packages, unchanged — nothing installed |

Local tests require no transformers, torch, Java, VnCoreNLP or network. Revision tests use
**temporary Git repositories created offline** by `git init`/`commit`.

| # | Proof | Test |
|---|---|---|
| 1 | manifest status PINNED | `test_committed_manifest_is_pinned_and_loads` |
| 2 | no placeholder remains | `test_no_provenance_placeholder_remains` |
| 3 | exact revision | `test_committed_manifest_records_the_exact_researcher_revision` |
| 4 | exact three digests | `test_committed_manifest_records_the_exact_researcher_hashes` |
| 5 | required jar | `test_committed_manifest_names_the_exact_required_jar` |
| 6 | matching revision → verified | `test_matching_hashes_and_revision_yield_pinned_true` |
| 7 | wrong revision → fail closed | `test_wrong_git_revision_fails_closed` |
| 8 | no `.git` → not pinned, hashes still checked | `test_unavailable_git_metadata_still_checks_hashes_but_is_not_pinned` |
| 9 | matching hashes → verified | `test_matching_hashes_and_revision_yield_pinned_true` |
| 10 | any hash mismatch → fail closed | `test_any_hash_mismatch_fails_closed` (×3), `test_revision_match_alone_is_not_enough_without_hash_match` |
| 11 | extra jars never alter selection | `test_extra_jar_cannot_change_the_selected_jar` |
| 12 | observed revision recorded | `test_run_metadata_records_full_provenance` |
| 13 | expected + observed hashes recorded | `test_expected_and_observed_hashes_are_both_recorded` |
| 14 | manifest path recorded | `test_run_metadata_records_full_provenance` |
| 15 | scratch files never read | `test_probe_never_reads_the_colab_provenance_scratch_file` |
| 16 | downloader absent | `test_no_downloader_was_reintroduced_by_the_hardening` |
| 17 | `--revision` required | `test_probe_fails_closed_without_a_revision` |
| 18 | no model loading | `test_no_model_loading_was_introduced_by_the_hardening` |
| 19 | output root survives cwd mutation | `test_output_artifacts_survive_a_dependency_changing_the_cwd` |
| 20 | previous tests green | full suite |

Plus `test_observed_tags_at_head_are_recorded_as_a_diagnostic` (a correct tag does not
rescue a revision mismatch) and `test_git_verification_uses_a_local_subprocess_not_the_network`.

---

## L. Final Colab rerun command

```bash
cd unmark-draft
pip install "transformers==4.57.6"
pip install py_vncorenlp
export HF_HOME="$PWD/.hf-cache"

python scripts/fetch_vietnamese_syllable_inventory.py      # B3A eligibility

# .vncorenlp/ must be the VnCoreNLP checkout at 62bbc58fe5d113c898eae112656be97dcf50b3a0.
# The probe verifies HEAD and every resource digest; it never downloads.
python scripts/b3b0_phobert_input_probe.py \
    --checkpoint vinai/phobert-base \
    --revision <FULL_TOKENIZER_SHA> \
    --vncorenlp-dir .vncorenlp \
    --vncorenlp-manifest configs/linguistics/vncorenlp_v1.2.json
```

Artifacts land in `<repo>/results/b3b0/<run_id>/`. **Read no number until
`config.json` shows `scientifically_usable: true`.**

`--vncorenlp-manifest` defaults to the committed pin and may be omitted. `--revision` for
the tokenizer is still required and still unpinned in the proposal (D-B3B0-002), so the
value used is a *probe* revision, not a locked backbone.

---

## M. Blocking issues

`None`.

Audit 008's blocker B1 is closed. D-B3B0-001 continues to block **B3B**, by design and
unchanged.

---

## N. Non-blocking issues

```
ID: N1  status field remains advisory
    Rejection is driven by placeholder digests, not by the `status` string, so a manifest
    with real digests and a stale status would still load. Carried over from audit 008 N3.

ID: N2  --vncorenlp-hashes retained
    Kept for compatibility. It cannot override the manifest (conflicts raise), but it is a
    second provenance path that could be removed once nothing uses it.

ID: N3  Tokenizer revision is not verified after loading
    --revision is passed to from_pretrained and recorded, but the probe does not read back
    the resolved commit hash to confirm it. The G-1 restorer does exactly that; the same
    check would close the gap for the tokenizer.

ID: N4  PRESEGMENTED_DATASET still unprobed
    Unchanged from audits 006-008; no dataset is pinned yet.

ID: N5  unmark-proposal.pdf
    Still stale from the earlier 4.2 and 5.3 edits. Unchanged by this task.
```

---

## O. Git state

* **Branch:** `main`
* **HEAD:** `7b17322` "prepare B3B0 PhoBERT input contract probe" — the researcher's
  commit; the reflog shows no commit from this session
* **Staged:** none (`git diff --cached --stat` empty)
* **Unstaged:** `.gitignore`, `README.md`, `docs/spec/decisions.md`,
  `scripts/b3b0_phobert_input_probe.py`, `tests/test_alignment_contracts.py`,
  `unmark/alignment/contracts.py`
* **Untracked:** `configs/linguistics/vncorenlp_v1.2.json`,
  `docs/audits/007-b3b0-colab-probe-repair.md`,
  `docs/audits/008-b3b0-vncorenlp-manifest-hardening.md`,
  `docs/audits/009-b3b0-provenance-closure.md`
* **Ignored by design:** `.vncorenlp/`, `.probe_*`

No `add`, `commit`, `push`, `tag`, `stash`, `reset`, `checkout` or `restore` was run.

```text
AUDIT FILE WRITTEN: docs/audits/009-b3b0-provenance-closure.md
OTHER AUDIT MODIFICATIONS MADE: NO
COMMIT CREATED: NO
```
