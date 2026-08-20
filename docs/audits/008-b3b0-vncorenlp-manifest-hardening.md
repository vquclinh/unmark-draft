# Audit 008 — B3B-0 VnCoreNLP manifest hardening

| | |
|---|---|
| **Audit id** | 008 |
| **Date (UTC)** | 2026-08-19 |
| **Scope** | Reproducibility hardening only: committed VnCoreNLP pin, exact jar selection, provenance conflict policy |
| **Repository state** | `HEAD = 7b17322`; this hardening and audit 007 uncommitted |
| **Predecessor** | [Audit 007](007-b3b0-colab-probe-repair.md) — closes its N1 and N2 |
| **Phase** | Phase 0 / B3B-0 |

---

## A. Verdict

**IMPLEMENTATION PARTIAL — RESEARCHER PROVENANCE REQUIRED**

Every mechanism asked for is implemented and tested: the VnCoreNLP dependency is now
represented by a committed manifest, the jar is named rather than discovered, extra jars
can never be substituted, `pinned=true` requires full verification, and a disagreement
between the manifest and any CLI-supplied provenance fails closed. 1480 tests pass offline.

**But the pin is not complete, and cannot be completed here.** The exact Git revision and
the three SHA-256 digests exist only in your Colab provisioning cells. They are not present
anywhere in this repository, not recoverable from audits 006 or 007, and not derivable
locally — `.vncorenlp/` is a git-ignored Colab-side runtime directory and is absent on this
machine. The instruction was explicit that no SHA may be invented, so none was. The
manifest is committed with `status: AWAITING_RESEARCHER_PROVENANCE` and
`PENDING_RESEARCHER_PROVENANCE` in every digest slot, and the loader **refuses** it.

The verdict is therefore not `PASS — READY FOR SCIENTIFIC COLAB RERUN`: the rerun cannot
be scientifically usable until you paste the Cell 08/09 values into the manifest. That is
a one-file edit; nothing else is outstanding.

---

## B. Files changed

**New**

| File | Purpose |
|---|---|
| `configs/linguistics/vncorenlp_v1.2.json` | Committed pin — schema, source, repository, release tag, revision, required jar, per-file SHA-256. **Digests pending.** |
| `docs/audits/008-b3b0-vncorenlp-manifest-hardening.md` | This audit |

**Modified**

| File | Change |
|---|---|
| `scripts/b3b0_phobert_input_probe.py` | `load_vncorenlp_manifest` with schema validation and placeholder rejection; `reconcile_provenance` conflict detection; `--vncorenlp-manifest`; exact-jar selection replacing the glob |
| `unmark/alignment/contracts.py` | `SegmenterContract` gains `required_jar`, `other_jars_present`, `manifest_path`; `jar_name` now means "the jar actually loaded" |
| `tests/test_alignment_contracts.py` | +28 hardening tests; two earlier tests updated to the exact-jar contract |
| `.gitignore` | `.probe_*` |
| `docs/spec/decisions.md` | D-B3B0-004 |
| `README.md` | Manifest-based invocation and the incomplete-pin warning |

`unmark-proposal.md` **unchanged**.

---

## C. VnCoreNLP committed manifest

`configs/linguistics/vncorenlp_v1.2.json`:

```json
{
  "schema_version": "vncorenlp-pin-v1",
  "status": "AWAITING_RESEARCHER_PROVENANCE",
  "source": "VnCoreNLP",
  "source_repository": "https://github.com/vncorenlp/VnCoreNLP",
  "release_tag": "v1.2",
  "revision": "PENDING_RESEARCHER_PROVENANCE",
  "required_jar": "VnCoreNLP-1.2.jar",
  "files": {
    "VnCoreNLP-1.2.jar": "PENDING_RESEARCHER_PROVENANCE",
    "models/wordsegmenter/vi-vocab": "PENDING_RESEARCHER_PROVENANCE",
    "models/wordsegmenter/wordsegmenter.rdr": "PENDING_RESEARCHER_PROVENANCE"
  }
}
```

**Why the placeholders.** I searched the whole repository for a 64-hex digest: the only one
present is the B3A syllable inventory's. Audits 006 and 007 describe the VnCoreNLP hashes
as *recorded* but never quote them. `.vncorenlp/` does not exist locally, so nothing could
be hashed. Fabricating a digest would produce a pin that verifies nothing while looking
authoritative — the exact failure this task exists to prevent.

**Validation.** The loader requires `schema_version`, `source`, `source_repository`,
`release_tag`, `revision`, `required_jar` and `files`; requires `required_jar` to appear in
`files`; requires all three resources to be listed; and rejects any digest that is not
64 hex characters, plus a revision that is neither a 40- nor 64-character hash. A test
also asserts that no plausible-looking fake digest has been committed.

**Semantics.** Not a dataset, not a model checkpoint — a reproducibility pin for the
word-segmentation dependency. Changing the release tag, revision, jar, vocabulary, RDR
model or any digest is an **experiment dependency change** and must be recorded in
`docs/spec/decisions.md` (D-B3B0-004).

---

## D. Exact jar contract

The glob-and-take-first behaviour is gone. The jar is read from the pin's `required_jar`
and opened by exact name.

| Situation | Behaviour |
|---|---|
| required jar present | loaded; `required_jar` and `jar_name` both recorded |
| required jar absent | **fail closed**, naming the missing jar |
| other `VnCoreNLP-*.jar` present | recorded in `other_jars_present`, **never substituted** |
| required jar absent, others present | fail closed, and the note says the others are "NOT substituted: the pin names exactly one" |

A test plants `VnCoreNLP-1.1.jar` — which sorts *before* the required one, so the old
globbing code would have selected it — and asserts the loader still uses
`VnCoreNLP-1.2.jar` and hashes only that.

`jar_name` now means "the jar actually loaded" and is `None` when loading was refused;
`required_jar` is what the pin names and is always recorded. The distinction matters: a
verified checkout whose library failed to import reports `required_jar` set, `jar_name`
null and `pinned=true`, because verification is a statement about files.

---

## E. Pin verification contract

`segmenter.pinned = true` requires **all** of:

1. a manifest was supplied and parsed;
2. no CLI-supplied revision or hashes contradict it;
3. the required jar exists at its exact name;
4. all three required resources exist;
5. every observed file's SHA-256 matches the manifest exactly.

Partial verification never yields `pinned=true`: no manifest, a manifest covering only some
files, or an unverified extra file all yield `pinned=false` with the reason recorded. Any
mismatch **refuses to load** rather than proceeding with a warning.

`scientifically_usable = revision_pinned AND segmenter.pinned`, so with the manifest
incomplete a run is marked unusable today — which is the intended fail-closed behaviour.

---

## F. CLI contract

Canonical scientific invocation:

```bash
python scripts/b3b0_phobert_input_probe.py \
    --checkpoint vinai/phobert-base \
    --revision <FULL_TOKENIZER_SHA> \
    --vncorenlp-dir .vncorenlp \
    --vncorenlp-manifest configs/linguistics/vncorenlp_v1.2.json
```

`--vncorenlp-manifest` defaults to the committed pin, so the flag can be omitted.
`--vncorenlp-revision` and `--vncorenlp-hashes` are retained for compatibility but are
**not** an override path: any disagreement with the manifest raises, naming the conflicting
field. Silent precedence would let a stale CLI value quietly replace the repository's pin.

---

## G. Scratch provenance policy

`.probe_phobert_revision`, `.probe_vncorenlp_revision` and `.probe_vncorenlp_hashes.txt`
are **notebook cell-to-cell state, not scientific configuration**. The probe never reads
them — asserted by test — and `.probe_*` is now git-ignored.

Provenance now comes from exactly two places: the committed manifest for VnCoreNLP, and an
explicit `--revision` flag for the tokenizer. No researcher files in any other runtime were
deleted or modified.

---

## H. Proposal / decision-log consistency

| Surface | State |
|---|---|
| `unmark-proposal.md` | **Unchanged.** Infrastructure only; no scientific semantics touched. |
| D-B3B0-001 (segmentation vs `T(b(x))`) | **Still OPEN.** |
| D-B3B0-002 (backbone checkpoint unpinned) | **Still OPEN.** |
| D-B3B0-003 (probe repair) | Unchanged. |
| D-B3B0-004 (this hardening) | New: original state, final state, reason, the blocked-on values, conflict policy, scratch policy, dependency-change policy. "Proposal updated: NO". |

---

## I. Test evidence

| Command | Result |
|---|---|
| `.venv/bin/python -m pytest -q -p no:cacheprovider` | **1480 passed in 6.49s** |
| `tests/test_alignment_contracts.py` | 98 passed |
| All previous suites | green |
| `pip list` | 7 packages, unchanged — nothing installed |

| # | Proof | Test |
|---|---|---|
| 1 | manifest parses | `test_committed_manifest_exists_and_parses` |
| 2 | exact jar named | `test_committed_manifest_names_the_exact_required_jar` |
| 3 | three files required | `test_committed_manifest_requires_exactly_the_three_resources` |
| 4 | matching hashes → pinned | `test_matching_hashes_yield_pinned_true` |
| 5 | any mismatch → fail closed | `test_any_hash_mismatch_fails_closed` (×3, parametrised per file) |
| 6 | missing jar → fail closed | `test_missing_required_jar_fails_closed` |
| 7 | extra jar cannot change selection | `test_extra_jar_cannot_change_the_selected_jar`, `test_extra_jars_are_reported_when_the_required_one_is_absent` |
| 8 | revision/hash conflict → fail closed | `test_cli_revision_conflicting_with_the_manifest_fails_closed`, `test_hashes_file_conflicting_with_the_manifest_fails_closed` |
| 9 | metadata records jar and hashes | `test_run_metadata_records_required_jar_and_hashes` |
| 10 | no downloader | `test_no_downloader_was_reintroduced_by_the_hardening` |
| 11 | manifest is the default | `test_committed_manifest_is_the_default_cli_provenance_source` |
| 12 | `.probe_*` not required | `test_probe_does_not_read_notebook_scratch_files`, `test_notebook_scratch_files_are_gitignored` |
| 13 | no model loading | `test_no_model_loading_was_introduced_by_the_hardening` |
| 14 | previous tests green | full suite |

Plus schema-validation tests (missing key, missing resource, `required_jar` absent from
`files`, invalid JSON), the incomplete-pin rejection, and a guard that no fake digest is
committed.

---

## J. Final Colab rerun command

**Step 0 — complete the pin.** In Colab, from the provisioned checkout:

```bash
cd .vncorenlp
git rev-parse HEAD
sha256sum VnCoreNLP-1.2.jar models/wordsegmenter/vi-vocab models/wordsegmenter/wordsegmenter.rdr
```

Paste those four values into `configs/linguistics/vncorenlp_v1.2.json`, replacing every
`PENDING_RESEARCHER_PROVENANCE`, and set `"status": "PINNED"`.

**Step 1 — run.**

```bash
cd unmark-draft
pip install "transformers==4.57.6"
pip install py_vncorenlp
export HF_HOME="$PWD/.hf-cache"

python scripts/fetch_vietnamese_syllable_inventory.py

python scripts/b3b0_phobert_input_probe.py \
    --checkpoint vinai/phobert-base \
    --revision <FULL_TOKENIZER_SHA> \
    --vncorenlp-dir .vncorenlp \
    --vncorenlp-manifest configs/linguistics/vncorenlp_v1.2.json
```

Artifacts land in `<repo>/results/b3b0/<run_id>/`. Before reading any number, check
`config.json` for `scientifically_usable: true` — it is false unless both the tokenizer
revision and every VnCoreNLP resource verified.

---

## K. Blocking issues

```
ID: B1  Committed pin is incomplete
File:   configs/linguistics/vncorenlp_v1.2.json
Problem: revision and all three SHA-256 values are placeholders.
Why:    They exist only in the Colab provisioning cells. Not in this repository, not in
        audits 006/007, not derivable locally (.vncorenlp/ is Colab-side and git-ignored).
        Inventing one was explicitly forbidden and would defeat the pin.
Effect: The probe refuses the manifest, so no run can be scientifically usable.
Fix:    Paste the Cell 08/09 values as in section J. One file, four values.
```

D-B3B0-001 also still blocks B3B, unchanged and by design.

---

## L. Non-blocking issues

```
ID: N1  Manifest revision is not cross-checked against the checkout
    Nothing verifies that .vncorenlp/ is actually at the manifest's Git revision; only the
    file digests are checked. In practice the digests pin the content, which is what
    matters, but a `git rev-parse` comparison would close the gap.

ID: N2  --vncorenlp-hashes retained
    Kept for compatibility. It cannot override the manifest (conflicts raise), but it is a
    second provenance path that could be removed once nothing uses it.

ID: N3  status field is advisory
    "AWAITING_RESEARCHER_PROVENANCE" is not what triggers rejection — the placeholder
    digests are. A manifest with real digests but a stale status string would still load.
    Harmless, but the two could drift.

ID: N4  PRESEGMENTED_DATASET still unprobed
    Unchanged from audits 006/007; no dataset is pinned yet.

ID: N5  unmark-proposal.pdf
    Still stale from the earlier §4.2 and §5.3 edits. Unchanged by this task.
```

---

## M. Git state

* **Branch:** `main`
* **HEAD:** `7b17322` "prepare B3B0 PhoBERT input contract probe" — the researcher's
  commit; the reflog shows no commit from this session
* **Staged:** none (`git diff --cached --stat` empty)
* **Unstaged:** `.gitignore`, `README.md`, `docs/spec/decisions.md`,
  `scripts/b3b0_phobert_input_probe.py`, `tests/test_alignment_contracts.py`,
  `unmark/alignment/contracts.py`
* **Untracked:** `configs/linguistics/vncorenlp_v1.2.json`,
  `docs/audits/007-b3b0-colab-probe-repair.md`,
  `docs/audits/008-b3b0-vncorenlp-manifest-hardening.md`
* **Ignored by design:** `.vncorenlp/`, `.probe_*`

No `add`, `commit`, `push`, `tag`, `stash`, `reset`, `checkout` or `restore` was run.

```text
AUDIT FILE WRITTEN: docs/audits/008-b3b0-vncorenlp-manifest-hardening.md
OTHER AUDIT MODIFICATIONS MADE: NO
COMMIT CREATED: NO
```
