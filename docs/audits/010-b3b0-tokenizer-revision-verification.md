# Audit 010 — B3B-0 tokenizer revision verification

| | |
|---|---|
| **Audit id** | 010 |
| **Date (UTC)** | 2026-08-19 |
| **Scope** | Closing audit 009 N3: verify the resolved PhoBERT tokenizer revision after loading |
| **Repository state** | `HEAD = 7b17322`; audits 007–010 and this work uncommitted |
| **Predecessors** | [006](006-b3b0-phobert-input-contract.md), [007](007-b3b0-colab-probe-repair.md), [008](008-b3b0-vncorenlp-manifest-hardening.md), [009](009-b3b0-provenance-closure.md) |
| **Phase** | Phase 0 / B3B-0 |

---

## A. Verdict

**PASS — READY FOR SCIENTIFIC COLAB RERUN**

Audit 009 N3 is **closed**. The probe no longer treats a supplied `--revision` as evidence:
after loading, it reads the resolved commit back out of the Hugging Face cache path of the
files the tokenizer actually loaded, and verifies it equals the request. A mismatch aborts
the run; an unrecoverable revision leaves it not scientifically usable. `--revision` must
now be a full 40-character lowercase commit SHA — branches, tags and abbreviated SHAs are
rejected at argument validation. `scientifically_usable` is the conjunction of tokenizer
revision verification and the VnCoreNLP pin. 1533 tests pass offline with no transformers,
torch, Java, VnCoreNLP or network.

Stated explicitly:

* **Audit 009 N3 is closed** — the tokenizer revision is verified after loading, not merely
  supplied.
* **VnCoreNLP provenance closure remains intact** — regression-tested, pin unchanged.
* **D-B3B0-001 remains OPEN** — whether word segmentation belongs in the pipeline.
* **D-B3B0-002 remains OPEN** — the backbone checkpoint is not locked; the probe revision is
  evidence provenance, not a paper lock.
* **No preprocessing policy has been selected.**
* **The first Colab run remains invalid** (audit 007 §D).
* **No scientific conclusion has yet been drawn.**

---

## B. Files changed

| File | Change |
|---|---|
| `scripts/b3b0_phobert_input_probe.py` | `is_full_commit_sha`, `extract_snapshot_revision`, `_candidate_resolved_paths`, `observe_tokenizer_revision`; post-load verification in `describe_tokenizer`; CLI full-SHA validation; fail-closed mismatch (exit 3); `scientifically_usable` retargeted; PhoBERT provenance in artifacts and report |
| `unmark/alignment/contracts.py` | `TokenizerContract`: `revision` → `revision_requested` / `revision_observed` / `revision_verified` / `revision_evidence` / `revision_evidence_source` |
| `tests/test_alignment_contracts.py` | +30 tests; 3 earlier tests updated to the new fields and formula |
| `docs/spec/decisions.md` | D-B3B0-006 |
| `README.md` | Verified-revision rule and the final `scientifically_usable` formula |
| `docs/audits/010-b3b0-tokenizer-revision-verification.md` | This audit |

`unmark-proposal.md` **unchanged**.

---

## C. Audit-009 N3 closure

**N3 said:** the revision was passed to `from_pretrained` and recorded, but the probe never
checked what actually loaded.

**Why that mattered.** `revision=` is an argument, not an outcome. A stale or shared cache,
or a tokenizer resolved from a local directory, could yield measurements attributed to the
wrong commit while the artifact recorded `revision_pinned: true`. That is precisely the
class of silent misattribution audits 007–009 were written to eliminate on the VnCoreNLP
side; leaving it on the PhoBERT side made the pipeline asymmetrically rigorous.

**Closed by** observing the resolved commit post-load and comparing it, with the ambiguous
`revision_pinned` flag removed entirely so it cannot be misread again.

---

## D. Tokenizer revision contract

```text
revision_requested       full 40-char lowercase commit SHA, from --revision
revision_observed        commit read back from the loaded tokenizer's own files
revision_verified        requested is set AND observed is recoverable AND equal
revision_evidence        the resolved file paths the observation came from
revision_evidence_source how it was recovered, or why it could not be
```

`revision_verified` is the only field that means "verified". The former
`revision_pinned` — which only ever meant "the CLI argument was present" — is gone, and a
test asserts it cannot reappear.

**Full-SHA policy.** `--revision` must be exactly 40 lowercase hex characters. `main`,
`master`, `refs/heads/main`, tag names, abbreviated SHAs, over-long strings and uppercase
are all rejected at argument validation with an explanation. Mixed case is rejected rather
than normalised, so a formatting problem surfaces as a formatting error instead of a
confusing comparison failure later.

---

## E. Resolved revision evidence

The Hugging Face hub caches as `models--org--name/snapshots/<commit_sha>/<file>`, and the
snapshot directory is always the **resolved** commit — passing `main` still lands under the
SHA it resolved to. Reading that back from a file the tokenizer actually loaded is genuine
post-load evidence, not a restatement of the request.

Candidate paths are collected from documented surfaces only: the `vocab_file`,
`merges_file`, `tokenizer_file` and `name_or_path` attributes, plus `init_kwargs` values.
Nothing depends on a guessed private attribute; anything that is not a snapshot path is
ignored rather than interpreted.

**No second lookup.** The observation performs no download and no hub query — a test greps
the function body for `hf_hub_download`, `snapshot_download`, `HfApi`, `list_repo_refs` and
`urlopen`. Re-resolving the repository would either echo the request or reintroduce the
mutability being guarded against.

**Ambiguity is not resolved by picking.** If two resolved files carry different snapshot
SHAs, `revision_observed` is `None` with the reason recorded, rather than one being chosen.

---

## F. Fail-closed behaviour

| Situation | Behaviour |
|---|---|
| observed == requested | `revision_verified: true` |
| observed != requested | **abort**, exit 3, printing requested, observed and the evidence paths |
| observed unrecoverable | `revision_verified: false`; warning printed; run not scientifically usable |
| `--revision` absent | exit 2 unless `--allow-floating-revision` |
| `--revision` not a full SHA | exit 2, naming the reason |

Nothing here is a warning-only path except the unrecoverable case, which is by design: it
does not falsify the run, it merely cannot certify it, so the run proceeds and is marked
unusable.

---

## G. `scientifically_usable` final contract

```text
scientifically_usable = tokenizer.revision_verified AND segmenter.pinned
segmenter.pinned      = vncorenlp_revision_verified AND vncorenlp_hashes_verified
```

So a run is scientifically usable only when all six hold:

1. an explicit PhoBERT revision was supplied, as a full immutable SHA;
2. the actual tokenizer snapshot revision was independently observed;
3. observed == requested;
4. the VnCoreNLP checkout's Git revision equals the manifest revision;
5. the exact required VnCoreNLP jar was used;
6. all three VnCoreNLP resource digests matched.

---

## H. VnCoreNLP regression check

Audit 009's closure is intact and regression-tested:

* the committed pin still reads `status: PINNED`, revision
  `62bbc58fe5d113c898eae112656be97dcf50b3a0`, and the three exact digests — asserted
  against hard-coded values so a silent edit fails the suite;
* a matching checkout still yields `revision_verified`, `hashes_verified` and `pinned` all
  true;
* revision mismatch, digest mismatch, missing jar and missing `.git` behave exactly as
  audited;
* no downloader returned, and exact-jar selection is unchanged.

---

## I. Run artifact provenance

`config.json` now carries a `phobert_provenance` block — `checkpoint`,
`revision_requested`, `revision_observed`, `revision_verified`, `revision_evidence`,
`revision_evidence_source`, `tokenizer_class`, `is_fast`, vocab size, special tokens — plus
top-level `tokenizer_revision_requested` / `tokenizer_revision_observed` /
`tokenizer_revision_verified` and `scientifically_usable`. The `vncorenlp_provenance` block
from audit 009 is unchanged, as are audit 007's path and cwd diagnostics.

`report.md` renders a **PhoBERT tokenizer provenance** table listing the evidence paths, and
when the revision was not verified it states plainly that "supplying `--revision` is an
argument, not a verification" and that the run is not scientifically usable. The console
prints the specific reasons — tokenizer revision, VnCoreNLP revision, VnCoreNLP hashes —
rather than a single unexplained false.

---

## J. Proposal / decision-log consistency

| Surface | State |
|---|---|
| `unmark-proposal.md` | **Unchanged.** Provenance strictness is not a method change. |
| D-B3B0-001 | **OPEN** — segmentation vs `T(b(x))`. |
| D-B3B0-002 | **OPEN** — backbone checkpoint not locked. The probe revision is evidence provenance only. |
| D-B3B0-003, D-B3B0-004, D-B3B0-005 | Unchanged; 004 remains CLOSED by 005. |
| D-B3B0-006 | New: original state, final state, evidence strategy, fail-closed contract, reason, affected areas, "Proposal updated: NO". |

---

## K. Test evidence

| Command | Result |
|---|---|
| `.venv/bin/python -m pytest -q -p no:cacheprovider` | **1533 passed in 7.50s** |
| `tests/test_alignment_contracts.py` | 141 passed |
| All previous suites (G−1, B1A, B2, B3A) | green |
| `pip list` | 7 packages, unchanged — nothing installed |

No transformers, torch, Java, VnCoreNLP or network. Tokenizer tests use a fake tokenizer
object with synthetic cache paths; VnCoreNLP revision tests use temporary Git repositories
created offline.

| # | Proof | Test |
|---|---|---|
| 1 | requested revision recorded | `test_matching_observed_revision_verifies` |
| 2 | observed recorded separately | same, plus `test_tokenizer_contract_serialises_the_separated_fields` |
| 3 | requested == observed → verified | `test_matching_observed_revision_verifies` |
| 4 | requested != observed → fail closed | `test_mismatched_observed_revision_does_not_verify`, `test_probe_fails_closed_on_a_post_load_revision_mismatch` |
| 5 | observed unavailable → unusable | `test_unobservable_revision_returns_none_with_a_reason`, `test_supplying_a_revision_is_not_by_itself_verification` |
| 6 | supplying `--revision` is not verification | `test_supplying_a_revision_is_not_by_itself_verification` |
| 7–9 | `main`, branches, tags, short/upper SHAs rejected | `test_non_immutable_revisions_are_rejected` (12 cases), `test_probe_rejects_a_non_full_revision_at_the_cli` (4 cases) |
| 10 | synthetic snapshot path extracts the SHA | `test_snapshot_path_yields_the_commit_sha`, `test_windows_style_snapshot_path_is_handled` |
| 11 | malformed paths fabricate nothing | `test_malformed_paths_do_not_fabricate_a_revision` (7 cases), `test_disagreeing_snapshot_paths_yield_no_revision` |
| 12 | `scientifically_usable` keys on verification | `test_scientifically_usable_depends_on_tokenizer_revision_verification`, `test_ambiguous_revision_pinned_flag_is_gone` |
| 13 | VnCoreNLP verification unchanged | `test_vncorenlp_verification_is_unchanged`, `test_committed_vncorenlp_pin_is_still_intact` |
| 14–15 | no downloader, no model loading | `test_no_downloader_and_no_model_loading_after_this_change` |
| 16 | output-root/cwd repair intact | `test_output_root_repair_is_still_intact` |
| 17 | previous tests green | full suite |

Plus `test_observation_performs_no_network_or_second_download` and
`test_revision_is_observed_from_init_kwargs`.

---

## L. Final scientific Colab command

```bash
cd unmark-draft
pip install "transformers==4.57.6"
pip install py_vncorenlp
export HF_HOME="$PWD/.hf-cache"

python scripts/fetch_vietnamese_syllable_inventory.py        # B3A eligibility

# .vncorenlp/ must be the checkout at 62bbc58fe5d113c898eae112656be97dcf50b3a0.
# <FULL_TOKENIZER_SHA> must be the full 40-character commit hash of the tokenizer
# revision, resolved from the model's Hugging Face page. Branches and tags are rejected.
python scripts/b3b0_phobert_input_probe.py \
    --checkpoint vinai/phobert-base \
    --revision <FULL_TOKENIZER_SHA> \
    --vncorenlp-dir .vncorenlp \
    --vncorenlp-manifest configs/linguistics/vncorenlp_v1.2.json
```

Exit codes: `2` bad or missing revision argument; `3` the tokenizer that loaded is not the
one requested. Artifacts land in `<repo>/results/b3b0/<run_id>/`. **Read no number until
`config.json` shows `scientifically_usable: true`** — and if it is false, the console and
report name which specific check failed.

---

## M. Blocking issues

`None`.

D-B3B0-001 continues to block **B3B**, by design and unchanged.

---

## N. Non-blocking issues

```
ID: N1  Observation depends on the Hugging Face cache layout
    The commit is parsed from `snapshots/<sha>/`. That layout is stable and is what the
    hub has used for years, but it is a convention rather than a documented API. If it
    ever changes, revision_observed becomes None and the run is marked unusable — the
    failure mode is safe, not silent.

ID: N2  A tokenizer loaded from a local directory cannot be verified
    Deliberate: there is no commit to observe. The run is marked not scientifically usable
    rather than being refused, so offline exploration still works.

ID: N3  status field in the VnCoreNLP manifest remains advisory
    Carried over from audits 008/009. Rejection is driven by placeholder digests, not the
    status string.

ID: N4  PRESEGMENTED_DATASET still unprobed
    Unchanged from audits 006-009; no dataset is pinned yet.

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
  `docs/audits/009-b3b0-provenance-closure.md`,
  `docs/audits/010-b3b0-tokenizer-revision-verification.md`
* **Ignored by design:** `.vncorenlp/`, `.probe_*`

No `add`, `commit`, `push`, `tag`, `stash`, `reset`, `checkout` or `restore` was run.

```text
AUDIT FILE WRITTEN: docs/audits/010-b3b0-tokenizer-revision-verification.md
OTHER AUDIT MODIFICATIONS MADE: NO
COMMIT CREATED: NO
```
