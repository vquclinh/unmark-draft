# Audit 005 — B3A Vietnamese syllable eligibility

| | |
|---|---|
| **Audit id** | 005 |
| **Date (UTC)** | 2026-08-19 |
| **Scope** | The full B3A state: inventory provenance, eligibility rule, B1A and B2 integration |
| **Repository state** | `HEAD = 76dace9` ("implement B2 deterministic corruption engine"); B3A work uncommitted |
| **Predecessors** | [002](002-b1a-orthography-core.md), [003](003-b2-deterministic-corruption.md), [004](004-b2-eligibility-safety-followup.md) |
| **Phase** | Phase 0 / B3A |

---

## A. Verdict

**PASS**

GAP-2 is closed. Eligibility is membership of a pinned Vietnamese syllable inventory,
tested on the stripped form — `canon → strip_to_base → casefold → lookup` — so
`eligibility(clean) == eligibility(corrupted)` holds by construction and the base grid
stays invariant. The source is pinned by gist revision and SHA-256; because the upstream
carries no license statement the raw list is **not** committed, only its provenance, and
the fetch script verifies the checksum and never advances the pin. The B2 scientific guard
now stands down when the inventory is present and still fires when it is absent. English
spans no longer enter the denominator (`toi dung Python va PyTorch`: 5 candidates → 3
eligible) and `café` is no longer stripped. The B2 deterministic engine is untouched:
`CORRUPTION_SCHEMA_VERSION` remains `b2-v1` and every audited score and output is
bit-identical. 1382 tests pass offline, with no network required.

---

## B. Files changed

**New**

| File | Purpose |
|---|---|
| `unmark/linguistics/inventory.py` | Provenance, verified loading, stripped-form set, memoisation |
| `unmark/linguistics/classify.py` | `classify_candidate`, `is_vietnamese_candidate`, `make_classifier` |
| `unmark/linguistics/__init__.py` | Package exports |
| `configs/linguistics/vietnamese_syllables.yaml` | Pinned manifest: source, revision, SHA-256, counts, license status |
| `scripts/fetch_vietnamese_syllable_inventory.py` | Fetch + verify; the only network operation |
| `scripts/b3a_eligibility_check.py` | Real-inventory integration check |
| `tests/test_linguistics_eligibility.py` | 77 tests |
| `tests/fixtures/vietnamese_syllables_sample.txt` | Small hand-written fixture so pytest needs no network |
| `results/b3a/.gitkeep` | Run-artifact directory |

**Modified**

| File | Change |
|---|---|
| `unmark/orthography/marks.py` | `Eligibility.VIETNAMESE_CANDIDATE` added; `UNDECIDED` redefined as "unresolvable" |
| `unmark/orthography/decompose.py` | Optional `eligibility_classifier` injection; core reconstruction unchanged |
| `unmark/corruption/eligibility.py` | `active_eligibility_policy()` resolves dynamically from inventory availability |
| `unmark/corruption/corrupt.py` | Filters spans by eligibility at the single reserved point; records provenance |
| `unmark/corruption/models.py` | `scored_units`; real `eligible_units` / `realized_probability` when resolved |
| `unmark/corruption/__init__.py` | Exports the dynamic policy |
| `scripts/b2_corruption_self_check.py` | Policy-aware assertions and reporting |
| `tests/test_corruption.py` | Migrated; safety tests now cover both resolved and missing-inventory states |
| `.gitignore` | `.resources-cache/` |
| `README.md` | "Vietnamese eligibility (B3A)" |
| `docs/spec/decisions.md` | D-B2-003 closed; D-B3A-001 recorded |

---

## C. GAP-2 closure

| | |
|---|---|
| **Was** | No syllable inventory existed. Every alphabetic run was a candidate; `SELF_CHECK` only; scientific path hard-blocked (D-B2-003, audit 004). |
| **Now** | `Eligibility.VIETNAMESE_CANDIDATE` when the stripped form is in the pinned inventory, `NOT_APPLICABLE` when not, `UNDECIDED` only when the inventory is unavailable. |

The rule is a pure function of the stripped form. `membership_form` applies
`canon → strip_to_base → casefold`, so it cannot observe whether the input carries
diacritics. Verified directly: `membership_form("học") == membership_form("hoc")`, and
eligibility is identical across `FULL`, `P25`, `P50`, `P75`, `P100` and `STRIP_ALL` for the
same text.

`UNDECIDED` no longer leaks into resolved runs: it appears only when no classifier was
supplied, which is the one state that genuinely means "cannot resolve".

---

## D. Source provenance

| Field | Value |
|---|---|
| source name | `all-vietnamese-syllables.txt` |
| author | `hieuthi` |
| gist | `https://gist.github.com/hieuthi/0f5adb7d3f79e7fb67e0e499004bf558` |
| gist id | `0f5adb7d3f79e7fb67e0e499004bf558` |
| revision | `135a4d9716e49a981624474156d6f247b9b46f6a` (latest of 5, 2017-03-22) |
| raw URL | revision-pinned, not `…/raw/…` floating |
| sha256 | `78eeb840d50455b14bd564da5aed7318d96468b8deaad5986b77bf5c538315d2` |
| bytes | 116,290 |
| encoding | UTF-8, LF, no BOM, all entries already NFC |
| entries | 17,974 non-empty, one syllable per line, no multi-word lines |
| retrieved | 2026-08-19 |
| upstream description | "All possibly existent Vietnamese syllables, created by combine all onsets with all rimes." |

The revision was resolved from GitHub's public gist metadata (no credentials). The raw URL
embeds the revision SHA, so it cannot silently follow upstream.

**Structural, not frequency.** The author's separate ~7,184-entry *common* list was
deliberately not used: proposal §4.3 asks which syllables are legal, not which are
frequent.

---

## E. License / redistribution status

**`NO_EXPLICIT_LICENSE`.** The gist has no LICENSE file, no license field, and no license
statement in its description or file listing — checked against the API response. A public
gist is not a licence, so redistribution permission is **not** established.

Consequences, all implemented:

* the raw list is **not committed** — a test asserts no tracked path matches
  `all-vietnamese-syllables`;
* it is fetched into `.resources-cache/`, which is repo-local and git-ignored (asserted by
  `git check-ignore` in a test);
* only provenance — source, revision, SHA-256, counts, license status — is in git;
* everything scientific fails loudly when the resource is absent;
* the test fixture was **hand-written** from well-known Vietnamese syllables rather than
  copied from the upstream file, so no excerpt is redistributed either.

If the author later states a compatible license, vendoring may be reconsidered and recorded
as a decision.

---

## F. Final eligibility contract

```text
classify_candidate(span, inventory):
    inventory is None            -> Eligibility.UNDECIDED
    form not alphabetic          -> Eligibility.NOT_APPLICABLE
    form in inventory            -> Eligibility.VIETNAMESE_CANDIDATE
    otherwise                    -> Eligibility.NOT_APPLICABLE

where form = casefold(strip_to_base(canon(span)))
```

`ELIGIBILITY_SCHEMA_VERSION = "vn-syllables-v1"`, distinct from
`CORRUPTION_SCHEMA_VERSION = "b2-v1"`. Both, plus the inventory revision and SHA-256, are
recorded in every scientific corruption result.

**Nothing semantic is consulted**: no language identification, frequency, dictionary,
capitalisation heuristic or sentence context. A test asserts the classifier body contains
no such reference, and the design reason is that any of them would destroy the
pure-function-of-the-stripped-form property.

---

## G. B1A integration

Eligibility is an **injected policy layer**, not a core dependency:
`decompose(text, eligibility_classifier=…)`. Without a classifier every span is
`UNDECIDED` and nothing else differs, so Unicode decomposition and reconstruction never
depend on an external file.

Verified: `recompose(decompose(x)) == canon(x)` holds with and without a classifier, and
`base_text`, `letter_channel` and `observed_tone_channel` are identical either way. All
B1A tests remain green.

---

## H. B2 scientific unblock

| Situation | Behaviour |
|---|---|
| inventory present | `active_eligibility_policy()` → `VIETNAMESE_SYLLABLE_INVENTORY`; `purpose=SCIENTIFIC` succeeds; `eligible_units` and `realized_probability` return real values |
| inventory absent | policy → `UNRESOLVED`; `purpose=SCIENTIFIC` raises; `SELF_CHECK` still works and is stamped provisional; `eligible_units` raises |

The policy is *computed*, not hard-coded, so deleting the git-ignored cache re-arms the
guard rather than leaving a stale "resolved" flag.

Non-eligible spans do not enter the denominator, are never selected, and are never
modified. Filtering happens at the single point audit 004 reserved for it.

**Engine untouched.** `CORRUPTION_SCHEMA_VERSION` is still `b2-v1`;
`unit_index` is still the span's index in the **full candidate list**, so a given unit's
score is bit-identical to audit 003/004. Filtering changes which units are scored, never
what score a unit gets. Re-verified:

```text
unit_score(seed=42, sample_id="s1", index=0) == 0.21396497977394088     unchanged
P75  seed=42 -> "Tôi đang nghiên cứu xư lý ngôn ngư tự nhiên."          unchanged
P100 seed=42 -> "Tôi đang nghiên cưu xư ly ngôn ngư tư nhiên."          unchanged
STRIP_ALL    -> "Toi dang nghien cuu xu ly ngon ngu tu nhien."          unchanged
```

`P100` vs `STRIP_ALL` remain distinct; H4 metadata (`clean_lexical_tone`,
`oracle_tone_is_missing`, `oracle_tone_is_genuine_ngang`) is unchanged; base invariance
holds for every condition.

---

## I. Proposal / decision-log consistency

| Surface | State |
|---|---|
| `unmark-proposal.md` §4.3 | Already specifies this rule exactly. **Unchanged** — the implementation follows it rather than changing it. |
| `unmark-proposal.md` §5.3 | Unchanged by B3A. |
| `docs/spec/decisions.md` D-B2-003 | Marked **CLOSED** by D-B3A-001, with the prior status preserved in the entry rather than erased. |
| `docs/spec/decisions.md` D-B3A-001 | New: original proposal wording, previous temporary state, final implementation, provenance, license handling, known error mode, affected areas, engine-unchanged statement. |
| Category table | `TEMPORARY IMPLEMENTATION FALLBACK` is now empty; `KNOWN DEFERRED GAP` retains only D-B2-005 (`VARIANT`). |
| `README.md` | Rule, provenance, fetch instructions, deliberate-false-positive note. |

---

## J. Spec changes / deviations

**None.** B3A implements proposal §4.3 as written, including its explicit acceptance of
ambiguous spans resolving towards Vietnamese. No proposal sentence needed changing or
clarifying, so none was changed — the decision log records the implementation, not a
rewrite of the specification.

---

## K. Blocking issues

`None`

---

## L. Non-blocking issues

```
ID: N1  Deliberate false positives
    English words that are valid stripped Vietnamese syllables (ban, the, com, on, in,
    an, la, co) are classified Vietnamese, so in an English sentence they are corrupted.
    This is proposal 4.3's accepted error mode and cannot be fixed without breaking the
    stripped-form invariance. Measured: 12/12 probed ambiguous forms accepted.

ID: N2  Inventory is memoised per process
    Loading parses ~18k entries, so the result is cached. A test that deletes or corrupts
    the cached file must call clear_inventory_cache() or it will see a stale object. The
    provided tests do.

ID: N3  Upstream availability
    The resource depends on a third-party gist remaining reachable. The checksum protects
    against silent change, not against disappearance. If it vanishes, the pin and its
    SHA-256 remain in git and any identical copy can be dropped into the cache.

ID: N4  Single-syllable coverage
    2,489 stripped forms is the count of distinct base shapes, not of words. Multi-
    syllable Vietnamese words are handled because spans are syllables, but a hyphenated
    or run-together compound is one span and will not match.

ID: N5  unmark-proposal.pdf
    Still stale from the earlier 4.2 and 5.3 edits. Unchanged by this task.
```

---

## M. Test evidence

| Command | Result |
|---|---|
| `.venv/bin/python -m pytest -q -p no:cacheprovider` | **1382 passed in 6.10s** |
| `tests/test_linguistics_eligibility.py` | 77 passed |
| `tests/test_corruption.py` | 518 passed |
| B1A suites | green |
| G−1 suites | green |
| `scripts/b2_corruption_self_check.py` | 102 records, 0 failures, resolved policy |
| `scripts/b3a_eligibility_check.py` | `B3A_ELIGIBILITY_RESOLVED` |
| `pip list` | 7 packages, unchanged — nothing installed |
| `ls ~/.cache/huggingface/hub` | unchanged — no model downloaded |

**No network is required by the suite.** Unit tests use the committed fixture; real-
inventory tests skip when the cache is absent. An AST test asserts that no module under
`unmark/` imports `urllib`, `requests`, `socket` or `http` — only the fetch script does.

Covered by name: provenance pinned, revision pinned, SHA-256 recorded, checksum mismatch
fails closed, entry-count change rejected, missing inventory message, repo-local cache,
git-ignored cache, no global cache, raw list not committed, deterministic construction,
order independence, collision collapse, casefold policy, NFC/NFD equivalence, all six
tones, every letter diacritic, đ/Đ, uppercase, stripped forms, long Latin strings, English
words, ambiguous ASCII, URLs, e-mail, digits, emoji, punctuation boundaries, and
`eligibility(clean) == eligibility(P25) == eligibility(P100) == eligibility(STRIP_ALL)`.

---

## N. Real-inventory evidence

```text
revision            : 135a4d9716e49a981624474156d6f247b9b46f6a
sha256              : 78eeb840d50455b14bd564da5aed7318d96468b8deaad5986b77bf5c538315d2
raw entries         : 17974
unique canonical    : 17954
unique stripped     : 2489
collisions          : 15465
known-valid accepted: 20/20
known-foreign reject: 12/12
ambiguous accepted  : 12/12
B2 guard            : SATISFIED
Status              : B3A_ELIGIBILITY_RESOLVED
```

Accepted: `tôi đang nghiên cứu xử lý ngôn ngữ tự nhiên học đường phở người được nguyễn hoà
thuý khoẻ quả`. Rejected: `machine learning python pytorch café google server email
javascript strength qwerty transformer`.

Collisions are expected and large: an inventory enumerating every tone and letter
diacritic collapses `ma má mà mả mã mạ` onto one stripped form.

---

## O. Mixed-language / ambiguity evidence

`STRIP_ALL`, real inventory:

| Input | Output | Candidates | Eligible |
|---|---|---:|---:|
| `toi dung Python va PyTorch de train model` | `toi dung Python va PyTorch de train model` | 8 | 4 |
| `Tôi đang học machine learning tại VNU-HCM` | `Toi dang hoc machine learning tai VNU-HCM` | 8 | 4 |
| `café ngon lắm` | `café ngon lam` | 3 | 2 |
| `Xem tại https://example.edu.vn nhé` | `Xem tai https://example.edu.vn nhe` | 7 | 3 |
| `Liên hệ lien.he@example.com` | `Lien he lien.he@example.com` | 6 | 5 |
| `Năm 2026, GDP tăng 6,5% 😄` | `Nam 2026, GDP tang 6,5% 😄` | 3 | 2 |

`Python`, `PyTorch`, `machine`, `learning` survive untouched. **`café` survives** — its
stripped form `cafe` is not in the inventory, so sharing the acute codepoint with a
Vietnamese tone mark is no longer sufficient to strip it. That was the concrete defect
audit 004 flagged, and it is fixed.

The deliberate error mode, asserted rather than hidden: `ban the com on in an la co nam ma
cam hoa` are all accepted, so in `de train model` the span `de` is eligible.

---

## P. Future B3B compatibility

- **No tokenizer work was done.** No transformers, torch, SentencePiece or PhoBERT; no
  subwords, embeddings, alignment or training. AST tests enforce this for
  `unmark/{linguistics,corruption,orthography}`.
- **What B3B inherits.** Each `SyllableSpan` now carries a resolved `eligibility` alongside
  its canonical and base offsets, which is exactly what §4.4's label-propagation step needs
  to assign `N/A` to non-Vietnamese subwords.
- **Grid invariance is preserved.** Eligibility reads the stripped form, so the labels B3B
  propagates are identical for clean and corrupted input — the property §4.4 step 2 relies
  on.
- **Versioning.** `ELIGIBILITY_SCHEMA_VERSION` travels in every artifact next to
  `CORRUPTION_SCHEMA_VERSION`, so a later inventory change cannot silently alter alignment
  results.

---

## Q. Git state

* **Branch:** `main`
* **HEAD:** `76dace9` "implement B2 deterministic corruption engine" — the researcher's
  commit, made during this session; the reflog shows no commit from this work
* **Staged:** none (`git diff --cached --stat` empty)
* **Unstaged:** `.gitignore`, `README.md`, `docs/spec/decisions.md`,
  `scripts/b2_corruption_self_check.py`, `tests/test_corruption.py`,
  `unmark/corruption/{__init__,corrupt,eligibility,models}.py`,
  `unmark/orthography/{decompose,marks}.py`
* **Untracked:** `configs/linguistics/`, `docs/audits/005-b3a-vietnamese-syllable-eligibility.md`,
  `results/b3a/`, `scripts/b3a_eligibility_check.py`,
  `scripts/fetch_vietnamese_syllable_inventory.py`, `tests/fixtures/`,
  `tests/test_linguistics_eligibility.py`, `unmark/linguistics/`
* **Not tracked by design:** `.resources-cache/` (git-ignored external resource)

No `add`, `commit`, `push`, `tag`, `stash`, `reset`, `checkout` or `restore` was run.

```text
AUDIT FILE WRITTEN: docs/audits/005-b3a-vietnamese-syllable-eligibility.md
OTHER AUDIT MODIFICATIONS MADE: NO
COMMIT CREATED: NO
```
