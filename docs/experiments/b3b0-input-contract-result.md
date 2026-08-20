# B3B-0 input-contract probe — result record

Persistent record of the **scientifically usable** B3B-0 Colab run. Numbers are
the researcher's verified measurements, transcribed, not recomputed here.

The earlier Colab run is **excluded**: it was invalidated for scientific
decision-making because the VnCoreNLP resource provenance was not guaranteed
(see [audit 007](../audits/007-b3b0-colab-probe-repair.md) §D).

---

## Run provenance

| Field | Value |
|---|---|
| run id | `20260820T031644Z` |
| repository HEAD | `48c44cdc597614eb06abd52c4fe16e8ab5235c07` |
| `scientifically_usable` | **true** |

**PhoBERT**

| Field | Value |
|---|---|
| checkpoint | `vinai/phobert-base` |
| requested revision | `01daacda68afe13d83023d16ec647239e344a1e6` |
| observed revision | `01daacda68afe13d83023d16ec647239e344a1e6` |
| `revision_verified` | true |
| tokenizer | `PhobertTokenizer` |
| `is_fast` | false |
| `model_weights_loaded` | false |

**VnCoreNLP**

| Field | Value |
|---|---|
| revision | `62bbc58fe5d113c898eae112656be97dcf50b3a0` |
| `revision_verified` | true |
| `hashes_verified` | true |
| `pinned` | true |

---

## Path results

18 cases × 6 conditions = 108 observations per path.

| Path | Grid invariant | Mean fragmentation | Unknown tokens | Offsets |
|---|---|---:|---:|---|
| `RAW_BASE` | **18/18** | 1.5674165421972441 | 12 | `ABSENT` |
| `BASE_THEN_SEGMENT` | 18/18 | 1.5424165421972438 | 12 | `ABSENT` |
| `CLEAN_SEGMENT_THEN_BASE` | 18/18 | 1.6191500426149548 | 12 | `ABSENT` |
| `OBSERVED_SEGMENT_THEN_BASE` | **9/18** | 1.5762836172923893 | 12 | `ABSENT` |

`OBSERVED_SEGMENT_THEN_BASE` broke invariance on: `vi_research`,
`vi_multisyllable`, `vi_city`, `vi_proper_names`, `vi_uppercase`, `email`,
`emoji`, `hyphenated`, `long_sentence`.

---

## Researcher analysis of the artifact

All 432 observations were inspected.

1. **`CLEAN_SEGMENT_THEN_BASE` is not deployable.** It requires clean text,
   which does not exist at inference — the premise of the project (§1.3).

2. **`OBSERVED_SEGMENT_THEN_BASE` violates the base-token-grid invariant** and is
   rejected. Segmenting whatever was observed makes the grid depend on the
   corruption level, contradicting §4.5.

3. **`BASE_THEN_SEGMENT` is invariant but recovers little segmentation.**
   VnCoreNLP applied to stripped Vietnamese preserves little of what it produces
   from clean Vietnamese. Excluding the deliberately pre-segmented diagnostic
   case:

   | | underscore separators |
   |---|---:|
   | clean-segmented output | 39 |
   | base-then-segment output | 8 |

   > **This is a post-hoc diagnostic count, not a formal word-segmentation
   > accuracy or recall metric.** It shows the magnitude of the discrepancy and
   > nothing more.

   The merges also differ, not merely their number:

   ```text
   clean            : Truong Dai_hoc Khoa_hoc_Tu_nhien ...
   base then segment: Truong_Dai hoc Khoa hoc Tu nhien ...
   ```

4. **The segmenter is not transparent on arbitrary mixed text.** In at least one
   diagnostic, the supplementary-plane emoji `😄🎉` came back as
   private-use-looking characters after the VnCoreNLP path. A single example —
   not generalised — but recorded, because it shows the segmentation path is not
   a pass-through for non-Vietnamese content.

5. **`BASE_THEN_SEGMENT`'s fragmentation gain over `RAW_BASE` is small**:
   1.5424 vs 1.5674, with an identical unknown-token count (12).

---

## Decision

`RAW_BASE` is selected as the main UNMARK base path. See
[`docs/spec/decisions.md`](../spec/decisions.md) D-B3B0-001 and
[audit 011](../audits/011-b3b1a-input-path-and-alignment-preflight.md).

---

## Known defect in this artifact

Every alignment eligibility label in the run reads `Eligibility.UNDECIDED`:

```text
UNDECIDED             3960
VIETNAMESE_CANDIDATE     0
NOT_APPLICABLE           0
```

**Cause**, reproduced locally and fixed: `DEFAULT_MANIFEST` in
`unmark/linguistics/inventory.py` was a *relative* path.
`py_vncorenlp.VnCoreNLP()` `chdir()`s into its resource directory before the
probe's case loop runs, so `try_load_inventory()` looked in the wrong place,
returned `None`, and no classifier was injected.

**Scope of the damage.** The same lookup failure means the probe's B2 corruption
also ran under the provisional candidate-span policy rather than the resolved
one. That affects only `OBSERVED_SEGMENT_THEN_BASE`, whose input is the corrupted
text: the other three paths tokenize `base_text` (or clean text), which is
invariant under corruption and under the eligibility policy alike. So

* the **path decision stands** — `RAW_BASE`'s invariance, fragmentation,
  unknown-token count and offset finding are unaffected;
* `OBSERVED_SEGMENT_THEN_BASE`'s specific outputs were produced under the
  provisional policy, but its rejection reason (corruption changes the observed
  text, so segmentation changes, so the grid breaks) holds under either policy;
* the **eligibility labels in this artifact carry no information** and must not
  be read.

The fix anchors the manifest path to the repository, so library behaviour no
longer depends on the caller's working directory.
