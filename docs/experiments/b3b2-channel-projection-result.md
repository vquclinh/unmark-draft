# B3B-2 — orthographic channel projection on the real PhoBERT token grid

Final scientific evidence for deterministic B3B. Produced by
[`scripts/b3b2_channel_projection_probe.py`](../../scripts/b3b2_channel_projection_probe.py)
on Colab against the real slow tokenizer.

**No model weights were loaded. Nothing was trained.** The probe uses the
tokenizer only.

## Provenance

| | |
|---|---|
| **Run id** | `20260820T041812Z` |
| **Repository HEAD** | `c09516e03300e670fc20ac10173d7c346106fd6a` |
| **Checkpoint** | `vinai/phobert-base` |
| **Revision** | `01daacda68afe13d83023d16ec647239e344a1e6` |
| **Tokenizer class** | `PhobertTokenizer` |
| **`is_fast`** | `false` |
| **Model weights loaded** | `false` |
| **Python** | 3.12.13 |
| **Final status** | **`B3B2_CHANNEL_PROJECTION_COMPLETE`** |

The revision is **reproducibility evidence, not the final backbone decision**.
[D-B3B0-002](../spec/decisions.md#d-b3b0-002) remains **OPEN**.

The counts below were transcribed from the saved run artifact, which was
independently inspected after the run. They are recorded here as reported; the
local ML-free environment cannot re-execute the probe.

## Conditions and cases

Six corruption conditions — `FULL`, `P25`, `P50`, `P75`, `P100`, `STRIP_ALL` —
applied to **seven test cases**:

`plain`, `tones`, `letters`, `mixed_en`, `punct`, `url`, `digits`

These are seven *cases*, not seven conditions: the grid is 7 × 6 = **42
case-condition combinations**.

## Structural results

| Measure | Result |
|---|---|
| Token grid invariant across all six conditions | **7 / 7** |
| Piece ranges invariant across all six conditions | **7 / 7** |
| Sequence consistent under every condition | **7 / 7** |
| Monotonic marked-tone degradation | **7 / 7** |
| `STRIP_ALL` leaves no marked tone | **7 / 7** |
| Multi-candidate authoritative pieces | **0** |

Per case-condition combination:

| Measure | Result |
|---|---|
| Token ids matched the per-case `FULL` authoritative grid | **42 / 42** |
| Piece ranges matched | **42 / 42** |
| Sequence consistency | **42 / 42** |

## Tone-label totals

Counts over all seven cases, per condition:

| Condition | UNMARKED | NANG | SAC | HUYEN | HOI | NGA | NA |
|---|---|---|---|---|---|---|---|
| `FULL` | 16 | 9 | 15 | 5 | 3 | 2 | 25 |
| `P25` | 29 | 5 | 12 | 2 | 1 | 1 | 25 |
| `P50` | 35 | 5 | 6 | 2 | 1 | 1 | 25 |
| `P75` | 43 | 2 | 3 | 2 | 0 | 0 | 25 |
| `P100` | 50 | 0 | 0 | 0 | 0 | 0 | 25 |
| `STRIP_ALL` | 50 | 0 | 0 | 0 | 0 | 0 | 25 |

Three checks derived from that table, each of which the table would have failed
had the projection been leaking:

* **Every row totals 75 labelled positions.** The number of authoritative
  positions does not move.
* **Vietnamese positions are constant at 50, `NA` positions constant at 25.**
  Corruption never converts a Vietnamese position into a non-applicable one, and
  never invents one.
* **Marked tones fall 34 → 21 → 15 → 7 → 0 → 0**, and `UNMARKED` rises to
  exactly compensate: 16 → 29 → 35 → 43 → 50 → 50.

`P100` reaches zero marked tones while the Vietnamese syllable positions remain
**`UNMARKED`, not `NA`**. `STRIP_ALL` preserves the same distinction. This is the
`UNMARKED` ≠ `NA` semantics holding under the strongest corruption: "a Vietnamese
syllable with no readable mark" never degrades into "no Vietnamese syllable here".

## Letter applicable-label totals

For **each** of `FULL`, `P25`, `P50`, `P75`, `P100`:

| Label | Count |
|---|---|
| `NONE` | 143 |
| `CIRCUMFLEX` | 10 |
| `HORN` | 6 |
| `STROKE` | 4 |
| `BREVE` | 2 |

For `STRIP_ALL`:

| Label | Count |
|---|---|
| `NONE` | 165 |

No letter-forming diacritic label remains under `STRIP_ALL`.

The **applicable-contributor count is invariant at 165** across all six
conditions. This verifies the intended channel separation from both directions:

* **Tone-only corruptions do not touch the letter channel.** `P25` … `P100`
  leave the letter counts bit-identical to `FULL` — 22 letter-forming marks
  survive every tone-only condition.
* **`STRIP_ALL` removes letter-forming marks while preserving applicability.**
  Those 22 characters become `NONE`, not `NA`: a stripped `ơ` is still a letter
  that *could* carry a Vietnamese letter diacritic. `NONE` ≠ `NA` survives the
  strongest corruption.

## Scope of the monotonic claim

The monotonic degradation result is **an observed result under the locked
deterministic B2 protocol and these seven probe cases**. It is *not* a universal
theorem about arbitrary corruption processes, and must not be described as one.

The genuinely structural invariant is the other one: the equality of the base
stream, the token grid, and the character-range structure. That is what
`b(C_c(x)) = b(x)` gives, and it holds by construction rather than by
observation.
