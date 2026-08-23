# Stage-1 prepared corpus — real descriptive profile

**Status: `STAGE1_CORPUS_PROFILE_RECORDED` — descriptive evidence only.**

Measured on the **real** prepared corpus in a fresh Colab GPU runtime during the
second no-update pre-train smoke. Nothing here changes chunking, `MAX_LENGTH`, or
any scientific constant; it is recorded because it **falsifies an informal
inference** the project had been carrying, and that inference could have led to a
bad decision later.

| | |
|---|---|
| **Measured at HEAD** | `b84b4daac0f2be31266e171d3f56a71611a421e0` |
| **Corpus producer** | `aa49785eadcbd67b64be28a5f67d725c79b41bbb` |
| **Restore** | byte-exact; `COMPLETE.json` and membership digest verified |
| **Model** | **none loaded** — this profile streams the payload |
| **Audit** | [030 §W](../audits/030-pretrain-repository-wide-audit.md) |

## Chunk counts

| Partition | Chunks |
|---|---|
| train | **2 621 624** |
| dev | **11 443** |
| total | **2 633 067** |

Parents: **1 118 224**.

## Chunks per parent document

| | mean | p25 | p50 | p75 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|---|
| **dev** | 2.29 | 1 | 1 | 2 | 4 | 7 | 20 | 481 |
| **train** | 2.35 | 1 | 1 | 2 | 4 | 7 | 22 | **2 479** |

## Token lengths

Recomputed under the exact locked PhoBERT tokenizer — `reference_length` and
`base_length` are **not persisted** in `chunks.jsonl`, so a profile must
recompute them (`recomputed_not_recorded`). Sampling is the partition-aware
deterministic scheme from [030 §U](../audits/030-pretrain-repository-wide-audit.md):
**all 11 443 dev chunks** and a **20 000**-chunk deterministic train sample.

| Stream | mean | p25 | p50 | p75 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|---|
| dev reference | 164.20 | 55 | 211 | 233 | 254 | 255 | 256 | 256 |
| dev RAW_BASE | 181.06 | 62 | 253 | 256 | 256 | 256 | 256 | 256 |
| train reference | 164.81 | 57 | 211 | 230 | 253 | 255 | 256 | 256 |
| train RAW_BASE | 182.71 | 64 | 254 | 256 | 256 | 256 | 256 | 256 |

**`over_max_length`: 0.** No chunk exceeds the 256-token ceiling in either
stream, which is the invariant Stage 6 was built to guarantee.

## What this falsifies

The informal inference was:

> 2.35 chunks per document ⇒ most training chunks are very short.

**That does not follow, and the real data rejects it.** The median document does
produce exactly one chunk (p50 = 1), but a document that fits in one chunk is not
thereby a *short* one — it is one that fits under 256 tokens. The chunks
themselves are commonly long and pressed against the ceiling: median RAW_BASE
length is **253–254 of 256**, and p75 onward is **exactly 256** in both
partitions. Roughly a quarter of chunks are at the ceiling.

The mean (~164 reference, ~182 RAW_BASE) sits well below the median because the
distribution is strongly left-tailed — p25 is 55–64 tokens — so a mean read
alone would have reinforced the same wrong conclusion.

RAW_BASE is consistently **~18 tokens longer** than reference at the mean and
~42 longer at the median. Stripping diacritics does not shorten the token
sequence; it lengthens it, because stripped forms fragment into more BPE pieces.
This is the expected direction given the base grid is defined on `b(x)`, and it
is why the RAW_BASE stream is the one that saturates first.

**No action taken.** Chunking, `MAX_LENGTH = 256`, and every corruption constant
are unchanged. This document exists so the next person reasoning about batch
cost, padding waste, or truncation starts from the measurement rather than from
the ratio.
