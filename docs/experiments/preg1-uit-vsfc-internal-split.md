# Pre-G1 UIT-VSFC internal split — protocol-train / protocol-dev

**Status: `MATERIALISED — BYTE-DETERMINISTIC ACROSS INDEPENDENT RUNS`**

The split has been materialised on the real approved pool and independently
reproduced. **The precommitted expectations below were written before any
membership existed, and the real run matched them exactly.**

**No membership ids are reproduced in this file** — the artifact digests are
recorded instead, and the ids live only in the persisted artifacts.

| | |
|---|---|
| **Implemented at** | `HEAD = 819d09f2df95ca57444a63c86363e614b44ce458` |
| **Executed at** | **`HEAD = 66f4522fa86e5f02f583204ddcad560a62b013c0`** |
| **Audit** | [023](../audits/023-pre-g1-internal-split-materializer-and-fail-closed-contract.md) |
| **Decision** | [D-PREG1-014](../spec/decisions.md#d-preg1-014--internal-split-materialisation-is-fail-closed-and-mapping-order-independent) |
| **Artifact schema** | `preg1-split-v1` |
| **Real membership** | **materialised, verified, persisted (ids only)** |
| **Assignment digest** | `7bd5d1892e23b96035c376936d2168f547661da07b8abc769f5656e9438f4f84` |
| **Downstream score** | **none** |

## Input contract

The materialiser consumes **only** the derived pre-G1 train csv — the
exclusion-applied pool approved in
[Audit 022](../audits/022-uit-vsfc-real-data-profile-integrity-closure.md).
Official validation and official test are not parameters of the program.

| | |
|---|---|
| Dataset | UIT-VSFC v1.0, sentiment only |
| Derived train SHA-256 | `a20c0f7760f32dc48263a79d73ddf5363526c17e9de2afc32d8346b23444d301` |
| Derived rows | 11 424 |
| Derived label counts | negative 5 324 · neutral 458 · positive 5 642 |
| Exclusion policy | `EXCLUDE_ENTIRE_CONFLICTING_CANONICAL_GROUP` (D-PREG1-011) |
| Canonical duplicate groups | 0 |
| Conflicting-label groups | 0 |

Every one of these is re-verified at runtime before any assignment. A pool with
the right row count but the wrong digest is refused: it is not the approved
corpus.

## Split parameters — imported, never restated

| | |
|---|---|
| Fractions | `protocol-train` 0.80 · `protocol-dev` 0.20 |
| Seed tag | `UNMARK-PREG1-SPLIT-UITVSFC-v1` |
| Seed | **17486** (derived from the tag, not chosen) |
| Grouping | `text_digest(canon(text))`; canonical groups are atomic |
| Allocation order | descending fraction, then ascending name — a function of the mapping's content, never its insertion order |
| Stratification | by each group's **single distinct** label; conflicts are an error, never a vote |

## Precommitted expectation

The derived pool has **zero** canonical duplicate groups, so every group is a
singleton and the per-class allocation is fully determined by the class totals
and the allocation rule. These numbers are therefore **derivable before any
membership exists**, which is what makes them a precommitment:

| Part | negative | neutral | positive | total |
|---|---|---|---|---|
| `protocol-train` | 4 259 | 366 | 4 514 | **9 139** |
| `protocol-dev` | 1 065 | 92 | 1 128 | **2 285** |

A synthetic pool carrying the real class totals reproduces these exactly under
the locked seed. When the input digest is the locked one, the materialiser
**refuses to write** unless the real membership matches.

## Observed result

The precommitment above was matched **exactly**:

| Part | negative | neutral | positive | total |
|---|---|---|---|---|
| `protocol-train` | 4 259 | 366 | 4 514 | **9 139** |
| `protocol-dev` | 1 065 | 92 | 1 128 | **2 285** |

Disjoint; union equals all 11 424 approved ids; each sample exactly once; all
11 424 canonical groups singleton; **cross-part canonical leakage 0**;
conflicting-label groups 0. Official validation and official test were not
inputs.

### Deterministic artifacts

| File | SHA-256 | Bytes |
|---|---|---|
| `protocol-train.ids.txt` | `275ae66d16582418093a1f4500904faefedd5936bb5cf383c52be302e151172e` | 109 668 |
| `protocol-dev.ids.txt` | `d342950ae183e6c08bfeecaeacfb0e42aaf3751c12dec0baf0ca515922ca5e31` | 27 420 |
| `split-manifest.json` | `225b109ea5fa58476e98bdf050a42ca89f12c6df02b37a882dc09cdc958b3685` | 3 240 |
| `report.md` | `17a9a6f116b1277bc063ff53d0840e20cae1b034177d6d2ae014a6428ee20459` | 796 |

`runtime-environment.json`
(`0ed15fc3f717e1d316194021969ec6fc8288073de99a051980d94f6b86bc2c6e`, 168 bytes,
keys `note`/`platform`/`python`) is **runtime evidence, not part of the
deterministic scientific artifact set**.

### Independent determinism

A second materialisation into a fresh directory, from the same bytes, the same
committed materialiser and the same HEAD — **without** runtime output — produced
**exactly four files, all byte-identical**, with the same assignment digest and
the same memberships. Identical *bytes*, not merely identical counts.

### Persistent evidence

`/content/drive/MyDrive/UNMARK/preg1-uit-vsfc-internal-split/preg1-split-v1-66f4522a-7bd5d189`
— five files, post-copy SHA verification **PASS** for all five.

**Not copied:** raw UIT-VSFC, the reconstructed derived TRAIN csv, the
validation csv, the test csv, or any corpus-text artifact. No model weights were
loaded, no training performed, no downstream score computed.

### Derived-csv reconstruction

The Audit-022 derived csv was not retained, so its exact bytes were rebuilt from
a fresh official download whose ten raw-file digests matched the Audit-022
records. A first attempt failed visibly because it omitted the historical
five-digit zero-padded id serialisation, and wrote no authoritative output; the
recipe was then recovered from the historical notebook and is recorded in
[D-PREG1-011](../spec/decisions.md#d-preg1-011--conflicting-canonical-groups-are-excluded-whole).
The rebuilt TRAIN matched `a20c0f77…` exactly. **No digest was relaxed and no
corpus identity substituted.**

## Artifacts the run produces

`protocol-train.ids.txt`, `protocol-dev.ids.txt`, `split-manifest.json`,
`report.md`, and optionally `runtime-environment.json` — which is deliberately
**not** part of the deterministic scientific artifact.

The membership artifacts are **byte-deterministic** for a given input, code
version and seed: no timestamp, run uuid, hostname or absolute path enters them.
Overwrite is refused, and writes are staged so a failed run leaves nothing that
looks authoritative.

**No corpus text is written to any artifact, or into any exception message.**

## Recorded

[Audit 023](../audits/023-pre-g1-internal-split-materializer-and-fail-closed-contract.md)
was revised **in place** (no Audit 024) with the run directory, the execution
HEAD, the artifact digests, the assignment digest, the observed totals and class
counts, and confirmation of zero cross-part canonical leakage. **There was no
deviation from 9139 / 2285.**

## Boundaries

Official **validation** remains untouched measurement-dev. Official **test**
remains **sealed**. No head has been trained, no optimizer exists, no downstream
score exists, and Stage-1 is untouched.

[D-B3B0-002](../spec/decisions.md#d-b3b0-002--the-first-backbone-checkpoint-is-not-locked)
remains **OPEN**. The compiled proposal PDF remains **stale**.
