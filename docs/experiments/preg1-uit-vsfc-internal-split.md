# Pre-G1 UIT-VSFC internal split — protocol-train / protocol-dev

**Status: `IMPLEMENTATION READY — REAL MEMBERSHIP NOT YET OBSERVED`**

The mechanism is committed, hardened and tested. **No split has been
materialised.** This file records the input contract and the precommitted
expectations so that when the real run happens, its output can be checked
against something written down **beforehand** rather than described afterwards.

**No real membership ids appear here, and no artifact hashes** — those artifacts
do not exist yet, and inventing digests for them would defeat the purpose of the
exercise.

| | |
|---|---|
| **Implemented at** | `HEAD = 819d09f2df95ca57444a63c86363e614b44ce458` (baseline); this work uncommitted |
| **Audit** | [023](../audits/023-pre-g1-internal-split-materializer-and-fail-closed-contract.md) |
| **Decision** | [D-PREG1-014](../spec/decisions.md#d-preg1-014--internal-split-materialisation-is-fail-closed-and-mapping-order-independent) |
| **Artifact schema** | `preg1-split-v1` |
| **Real membership** | **not observed** |
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

## Artifacts the real run will produce

`protocol-train.ids.txt`, `protocol-dev.ids.txt`, `split-manifest.json`,
`report.md`, and optionally `runtime-environment.json` — which is deliberately
**not** part of the deterministic scientific artifact.

The membership artifacts are **byte-deterministic** for a given input, code
version and seed: no timestamp, run uuid, hostname or absolute path enters them.
Overwrite is refused, and writes are staged so a failed run leaves nothing that
looks authoritative.

**No corpus text is written to any artifact, or into any exception message.**

## What must be recorded after the real run

Revise **Audit 023 in place** — there will be no Audit 024 — with: the run
directory, the repository HEAD, the two id-file SHA-256s, the combined
assignment digest, the observed totals and class counts, and confirmation of
zero cross-part canonical leakage.

Any deviation from 9139 / 2285 is a **finding**, not a result to be accepted.

## Boundaries

Official **validation** remains untouched measurement-dev. Official **test**
remains **sealed**. No head has been trained, no optimizer exists, no downstream
score exists, and Stage-1 is untouched.

[D-B3B0-002](../spec/decisions.md#d-b3b0-002--the-first-backbone-checkpoint-is-not-locked)
remains **OPEN**. The compiled proposal PDF remains **stale**.
