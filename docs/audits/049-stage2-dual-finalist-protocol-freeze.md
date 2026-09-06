# Audit 049 - Stage-2 Dual-Finalist Protocol Freeze

**Scope:** freeze the Stage-2 protocol under which **both** Stage-1 finalists are
carried forward as separately reported UNMARK arms, with **no downstream
selection between them**.
**Date:** 2026-09-06
**Type:** documentation and specification freeze. **Nothing is trained, no
downstream row is read, and no runnable Stage-2 campaign is implemented.**

---

## 1. Executive verdict

**PASS — protocol frozen, nothing executed.**

| | |
|---|---|
| Downstream A-vs-B selection | **NONE** — option (c) adopted |
| D-S1B-001 | **INTACT** — no exception created |
| Stage-2 UNMARK arms | **2**, both frozen, both reported |
| Final adapter selected | **NO** |
| New tuning degrees of freedom on labelled data | **ZERO** — see §5 |
| Runnable campaign implemented | **NO** — deliberately not |
| Downstream data read | **NO** |
| Official UIT-VSFC TEST | **SEALED**, structurally unreachable |

---

## 2. Starting state

```
branch : main
HEAD   : fa573c064bc4b5a3b08c991389cabc9a492ef54a
status : clean (git status --porcelain produced no output)
```

Audit 048 is closed. Both finalists carry authoritative PASS verification and
bound digests; the freeze is complete; neither is selected.

| Arm | Finalist | Source stage | Seed | Update | checkpoint sha256 |
|---|---|---|---|---|---|
| `UNMARK-A` | A | `final_main` | 36930 | 3500 | `6773fbb59c7381ba8ddaa944302124a124f5b8a5cb0a5dbb1a5063f3db4a2a91` |
| `UNMARK-B` | B | `lr_pilot` | 21230 | 14500 | `9405bd76c04939641170cb71507ce8eb669eb2987016b86b495a403ceafcb9d2` |

---

## 3. The decision, and why it is the conservative one

The Stage-2 design review considered three options for resolving the A/B
question. Option **(c) — no downstream selection** was chosen.

[D-S1B-001](../spec/decisions.md) forbids UIT-VSFC from selecting **any** Stage-1
quantity and names `checkpoint selection` explicitly. Its reasoning has two
halves, and they fail differently under a DEV-only adjudication:

| D-S1B-001 reasoning | Under a `protocol-dev`-only adjudication |
|---|---|
| **Leakage into the headline number** — official validation is the set the Stage-2 result is reported from | **Mitigated.** `protocol-dev` is carved from official *train*; `measurement-dev` would stay untouched |
| **Circularity** — "tuning Stage-1 on a labelled downstream task would make Stage-1 a supervised search over that task, and the claim circular" | **Not mitigated.** A one-bit supervised choice is a far weaker search than tuning, but it is not zero |

Because the second half survives, adjudicating would have required **amending
D-S1B-001**. The review declined. The cost of option (c) is one additional
Stage-2 arm; the cost of the alternative would have been an exception to the rule
that keeps UNMARK's central claim non-circular.

**What is given up.** The project does not learn which finalist is better
downstream. That is deliberate: the Stage-1 evidence is genuinely ambiguous
(Audit 048 §4 — 36930@3500 wins the written local-stability rule while 21230 has
better whole-run trajectory behaviour), and resolving that ambiguity with
labelled downstream data is precisely the move D-S1B-001 exists to prevent.

---

## 4. What is inherited unchanged

The protocol reuses `preg1-protocol-v4` and its decision records **exactly**. No
dataset identity, split seed, head contract, optimiser setting or metric
definition is altered.

| Area | Inherited |
|---|---|
| Dataset | UIT-VSFC v1.0, sentiment, 3 labels, `{negative:0, neutral:1, positive:2}` |
| Splits | official train → 80% `protocol-train` / 20% `protocol-dev`, split seed `17486`, tag `UNMARK-PREG1-SPLIT-UITVSFC-v1`, existing deterministic group-aware splitter |
| Roles | `protocol-train` = clean head training; `protocol-dev` = head-checkpoint selection only; official validation = `measurement-dev`, reporting only; official TEST **SEALED** |
| Encoder | `vinai/phobert-base @ 01daacda68afe13d83023d16ec647239e344a1e6`, frozen, `eval()`, FP32, no mixed precision |
| Sequence | `max_length = 256`, truncation on, padding `max_length`, no word segmenter |
| Head | `Linear(768, 3, bias=True)`, xavier-uniform weight, zero bias, no hidden layer / dropout / LayerNorm / activation |
| Loss | cross-entropy, no class weights, no label smoothing, mean reduction |
| Optimiser | AdamW `(0.9, 0.999)`, eps `1e-8`, wd `0.01` weight / `0.0` bias, no amsgrad, constant schedule, no warmup, no clipping |
| Budget | batch `128`, grad-accum `1`, `drop_last=false`, **30 epochs, no early stopping** |
| Selection rule | highest macro-F1 → highest accuracy → **earliest** epoch |
| Metrics | macro-F1 primary, accuracy secondary, per-class F1 diagnostics |
| Seeds | measurement seeds `53148, 59945, 42941, 720, 9428`, tag `UNMARK-PREG1-MEASURE-v1` |
| Corruption | `unmark.corruption.corrupt`, `purpose=SCIENTIFIC`, keyed by `sample_id` and never by row order |

---

## 5. What this protocol adds — and the degrees of freedom it removes

Three things are newly *scoped*, and each is deliberately fixed rather than
tuned. **This protocol introduces no new tuning contact with labelled data.**

**Pooling — first token `<s>`, no pilot.** D-PREG1-005 scoped first-token pooling
to the pre-G1 diagnostic; D-G1-005 left Stage-2 pooling OPEN and recorded that
the Stage-1 masked-mean rule does **not** transfer downstream. This freeze
promotes first-token to an explicit Stage-2 scope. A pooling pilot is
**forbidden**: choosing pooling on labelled DEV would add exactly the kind of
downstream contact the option-(c) decision is avoiding, and masked-mean pooling
would additionally introduce a second robustness mechanism, confounding the
adapter's contribution.

**Head LR — `0.01`, inherited, no new pilot.** From the closed pre-G1 diagnostic
(D-PREG1-015): the shared-LR protocol froze `0.01`, and the precommitted own-LR
sensitivity independently selected `0.01` for Base-only. Inheriting it **removes**
a labelled-DEV tuning degree of freedom rather than adding one.

> **Recorded limitation.** `0.01` was selected over *adapter-free* representations
> (VANILLA and BASE_ONLY). The UNMARK arms present the head with adapter-modified
> representations at the same dimensionality and the same first-token pooling, so
> the setting is plausible but **not separately validated for this pathway**. The
> alternative — a new LR pilot on labelled DEV — was judged the worse trade. This
> is a limitation of the frozen protocol, not an oversight, and it applies
> **identically to both arms**, so it cannot bias the A/B contrast.

**Head seeds — the existing five, no new tag.** Earlier design drafting proposed a
new `UNMARK-STAGE2-ADJUDICATION-v1` seed tag. That is **not created**: no
adjudication exists, and inventing an adjudication-named seed stream would leave
a misleading artefact in the seed lineage.

---

## 6. Freezing and pairing guarantees

**Frozen.** Encoder `requires_grad=False` and `eval()`; Stage-1 adapter
`requires_grad=False` and `eval()`; only classification-head parameters train. No
encoder update, no adapter update, no A→B parameter reuse, no ensemble, no
restoration, no tokenizer change, no word segmenter. UNMARK base-stream, tone and
letter-channel semantics remain exactly the Stage-1 locked ones, and **token-grid
invariance is required**: corruption alters orthographic side-channel state, not
token positions.

**Paired heads.** For every seed `s`, each arm receives a freshly instantiated
head initialised with `s`. `build_head` resets the RNG immediately before
initialisation, so the two arms obtain **bit-identical** initial tensors
regardless of which arm runs first or what consumed RNG in between. Minibatch
ordering uses the existing dedicated-generator contract, not the ambient RNG. No
head weights are reused between arms.

**Selection hygiene.** Head checkpoint selection runs per arm and per seed on
**clean `FULL` `protocol-dev`** only. No corrupted `protocol-dev` score may
influence it — otherwise the head would be tuned for robustness, and the frozen
head would no longer be the "trained on clean, then frozen" object §5.2/§8.3
specify. Measurement on `measurement-dev` happens **only after** the head is
frozen.

---

## 7. Reporting, and what it may not do

Per arm and per condition (`FULL, P25, P50, P75, P100, STRIP_ALL`): macro-F1,
accuracy, per-class F1 diagnostics, with mean and sd over the five seeds; plus
descriptive per-seed paired `B − A` deltas.

Two robustness summaries are reported: `STRIP_ALL` macro-F1, and the equal-weight
mean macro-F1 over `P25, P50, P75, P100, STRIP_ALL`.

**These are reporting summaries only. They may not choose an arm.** There is no
winner rule, no tie-break, no no-decision margin, no significance test and no
p-value — the last two consistent with D-PREG1-010, which forbids them by design.
GRR is **not** an A/B-selection metric and is deferred until frozen `UPPER` /
`FLOOR` anchors exist.

`VARIANT` is excluded from the condition grid: it is unimplemented and
fail-closed, because emitting only its NFD half would misrepresent the condition.

---

## 8. Post-measurement lock

Once any `measurement-dev` result from these arms is observed:

* **A** may not be dropped because **B** looks better, and vice versa;
* no Stage-1 checkpoint may replace either arm;
* pooling, LR, head architecture, seeds and the epoch rule may not change in
  response to the results;
* neither `measurement-dev` nor TEST may select A vs B;
* if official TEST is ever opened under a later fully frozen protocol, **both**
  surviving UNMARK arms are evaluated.

---

## 9. Implementation

### 9.1 `docs/spec/stage2-dual-finalist-protocol.json` (new)

The machine-readable freeze, in the `classification` / `value` convention
established by `docs/spec/stage1-final-freeze.json`. Sections: `protocol`,
`dataset`, `splits`, `arms`, `frozen_representation`, `pooling`, `head`,
`optimization`, `seeds`, `head_checkpoint_selection`, `measurement`, `reporting`,
`post_measurement_lock`, `state`.

The selection-safety fields are represented explicitly rather than by omission:
`winner_rule`, `tie_break` and `no_decision_margin` are present and `null`;
`summaries_may_select_arm`, `downstream_may_select_a_vs_b`,
`d_s1b_001_exception_required`, `new_lr_pilot_permitted` and `pilot_permitted`
are present and `false`. A future reader can see that these were decided, not
forgotten.

### 9.2 `docs/spec/decisions.md`

**D-S2-001** appended (84 insertions, 0 deletions — strictly append-only). The
`D-S2` namespace was confirmed unused before it was chosen.

### 9.3 Not implemented, deliberately

`unmark/evaluation/stage2_adjudication.py` is **not** created. The word
"adjudication" must not name the executable protocol, because no winner is
selected and the name would misdescribe it. When a runner is written it should
carry a dual-finalist name.

Two implementation gaps remain open for that future task and are recorded here so
they are not discovered late:

1. **No UNMARK pathway exists.** `SystemPathway` contains only `VANILLA` and
   `BASE_ONLY`; `unmark/evaluation/preg1_head.py` states plainly "This is not
   UNMARK. No tone channel, no letter channel, no adapter." A frozen-adapter
   pathway must be built.
2. **No downstream corruption applier exists** in `unmark/evaluation/`. The
   corruption engine itself is complete and reusable; wiring it into downstream
   evaluation is new work.

---

## 10. Checks run

Structural and specification checks only. **No downstream data was loaded, no
model was constructed, and no torch computation ran.**

* The freeze artifact parses, and every field carries both a `classification`
  from the allowed set and a `value`.
* Every inherited value was compared against the **imported constant**, not a
  re-typed literal: dataset name and label count, label mapping, split seed and
  fractions, encoder revision, `max_length`, adapter parameter count, head shape
  and initialisers, AdamW betas, batch size, epochs, early-stopping flag, the
  five measurement seeds, and the pooling enum name. All matched.
* Both arms' `checkpoint_sha256` values were compared against `FINALIST_A` and
  `FINALIST_B` in `unmark/stage1/finalists.py`. Both matched.
* Selection-safety invariants were asserted: no winner rule, no tie-break, no
  margin, summaries may not select, no D-S1B-001 exception, no LR pilot, no
  pooling pilot, arm count 2, no ensemble, TEST sealed.

```
git diff --check
```

produced no output.

---

## 11. Remaining protocol ambiguity

Recorded rather than silently resolved:

1. **Head LR provenance.** `0.01` is inherited from adapter-free representations
   (§5). Symmetric across arms, so it cannot bias the contrast, but it is not
   validated for the UNMARK pathway.
2. **`measurement-dev` is used once per arm, for reporting.** With two arms it is
   read twice. No selection is performed on it, so no selection leakage occurs,
   but the set is no longer touched exactly once. The alternative — reporting only
   one arm — is the selection this decision refuses.
3. **G1 pass-threshold precision remains OPEN** (D-G1-004): §7's "within ≈1 point"
   names neither the metric nor what "≈" tolerates. Not needed to run this
   protocol, but it must be pinned before a G1 pass/fail claim is made, and with
   two arms the threshold's application to *both* needs stating.
4. **GRR anchors do not yet exist** for these conditions on `measurement-dev`;
   GRR is deferred (§7).
5. **Two implementation gaps** (§9.3): the UNMARK pathway and the downstream
   corruption applier.
6. **Backbone generalisation.** §6.1 names ViSoBERT as a later ablation. With two
   UNMARK arms, a second backbone would double again; scope needs a decision
   before that grid is planned.

---

## 12. Final state

```
STAGE1_TRAINING=CLOSED
STAGE1_CANDIDATE_GENERATION=CLOSED
FINALIST_COUNT=2
FINALIST_A=36930@3500
FINALIST_B=21230@14500
FINAL_ADAPTER_SELECTED=NO
ADJUDICATION=CLOSED_WITHOUT_SELECTION
STAGE2_UNMARK_ARM_COUNT=2
DOWNSTREAM_MAY_SELECT_A_VS_B=NO
DOWNSTREAM_RESULTS_SEEN=NO
DOWNSTREAM_TEST=SEALED
STAGE2_STARTED=NO
```

D-S1B-001 remains intact and required no exception. Nothing was trained, no
downstream row was read, no runnable campaign was implemented, and nothing was
committed or pushed.

---

## 13. Exact next allowed step

**Not** "run Stage 2." In order:

1. **Independently review this freeze** — `docs/spec/stage2-dual-finalist-protocol.json`,
   D-S2-001 and this audit — and commit it. The author performs all commits and
   pushes.
2. **Implement the two missing pieces** (§9.3) under a dual-finalist name: the
   frozen UNMARK pathway and the downstream corruption applier, with tests, and
   *still* without reading any downstream row beyond what the split materialiser
   already verified.
3. Only then may head training begin, on clean `protocol-train` only.
