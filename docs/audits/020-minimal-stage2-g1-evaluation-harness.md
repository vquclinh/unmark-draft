# Audit 020 — minimal Stage-2 / G1 evaluation harness

| | |
|---|---|
| **Audit id** | 020 |
| **Date (UTC)** | 2026-08-20 |
| **Scope** | Minimal reusable downstream evaluation infrastructure for the Vanilla vs Base-only diagnostic |
| **Repository state** | `HEAD = 6a1724c`; this work uncommitted |
| **Predecessors** | [018](018-stage1-objective-and-data-path-implementation.md), [019](019-stage1-real-phobert-diagnostic-closure.md) |
| **Phase** | Stage-2 / G1 harness |
| **Type** | **Infrastructure only.** No experiment run, no dataset, no training |
| **Revised** | 2026-08-20 — review found extraction silently inheriting Stage-1 pooling into an OPEN Stage-2 decision, and an over-claim that a Base-only FAIL would be decisive against the input-level design. Both repaired. Verdict unchanged. |
| **Revised (2)** | 2026-08-20 — documentation consistency: §D/§E, §C and §A brought into line with the repaired code; §N split by when values are needed. |
| **Revised (3)** | 2026-08-20 — bookkeeping: §C's stale test count and decision range corrected. Documentation only. |

---

## A. VERDICT

**PASS — MINIMAL G1 HARNESS IMPLEMENTED; NO SCIENTIFIC VALUE SELECTED, NO
EXPERIMENT RUN**

The pathway contracts, metrics, GRR and leakage guards exist and are tested.
**1999 local tests pass, 56 skip** because torch is absent by design.

**No task or dataset was selected**, **no head hyperparameter was chosen**, and
**the real measurement was not run.** §5's open-items table names the
classification-head concrete values as blocking **G1**, and they remain open.
**D-B3B0-002 remains OPEN.** The compiled PDF **remains stale**.

**Two issues found in review and repaired before commit.** Neither resolves a
scientific value:

| Issue | Was | Now |
|---|---|---|
| **Stage-2 pooling silently inherited** | extraction called `masked_mean_non_special` — the **Stage-1** §4.6 rule — returning pooled `[N, d]`, while `HeadConfig.pooling` was required but **never read** | extraction returns **unpooled** `[N, L, d]` + masks; the one pooling function is `TEST_ONLY_masked_mean_pool` and **raises** on `SCIENTIFIC` (D-G1-005) |
| **Over-claimed failure semantics** | "a FAIL would be decisive against the input-level design" | a Base-only FAIL does **not** equal a G1 FAIL; it measures the **burden** of the grid swap, not a ceiling on the trained adapter (D-G1-001) |

The first was a real defect: it answered a §5.2 spec-lock item by implementation
convenience.

### Four things this audit keeps apart

1. **Stage-1 representation pooling** — masked mean over non-special content
   tokens. **LOCKED**, §4.6, and **for the Stage-1 alignment objective only**.
2. **Stage-2 task-head pooling** — **OPEN**, §5.2 pins it during spec lock.
   Does **not** inherit (1).
3. **Pre-G1 Vanilla-vs-Base-only clean-path burden diagnostic** — what this
   harness supports. Descriptive; no pass threshold.
4. **Full proposal G1** — §7: attach the fusion layer, train briefly, force the
   gate towards identity, evaluate on `FULL`, "within ≈1 point". Unchanged and
   still required.

---

## B. H1 / G1 SOURCE CONTRACT

Read from the editable proposal, not from the task description.

**H1 (§2).** *"A frozen pretrained encoder accepts a synthesised input embedding
built from orthographic channels without substantial loss on fully diacritized
input."*

**G1 (§7).** *"Attach the fusion layer, train briefly on a small corpus, force
the gate towards identity, evaluate on one classification task with `FULL` input.
**Pass:** within ≈1 point of the unmodified model. **Fail:** the encoder rejects
the synthesised embedding distribution."*

**Why Vanilla vs Base-only is the right first measurement (§4.5).** *"Since
`e_i = Emb_θ(b_i)` is computed from the **stripped** base stream, `g_i → 0`
yields `E_θ(T(b(x)))`, not `E_θ(T(x))`. The gate recovers the **base-only
pathway**, not the original model. Whether clean-input performance survives that
substitution is not a structural guarantee at all — it is exactly hypothesis H1,
and exactly what G1 measures."*

| Item | Status | Source |
|---|---|---|
| What G1 compares | **LOCKED** — a system's clean-`FULL` score against the unmodified model | §7 |
| Metrics | **LOCKED** — macro-F1 and accuracy per task and condition | §6.5 |
| GRR formula | **LOCKED** — `(S_system − S_FLOOR) / (S_UPPER − S_FLOOR)` | §6.5 |
| GRR anchors | **LOCKED** — `UPPER` clean/unmodified, `FLOOR` corrupted/unmodified | §6.4 |
| Head trained on clean only | **LOCKED** | §5.2, §8.3 |
| Head protocol identical across systems | **LOCKED** | §5.2, §8.3 |
| Split discipline | **LOCKED** | §5.4 |
| Encoder frozen | **LOCKED** | §5.1 |
| Corruption condition set | **LOCKED** | §6.3 |
| Seeds — minimum count | **LOCKED** — at least three | §6.6 |
| G1's condition | **LOCKED** — `FULL` | §7 |
| **Which task/dataset** | **OPEN** | §6.2 names four *categories*; §5 table; §13 item 2 |
| **Head architecture, hidden size** | **OPEN — explicitly blocks G1** | §5.2, §5 table |
| **Stage-2 head pooling** | **OPEN** — and §4.6's Stage-1 masked mean does **not** transfer | §5.2, §13 item 4 |
| **Head optimizer, LR, batch size, epochs, early stopping** | **OPEN** | §5.2 |
| **Seed list** | **OPEN** | §5.2 requires a pinned list; none given |
| **`max_length`** | **OPEN** | §5.3 pins one per task; no value |
| **G1 pass-threshold precision** | **OPEN** | §7's "≈1 point" — metric unstated |
| **GRR degenerate denominator** | **OPEN** | §6.5 gives no policy |
| `RESTORE`, `ALIGN` | **NOT_APPLICABLE** here | §6.4; out of scope |
| Representation-level and cost metrics | **NOT_APPLICABLE** here | §6.5; not needed for this diagnostic |

**No OPEN item was turned into an implementation default.**

---

## C. FILES CHANGED

| File | State | Why | Resolves an OPEN value? |
|---|---|---|---|
| `unmark/evaluation/contracts.py` | **new** | `Split`, `SystemPathway`, `TaskExample`, `TaskSplit`, `HeadConfig`, `EvaluationRunConfig`, LOCKED/OPEN registers, leakage guards. Torch-free | No |
| `unmark/evaluation/metrics.py` | **new** | Accuracy, per-class scores, macro-F1, GRR. Pure Python | No |
| `unmark/evaluation/pathways.py` | **new** | Vanilla/Base-only text selection, tokenization, **frozen unpooled hidden-state extraction + masks**, `HeadBinding`, and the separately-gated `TEST_ONLY_masked_mean_pool`. Torch imported lazily | No |
| `unmark/evaluation/__init__.py` | **new** | Re-exports the torch-free surface only | No |
| `tests/test_evaluation_harness.py` | **new** | **61 local + 10 torch-gated** (71 collected) | No |
| `docs/spec/decisions.md` | modified | **D-G1-001…005** and the category index. D-G1-005 (Stage-2 pooling stays OPEN) came from the pooling repair and Audit 020 relies on it | No — each names what stays OPEN |
| `docs/audits/020-…md` | **new** | This audit | No |

`unmark/evaluation/` is a new package because the metrics and task contracts are
reusable beyond G1 (the full §6 grid needs the same GRR and macro-F1);
`unmark/gates/` holds gate-specific code such as the G−1 smoke test.

---

## D. VANILLA PATHWAY

```
canon(x) → frozen tokenizer → frozen encoder → unpooled hidden states [N, L, d]
                                               + attention_mask, special_tokens_mask
```

**Extraction stops at hidden states.** No pooling is applied, because Stage-2
head pooling is OPEN (§5.2). The masks travel with the states so that whichever
rule is eventually pinned can be applied correctly — **head pooling happens only
after `HeadConfig.pooling` is scientifically resolved**, and never here.

This is §6.4's unmodified model: `UPPER` at `FULL`, `FLOOR` at a corrupted
condition. The same pathway supplies **both GRR anchors**, which is what makes
§6.5 and the per-condition form identical (§H).

**Canonicalisation, and why.** Both pathways canonicalise first, consistent with
D-B2-004 (corruption operates on `canon(x)`) and D-S1A-001. Two inputs differing
only in NFC/NFD form or tone placement are *the same example* and must not get
different scores; letting the raw form through would smuggle in the `VARIANT`
condition (§6.3), which is a separate experimental axis.

---

## E. BASE-ONLY PATHWAY

```
canon(x) → b(x) → the same frozen tokenizer → the same frozen encoder
                → unpooled hidden states [N, L, d] + masks
```

**No adapter. No tone channel. No letter channel. No restoration.** And, exactly
as for Vanilla, **no pooling at extraction** — both pathways stop at hidden
states so that neither inherits the Stage-1 §4.6 rule as a Stage-2 decision.

### Is it equivalent to the `g → 0` architectural limit?

**Yes, numerically — and the evidence is real-model, not reasoning
(D-G1-002).** From B4B run `20260820T081554Z`:

* `model(input_ids=…)` vs `model(inputs_embeds=Emb(input_ids), position_ids=authoritative, attention_mask=…)`
  → `max_abs_diff = 0.0` **exactly**, including padding;
* the forced `g := 0` wiring identity gives `z = g⊙f + (1−g)⊙e = e`.

Composing: the adapter at `g = 0` produces exactly `Emb_θ(T(b(x)))`, and feeding
that is exactly equivalent to feeding the ids. So running the frozen encoder
directly is not an approximation of the architectural definition — it is the same
computation, with less machinery to go wrong.

### The caveat that must travel with that claim

**`g = 0` is not attainable** by the locked sigmoid gate (D-B4A-004); it is a
limit. `BASE_ONLY` therefore implements the **architectural limit** §4.5
describes — **not** the behaviour of any initialised or trained adapter.

**It is not UNMARK, and the code says so.** §4.5: "the gate recovers the
**base-only pathway**, not the original model." Calling it UNMARK would claim a
result about a module that is not present. A test asserts the naming.

### What a Base-only result does and does not mean

Base-only runs `b(x)` with **no channels and no adapter**. The real UNMARK clean
pathway is `base + tone + letter -> trainable fusion -> frozen encoder`, and both
the channels and the adapter can recover information the bare base grid loses.

So this measures the **burden** created by replacing the original clean token
grid with the stripped base grid. It is a *pre-G1 clean-path burden diagnostic*:

* a **PASS does not discharge G1** — §7 additionally attaches the fusion layer;
* a **FAIL does not automatically equal a G1 FAIL**, and it is **not** an upper
  bound on the trained adapter.

An earlier draft of this audit said a failure "would be decisive against the
input-level design". That was an over-claim and is retracted; the proposal
defines G1's fail criterion for the *fusion-attached* measurement, not for this
one.

---

## F. HEAD-TRAINING CONTRACT

**Locked, and implemented as guards:**

* **Clean data only.** §5.2: "The head is trained on clean data only… then frozen
  and evaluated on every condition", chosen so a reviewer cannot attribute a gain
  to the head having seen corrupted labels. §8.3 repeats it.
  `HeadBinding.require_clean_training` refuses any condition but `FULL`.
* **Per-system heads, identical protocol.** §8.3: "Run identically for all five
  systems"; §5.2: "one architecture, identical across all five systems". Combined
  with §4.5/§6.6 — `UPPER` and the adapted pathways are *different input
  pathways* — this means each system trains its **own** head through its **own**
  clean pathway under the **same** protocol. `RepresentationSet.require_same_pathway`
  refuses cross-pathway reuse; `HeadConfig.identical_protocol_to` checks the
  protocol matches.
* **Train split only.** §5.4. Enforced on both `TaskSplit` and `RepresentationSet`.

**Deliberately not implemented: the head trainer itself.** Every value it needs —
architecture, **pooling**, optimizer, learning rate, batch size, epochs, early
stopping — is OPEN, and §5's table names them as blocking **G1**. Writing a
trainer would mean choosing them. `HeadConfig` therefore *carries* a decision
with **every field required**, and `EvaluationRunConfig(purpose=SCIENTIFIC, …)`
**cannot be constructed** until they are resolved.

**`HeadConfig.pooling` is load-bearing, not decorative.** Because extraction
returns unpooled `[N, L, d]`, a head *must* apply this rule — there is no pooled
vector lying around for it to inherit. A `SCIENTIFIC` configuration additionally
rejects any pooling name prefixed `TEST_ONLY_`. Tests assert both.

---

## G. METRICS

§6.5 "Task level": **macro-F1 and accuracy per task and condition**. Both
implemented in pure Python — no torch, no sklearn — so they are deterministic and
exactly testable without downloads.

Macro-F1 is the unweighted mean of per-class F1, so a rare class counts as much
as a common one, which is the point on the imbalanced tasks §6.2 names. A class
that is **predicted but never present** is included in the average rather than
ignored, so a model inventing a label is penalised.

§6.5's "Representation level" and "Cost" metrics are **not** implemented: they
are not needed for this diagnostic, and stubbing them would suggest coverage that
does not exist.

**No score is computed by this harness yet**, because computing one needs a head,
which needs the pooling and architecture §5.2 leaves OPEN. The metrics are pure
functions over predictions and labels, ready for whatever head is eventually
pinned.

---

## H. GRR FORMULA VERIFICATION

**Verified against the editable proposal, not memory.** §6.5:

```
GRR = (S_system − S_FLOOR) / (S_UPPER − S_FLOOR)
```

§6.4 fixes the anchors: `UPPER` = "Clean input, unmodified model", `FLOOR` =
"Corrupted input, unmodified model". **Both are the VANILLA pathway.**

**Reconciliation with the per-condition form.** Substituting the anchors gives
`[S(system,c) − S(vanilla,c)] / [S(vanilla,FULL) − S(vanilla,c)]` — **identical**
to the candidate form in the task brief. **There is no discrepancy**, and nothing
contradictory was implemented. The implementation follows §6.5 with the anchors
named in the signature so the identity cannot be lost.

**Required numerical case, verified:**

| `S_vanilla,FULL` | `S_vanilla,c` | `S_system,c` | GRR |
|---|---|---|---|
| 84 | 60 | 72 | **0.5** exactly |
| 84 | 60 | 60 | 0.0 |
| 84 | 60 | 84 | 1.0 |
| 84 | 60 | 90 | **1.25 — not clamped** |
| 84 | 60 | 48 | **−0.5 — not clamped** |

**No clamping**, because §6.5 prescribes none and clamping would erase two
informative outcomes: beating clean vanilla, and doing worse than the corrupted
unmodified model. A structural test asserts the metric module calls no
`clamp`/`clip`/`min`/`max` and contains no stray float constants.

**Degenerate denominator — honestly open.** When `S_UPPER == S_FLOOR`, corruption
cost the unmodified model nothing, so "the fraction of the gap recovered" is not
a meaningful quantity rather than a large one. §6.5 defines **no** epsilon, clamp
or fallback, so **none was invented**: `gap_recovery_rate` raises `UndefinedGRR`,
`is_grr_defined` lets a caller check first, and
`grr_degenerate_denominator_policy` is registered OPEN.

---

## I. SPLIT / LEAKAGE INVARIANTS

| Risk | Guard |
|---|---|
| Training on dev/test | `Split.may_train_head`; `TaskSplit.require_trainable`; `RepresentationSet.require_trainable`; `HeadBinding` refuses a non-train split at construction |
| Head from another pathway | `RepresentationSet.require_same_pathway` — refuses on pathway **or** task mismatch |
| Head trained on corrupted data | `HeadBinding.require_clean_training` |
| Mixed ids/labels in cached representations | `RepresentationSet` validates counts; `EncodedSplit`/`RepresentationSet` carry `(task, split, pathway)` identity |
| Duplicate ids within a split | rejected by `TaskSplit` |
| Same id across splits of one task | `assert_disjoint_splits` — the cheapest leak, and invisible once representations are cached under those ids |
| Mutating the frozen encoder | extraction runs under `no_grad` with `encoder.eval()`; a test asserts parameters, `requires_grad` and `grad` are all unchanged |

Deliberately not built: caching layers, experiment tracking, a run registry.

---

## J. LOCAL ML-FREE BOUNDARY

`unmark.evaluation` re-exports only contracts and metrics and is **torch-free**;
`pathways.py` imports torch **lazily, inside functions**, and is not re-exported
from `__init__`. Tests assert both. Torch-gated tests use the established
`pytest.mark.skipif` mechanism, **per test rather than per module**.

`.venv` remains ML-free: no torch, transformers, datasets, sklearn or numpy. No
model or dataset was downloaded; no network was accessed.

---

## K. TEST EVIDENCE

```
.venv/bin/python -m pytest -q -p no:cacheprovider
1999 passed, 56 skipped in 8.99s
```

Baseline before this task: 1938 passed, 46 skipped. `tests/test_evaluation_harness.py`
holds **61 local + 10 torch-gated**. The count rose by 8 over the pre-repair
figure: seven new pooling-openness tests, plus three torch-gated ones for the
unpooled contract, minus two that tested the now-removed pooled extractor.

Covering: the GRR formula string, the 84/60/72 case, both endpoints,
non-clamping in both directions, undefined-on-zero-denominator, no
epsilon/clamp structurally; accuracy and macro-F1 including the macro-vs-accuracy
divergence on an imbalanced case and the predicted-but-absent class; only two
pathways existing; Base-only not named UNMARK; neither pathway using channels;
stripping only for Base-only; NFC/NFD canonicalisation agreeing; no restoration
call; split permissions; head-training refusal on dev/test; duplicate and
cross-split ids; `HeadConfig` having **no defaults at all** by signature
inspection; every head value registered OPEN; protocol-identity comparison;
clean-only binding; no dataset name, URL or `datasets` import anywhere;
`SCIENTIFIC` config refusing to construct; `DIAGNOSTIC` config labelled;
torch-free package; no optimizer; no second pooling; frozen `no_grad`
extraction.

**Added by the pooling repair.** Stage-2 pooling registered OPEN and documented
as not transferring; the **Stage-1 rule itself unchanged**; `encoder_hidden_states`
containing no `masked_mean` call by AST inspection; **exactly one** function in
the harness reaching `masked_mean_non_special`, and it being
`TEST_ONLY_masked_mean_pool`; `HeadConfig.pooling` required and non-empty; a
`SCIENTIFIC` config rejecting a `TEST_ONLY_` pooling name; and **no pooling
option invented** (no `PoolingStrategy`, `CLS_TOKEN`, `MAX_POOL`,
`ATTENTION_POOL`).

Torch-gated: the two pathways producing **different token ids** (the diagnostic
rests on this), identity propagation, extraction returning **3-D unpooled**
states with `requires_grad=False`, masks travelling with them, the scientific
path **refusing** to reach masked mean, the TEST-only pooler working under
`DIAGNOSTIC`, the encoder provably unmutated, cross-pathway head reuse refused,
dev representations refusing to train, and task mismatch refused.

**Test-only values are named `TEST_ONLY_…`** and are not a protocol — the harness
enforces that rather than relying on the name alone.

---

## L. OPEN VALUES BEFORE REAL G1

Nothing below was chosen. **§N splits these by when they are needed**: §N.A must
be resolved before the descriptive pre-G1 burden diagnostic; §N.B may stay open
until before full G1. `HeadConfig` requires every field;
`EvaluationRunConfig(purpose=SCIENTIFIC, …)` cannot be constructed until
`resolved_values` covers `SCIENTIFIC_REQUIRED_VALUES`.

* **Task / dataset**, and G1's "one classification task" — §6.2 names four
  *categories* only.
* **Head architecture and hidden size** — §5's table names these as blocking
  **G1**.
* **Stage-2 head pooling** — §5.2 pins it during spec lock. §4.6's masked mean is
  the **Stage-1 alignment** rule and does not transfer; the harness cannot pool
  on a scientific path until this is resolved (D-G1-005).
* **Head optimizer, learning rate, batch size, epochs, early-stopping patience.**
* **Seed list** — §6.6 fixes a minimum of three; no list is given.
* **`max_length`.**
* **Checkpoint selection** for the head.
* **G1 pass-threshold precision** — §7's "within ≈1 point" does not say whether
  the point is accuracy or macro-F1, nor what "≈" tolerates across the ≥3 seeds
  §6.6 requires. **It must be pinned before the *full-G1* result is observed** —
  and it is **not** a prerequisite for the pre-G1 burden diagnostic (§N.B).
* **Any threshold for the pre-G1 burden diagnostic** — the proposal defines
  **none**, and §7's "≈1 point" is stated for the fusion-attached measurement.
  It is **not borrowed** here. The burden diagnostic is **descriptive**: report
  the clean score gap with uncertainty across the required seeds. Any gating rule
  must be a researcher decision recorded **before** the numbers exist.
* **GRR degenerate-denominator policy.**
* **Backbone finalisation** — D-B3B0-002.

---

## M. SCIENTIFIC NON-ACTIONS

* **No model download** — none, locally or otherwise.
* **No real dataset download**, and no dataset name appears in the library.
* **No real downstream training** — no head trainer exists.
* **No Stage-1 training.**
* **No optimizer** — none constructed anywhere; asserted structurally.
* **No Stage-1 HPO**, no lambda search, no corpus selection.
* **No scientific value silently chosen** — every OPEN item is a required
  argument or an explicit register entry.
* **No RESTORE, no ALIGN, no second backbone, no full task grid.**
* **D-B3B0-002 OPEN.**
* **Compiled PDF stale** — unchanged from the v1.4 source changes; not
  regenerated and not claimed synchronised.

---

## N. NEXT STEP

**Select and lock the smallest scientifically valid pre-G1 pilot protocol**, then
run the descriptive Vanilla-vs-Base-only burden diagnostic in Colab.

An earlier draft of this section listed "the pass-threshold precision" among what
must be resolved first. **That was wrong** and contradicted this audit's own
finding: §7's "within ≈1 point" belongs to the **fusion-attached** G1, and the
pre-G1 burden diagnostic is **descriptive with no proposal-defined threshold**.
Requiring it here would have imported a criterion the proposal does not state for
this measurement.

### A. Must be resolved **before** the pre-G1 burden diagnostic

These are what the executable run actually needs, and none is chosen here:

* **task / dataset** — §6.2 names four categories only;
* **head architecture** and hidden size;
* **Stage-2 head pooling** — §5.2 pins it during spec lock; §4.6's masked mean is
  the Stage-1 rule and does not transfer (D-G1-005);
* **head optimizer**;
* **head learning rate**;
* **batch size**;
* **epochs / steps**;
* **early-stopping policy**, if one is used;
* **seed list** — §6.6 fixes a minimum of three; the list itself is unpinned;
* **`max_length`**;
* **head checkpoint-selection policy**;
* **reporting and statistical aggregation** for comparing Vanilla against
  Base-only — the diagnostic is descriptive, so how the gap and its uncertainty
  are summarised across seeds is itself a choice that should be stated in
  advance rather than after the numbers appear.

### B. May remain OPEN until **before full G1**

* the precise interpretation of §7's **"within ≈1 point"**;
* **which task metric** controls that full-G1 threshold — accuracy or macro-F1;
* any **full-G1-specific aggregation rule** for applying it across the ≥3 seeds.

These **must be precommitted before the full-G1 result is observed**, but they
are not prerequisites for the descriptive pre-G1 measurement. The proposal does
not say otherwise.

### C. Pre-G1 threshold

**None exists.** The proposal defines no Base-only-specific pass criterion, and
**none is invented**. If a threshold or risk band is eventually wanted, it must
be a researcher decision recorded **before** the real numbers are seen.

### Roadmap position

**It is not yet Stage-1 HPO.** The order set in D-S1A-009 still holds: harness →
Vanilla vs Base-only → precommit Stage-1 HPO → training runner → regenerate PDF →
PRE-TRAIN audit → only a PASS permits scientific Stage-1 training.

**Both directions matter, and neither is a gate.** A PASS does **not** discharge
G1: §7 additionally attaches the fusion layer and trains briefly with the gate
pushed towards identity. A FAIL does **not** equal a G1 FAIL either — Base-only
has no channels and no adapter, so it measures the burden of the grid swap, not a
ceiling on the trained module (D-G1-001).

What the diagnostic buys is **cheap information early**: it sizes the burden the
adapter would have to overcome, before any Stage-1 HPO is spent. Reporting it
descriptively, with uncertainty across seeds and no borrowed threshold, is the
honest use of it.

---

## O. TASK-END SELF-AUDIT

Verified at the close of the final bookkeeping pass. Both §C figures were checked
against reality — pytest collection and the decision log — rather than copied
from §K.

| Check | Result |
|---|---|
| §C says **61 local + 10 torch-gated** | **OK** — `tests/test_evaluation_harness.py` collects **71**: 61 passed, 10 skipped |
| §C accurately includes **D-G1-005** | **OK** — `decisions.md` contains D-G1-001 … D-G1-005; the row now reads `D-G1-001…005` |
| §A says **1999 passed, 56 skipped** | **OK** — matches the rerun |
| Historical **1938 / 46** baseline still labelled historical | **OK** — §K reads "Baseline before this task", untouched |
| §D/§E describe **unpooled** `[N, L, d]` + masks | **OK** |
| Stage-2 scientific pooling still **OPEN** | **OK** — `head_pooling` in the OPEN register |
| Stage-1 masked mean unchanged | **OK** — `STAGE1_POOLING` untouched |
| `SCIENTIFIC` reaches no implicit masked mean | **OK** — `encoder_hidden_states` contains no pooling call |
| `HeadConfig.pooling` load-bearing | **OK** — required, and extraction is unpooled |
| `TEST_ONLY_masked_mean_pool` blocked from `SCIENTIFIC` | **OK** — raises |
| Pre-G1 diagnostic descriptive, **no threshold** | **OK** — §N.C |
| Base-only ≠ UNMARK, ≠ ceiling; PASS ≠ discharge, FAIL ≠ G1 FAIL | **OK** |
| GRR unchanged — `84/60/72 → 0.5`, unclamped | **OK** |
| **No scientific code changed** in this pass | **OK** — documentation only |
| **No OPEN scientific value resolved** | **OK** |
| Stage-1 mathematics unaltered | **OK** — no changed files under `unmark/stage1/` or `unmark/modeling/` |
| D-B3B0-002 **OPEN** | **OK** |
| Compiled PDF **stale** | **OK** — not regenerated, not claimed synchronised |
| No banned git operation | **OK** |
| All changes unstaged | **OK** — 0 staged |
| `git diff --check` | **OK** — clean |
| Full local suite | **OK** — 1999 passed, 56 skipped |

---

## P. GIT STATE

`HEAD = 6a1724c`.

```
 M docs/spec/decisions.md
?? docs/audits/020-minimal-stage2-g1-evaluation-harness.md
?? tests/test_evaluation_harness.py
?? unmark/evaluation/
```

`git diff --check` is clean. Everything is left **unstaged**.

**No prohibited git operation was used.** No `add`, `commit`, `push`, `tag`,
`stash`, `reset`, `checkout` or `restore`. No unrelated researcher work was
touched.

```text
AUDIT FILE WRITTEN: docs/audits/020-minimal-stage2-g1-evaluation-harness.md
OTHER AUDIT MODIFICATIONS MADE: NO
COMMIT CREATED: NO
```
