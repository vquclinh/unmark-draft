# Audit 016 — B4B real-model provenance and position-id repair

| | |
|---|---|
| **Audit id** | 016 |
| **Date (UTC)** | 2026-08-20 |
| **Scope** | Repair the model-revision verifier; close D-B4B-002; enforce authoritative position ids |
| **Repository state** | `HEAD = 749cc39`; this work uncommitted |
| **Predecessors** | [014](014-b4a-neural-adapter-contract-preflight.md), [015](015-b4b-neural-adapter-implementation-preflight.md) |
| **Phase** | Phase 1 / B4B integration repair |
| **Type** | **Integration repair.** No training, no weights loaded locally |
| **Revised** | 2026-08-20 — review found the verified-position scope too broad (family-wide `{"roberta"}`) and caller-supplied `position_ids` unchecked. Both hardened **before** the real rerun. Verdict unchanged; no adapter mathematics touched. |

---

## A. VERDICT

**PASS — B4B REAL-MODEL RERUN REQUIRED**

The first real PhoBERT run passed **21 of 22** checks. The single failure was a
defect in the probe's own provenance *reporter* — not the model, not the weights,
not the adapter. Both gaps this task addresses are repaired locally, and **1870
tests pass, 32 skip** because torch is absent by design.

**B4B is NOT complete.** The repaired probe has not run. It must not be declared
complete merely because the failing check was fixed: the repaired verifier has to
genuinely prove both tokenizer and model resolve to
`01daacda68afe13d83023d16ec647239e344a1e6`, and every previously passing neural
check must still pass.

**Nothing was trained.** No optimizer, no `optimizer.step()`, no parameter
update, no dataset, no checkpoint. The local `.venv` remains ML-free.

**Two scope/safety issues found in review of this audit and repaired before the
rerun.** Neither touches the adapter mathematics:

| Issue | Was | Now |
|---|---|---|
| **Position scope too broad** | any checkpoint with `model_type == "roberta"` was cleared | a `VerifiedPositionProfile` matching **checkpoint + model type + model class**; only `vinai/phobert-base` is registered |
| **Supplied `position_ids` trusted** | an explicitly passed tensor was honoured | it is **validated** against the authoritative tensor; a mismatch raises `PositionContractViolation` |

The first was a genuine overreach on my part: this audit's own §S said "any
second backbone needs its own measurement" while the code cleared an entire
family. The two statements could not both be true.

---

## B. FILES CHANGED

| File | Change |
|---|---|
| `scripts/b4b_phobert_adapter_probe.py` | `verify_model_revision`, `cached_artifact_path`, `hub_cache_root`, `read_main_ref`; position-helper check; `provenance.json`; gradient path now omits `position_ids` |
| `unmark/modeling/adapter.py` | `UnsupportedPositionSemantics`, `PositionContractViolation`, `VerifiedPositionProfile` + `VERIFIED_POSITION_PROFILES`, `detect_padding_index`, `detect_model_family`, `detect_checkpoint`, `resolve_position_profile`, `roberta_position_ids_from_input_ids`, `authoritative_position_ids`; `UnmarkEncoder` resolves a verified profile at construction, derives positions, and **validates** any supplied tensor |
| `docs/spec/decisions.md` | D-B4B-002 **CLOSED**; D-B4B-006 added |
| `docs/experiments/b4b-phobert-adapter-integration-result.md` | **new** — first run recorded as INCOMPLETE; position scope clarified |
| `tests/test_neural_adapter.py` | one import allowlist widened for `dataclasses` (§P) |
| `tests/test_b4b_provenance_and_positions.py` | **new** — 36 local + 16 torch-gated |

Deterministic B1/B2/B3 code untouched. **No proposal change**; PDF stale: **YES**
(unchanged from v1.4).

---

## C. FIRST REAL B4B RUN

`vinai/phobert-base` @ `01daacda…`, `RobertaModel` + `PhobertTokenizer`,
transformers 4.57.6, torch 2.11.0+cu128, hidden size 768, vocab 64001. **Weights
loaded: true. Training performed: false.** 22 checks computed, **21 passed**.

Full record: [`docs/experiments/b4b-phobert-adapter-integration-result.md`](../experiments/b4b-phobert-adapter-integration-result.md).

---

## D. FAILURE ROOT CAUSE

```
revision verified (tokenizer and model)   FAIL
resolved_tokenizer_revision = 01daacda68afe13d83023d16ec647239e344a1e6
resolved_model_revision     = null
```

The collector searched the loaded object for a Hugging Face cache snapshot path.
That works for a **tokenizer**, which keeps real resolved file paths on the
instance (`vocab_file`, `merges_file`). It cannot work for a **model**:
`model.name_or_path` is the repo id, `"vinai/phobert-base"`, and never a path.

**The check could not have passed for any model at any revision.** It was not a
near-miss or a flaky path — it was verifying nothing, and reported that honestly
as a failure rather than passing vacuously. Recorded as D-B4B-006.

---

## E. B14A PROVENANCE EVIDENCE

The independent offline diagnostic, on the same runtime, established that the
requested revision **was** loaded:

```
HF_HOME     /content/unmark-draft/.hf-cache
hub cache   /content/unmark-draft/.hf-cache/hub
snapshot    .../models--vinai--phobert-base/snapshots/01daacda68afe13d83023d16ec647239e344a1e6
```

All seven checks true — `exact_snapshot_exists`,
`direct_config_in_requested_snapshot`, `direct_weight_in_requested_snapshot`,
`main_ref_matches_requested`, `transformers_cached_config_matches`,
`transformers_cached_weight_matches`, `autoconfig_commit_matches` — status
**`MODEL_REVISION_CACHE_PROVENANCE_CONFIRMED`**.

`AutoConfig` reported `_commit_hash = 01daacda…`; `cached_file` and
`huggingface_hub.try_to_load_from_cache` independently returned the same raw
snapshot paths for `config.json` and `pytorch_model.bin`.

**The symlink trap.** `snapshots/01daacda…/pytorch_model.bin` symlinks to the
content-addressed blob `blobs/a0b0f091…`. The **raw** snapshot path carries the
revision; the blob path carries none — it is addressed by content, so it is the
same blob whatever revision points at it. `Path.resolve()` would follow the link
and destroy the only evidence being collected.

---

## F. REPAIRED MODEL REVISION VERIFIER

`verify_model_revision(model, checkpoint, revision)` collects structured
evidence:

| Evidence | Role |
|---|---|
| Cached **config** raw path under `snapshots/<revision>/` | **required** |
| Cached **weight artifact** raw path under `snapshots/<revision>/` | **required** |
| `model.config._commit_hash` | must **agree** when present; absence is not failure |
| `model.name_or_path` | **not evidence** — recorded, flagged `name_or_path_is_revision_evidence: false` |
| `refs/main` | recorded, flagged `refs_main_is_required: false` |
| Resolved blob path | forensic only, flagged `weight_blob_is_not_revision_evidence: true` |

**Config alone is insufficient** — it would not show the *weights* came from that
revision, which is the thing that matters. `pytorch_model.bin`,
`model.safetensors` and both sharded index layouts are accepted, so the verifier
is not silently checkpoint-specific.

**`_commit_hash` asymmetry, deliberate.** A *disagreement* fails loudly (the
probe exits 3). An *absence* does not: it is a private attribute transformers is
free to drop, so treating its absence as failure would make the verifier brittle
against a library upgrade for no gain.

**Cache root.** `HF_HOME=/x/.hf-cache` means the hub cache is
`/x/.hf-cache/hub`. Passing `HF_HOME` itself makes transformers look for
`HF_HOME/models--…` when the layout is `HF_HOME/hub/models--…`, and the lookup
**silently finds nothing** — indistinguishable from a genuinely absent file. The
verifier takes `huggingface_hub.constants.HF_HUB_CACHE` rather than
reconstructing it, and falls back to `HF_HOME/hub` only if that constant is
unavailable. Works offline under `local_files_only=True`.

**`refs/main` is not required.** The project pins an exact commit; upstream
`main` may legitimately move later while the pinned snapshot stays correct and
reproducible. Requiring a match would fail a valid pinned run for a reason
unrelated to it. It is recorded as context only.

A test asserts `extract_snapshot_revision` is never fed a resolved path.

---

## G. POSITION-ID EMPIRICAL RESULT

| Case | Implicit `inputs_embeds` vs authoritative `input_ids` |
|---|---|
| 1 — one sentence, no padding | **identical** |
| 2 — right-padded batch | **DIFFERENT** |
| 3 — unequal-length batch | **DIFFERENT** |
| 4 — real special-token batch | **DIFFERENT** |

```
authoritative (input_ids)      2, 3, 4, 5, 6, 1, 1, 1
implicit (inputs_embeds)       2, 3, 4, 5, 6, 7, 8, 9
```

Padding positions take the padding index on the authoritative path; the implicit
path numbers straight through them.

**Case 1 matching is what makes this dangerous.** A single unpadded sentence
looks correct, so the bug would not surface until batching — by which point it
would be inside a training run, silent, exactly as §4.5 warns.

---

## H. D-B4B-002 CLOSURE

The pre-committed rule: *if **any** required case differs, the adapted
`inputs_embeds` path must pass explicit `position_ids` reproducing the
authoritative `input_ids` behaviour.* Three of four differed.

**CLOSED: explicit authoritative `position_ids` are REQUIRED for the PhoBERT
adapted path.**

The rule was committed **before** the result was known (Audit 015), so the
outcome could not be read selectively.

---

## I. POSITION-ID IMPLEMENTATION

`UnmarkEncoder.forward` **derives and passes** authoritative `position_ids`
whenever the caller omits them, from the **same** `input_ids` used for the frozen
word-embedding lookup, so the two cannot drift apart.

**Omission can no longer produce the sequential fallback.** Documenting "callers
should remember to pass `position_ids`" would have left the wrong answer as the
default; the empirical decision is enforced by the wrapper, not by discipline.

**A supplied tensor is checked, not trusted.** It must equal the authoritative
tensor exactly — same shape, exact integer equality via `torch.equal`, **no
floating tolerance**, because these are indices rather than values — or
`PositionContractViolation` is raised. Honouring an arbitrary tensor would have
left a live footgun: `wrapper(..., position_ids=wrong_ids)` could silently
reintroduce exactly the invariant D-B4B-002 closed. Tests cover a wrong tensor, a
shape mismatch, and specifically the sequential `[[2,3,4,5,6]]` the implicit path
would have produced.

`roberta_position_ids_from_input_ids(input_ids, padding_idx)` implements the
verified rule. The **padding index is read from the model** —
`embeddings.padding_idx`, the word-embedding table's, then `config.pad_token_id`
— and never hardcoded; failing to find one raises rather than assuming `1`,
because a wrong padding index silently shifts every position id.

**The attention mask is deliberately not used as a substitute.** It marks what to
attend to, not how the model numbers positions, and the two disagree precisely at
the padded positions where the bug lives.

Positions are passed to the encoder **exactly once**, through the model API.
Nothing is added into `z`; a test asserts neither `forward` nor
`adapted_embeddings` mentions `position_embeddings`.

**Path C keeps its freedom.** The probe's A/B/C position comparison genuinely
needs to feed arbitrary explicit tensors to the model, so it calls the **frozen
encoder directly** — an inference diagnostic of the model API. A test asserts
`compare_paths` never routes through the wrapper. The production wrapper stays
invariant-enforcing rather than being weakened to accommodate a diagnostic.

**The gradient path is unchanged from Audit 015** and still omits
`position_ids`, so the run exercises the wrapper's own derivation end to end:
`φ → derived positions → z → frozen E_θ → loss → gradients into φ`.

---

## J. BACKBONE SCOPE

**The earlier scope was too broad and has been repaired.** This audit first
registered `VERIFIED_POSITION_FAMILIES = {"roberta"}`, clearing any checkpoint
whose `model_type` matched. That asserted an empirical result never obtained: the
probe measured **`vinai/phobert-base`**, not every RoBERTa-family checkpoint, and
weight-tying, embedding resizing or a custom embedding subclass could change the
behaviour. It also contradicted this audit's own statement that a second backbone
needs its own measurement.

Permission is now a **profile**, matched on checkpoint, model type and model
class **together**:

```
VerifiedPositionProfile(
    checkpoint="vinai/phobert-base",
    model_type="roberta",
    model_class="RobertaModel",
    position_rule="roberta_input_ids_offset",
    evidence="D-B4B-002 (real-model B4B probe)",
)
```

`VERIFIED_POSITION_PROFILES` has exactly one entry, because exactly one was
measured. `roberta-base`, `xlm-roberta-base` and `vinai/phobert-large` are all
**rejected** — tests cover each — as is a matching checkpoint under an unexpected
model class. A non-matching backbone raises `UnsupportedPositionSemantics` **at
wrapper construction**, not deep inside a training run.

**The distinction being drawn.** The *arithmetic* is ordinary RoBERTa-style and
nothing about it is unique to PhoBERT; `position_rule` is a reusable selector so
a future profile can adopt it without inheriting its evidence. What is
PhoBERT-specific is the **empirical permission** to rely on it.

**Profile identity and revision are separate obligations.** The profile
identifies the *checkpoint* through `name_or_path` — which is the repo id, and
therefore exactly the wrong source for a revision (§F) and exactly the right one
for a checkpoint identity. The provenance layer verifies the exact *revision*
from cache snapshot paths. **Neither substitutes for the other, and the rerun
must prove both.**

The checkpoint-specific logic lives in the **encoder integration layer**.
`OrthographyInputAdapter` remains backbone-independent: a test asserts it
mentions no `position_ids`, `padding_idx`, `roberta`, `position_embeddings`,
`phobert`, `checkpoint` or `VerifiedPositionProfile` anywhere.
**[D-B3B0-002](../spec/decisions.md#d-b3b0-002) remains OPEN.**

---

## K. INPUT_IDS VS INPUTS_EMBEDS EQUIVALENCE

With explicit authoritative `position_ids`, the real model gave:

```
max_abs_diff          = 0.0      mean_abs_diff         = 0.0
max_abs_diff_content  = 0.0      mean_abs_diff_content = 0.0
max_abs_diff_padding  = 0.0
```

**Exact, not within a tolerance**, and including padding positions. This is
real-model evidence. The control remains inference-only — `model.eval()` under
`torch.no_grad()` — which is correct and separate from the gradient path.

---

## L. ADAPTER / PARAMETER EVIDENCE

| Measure | Value |
|---|---|
| Hidden size (read from model) | 768 |
| Adapter trainable parameters | **3,551,232** |
| Expected `6d² + 16d` at d=768 | **3,551,232** |
| Encoder trainable parameters | **0** |
| Initial gate min/max | 0.009999998845160007 |
| Initialised `z` max abs diff from base | 0.038273051381111145 |

Gate `= 0.01` to floating-point precision; initialised adapter **not** identity.
`d` came from `model.config.hidden_size`; a test asserts `768` appears nowhere in
the adapter or probe as a literal.

---

## M. GRADIENT EVIDENCE

```
gradient_loss_source                 encoder_final_hidden_state
gradient_path_includes_encoder       true
encoder_output_requires_grad         true
encoder_grad_count                   0
encoder_parameters_with_grad_tensor  0
```

Gradient tensors existed for all eight required adapter components; all finite,
at least one nonzero. **The Audit-015 repair is validated on the real model** —
the diagnostic scalar comes from the encoder's final hidden state, so the
backward genuinely traverses the frozen encoder into `A_φ`. This probe design is
unchanged by this task.

---

## N. FROZEN-ENCODER MODE EVIDENCE

| Step | wrapper | encoder | adapter |
|---|---|---|---|
| constructed | — | **false** | true |
| `wrapper.train()` | true | **false** | true |
| `wrapper.eval()` | false | **false** | false |
| `wrapper.train()` again | true | **false** | true |

`requires_grad` stayed false throughout. D-B4B-004 holds on the real model.
Unchanged by this task.

---

## O. REGRESSION SAFETY

The 21 passing checks are untouched. Explicit regression tests assert the earlier
repairs survive: the gradient loss still comes from `encoder_final_hidden_state`
and `z_grad.sum().backward()` cannot return; `super().train(mode)` +
`self.encoder.eval()` is still present; no optimizer or `save_pretrained` call
exists; exactly one `.backward()` in the probe.

The probe now computes **27 checks** — the original 22, plus two from the
provenance/position repair (derived ids match the model's authoritative ids;
padding index derived from the model), plus three from this hardening (the
wrapper passes the authoritative ids to the encoder; it rejects a wrong
caller-supplied tensor; the backbone matched a **verified profile**, not just a
family). The count is derived via `len(summary['checks'])`, so code and
documentation cannot drift.

The adapter mathematics is untouched by this hardening: the fusion equation, the
gate, its initialisation, both channel contracts and the parameter formula are
exactly as B4A locked them.

---

## P. LOCAL TEST EVIDENCE

```
.venv/bin/python -m pytest -q -p no:cacheprovider
1870 passed, 32 skipped in 7.89s
```

Baseline before this task: 1831 passed, 16 skipped.
`tests/test_b4b_provenance_and_positions.py` holds **39 local + 16 torch-gated**.

The probe imports torch only inside `main()`, so its provenance helpers are
**genuinely executed** locally rather than only inspected: the cache lookup is
monkeypatched to simulate hits and misses, and the verification logic runs for
real.

Covering: revision extracted from raw snapshot paths; a blob path yielding no
revision; the extractor never fed a resolved path; repo id not a snapshot path;
config **and** weight both required; safetensors and sharded layouts accepted; a
disagreeing `_commit_hash` failing while an absent one does not; `refs/main`
recorded but not required; `name_or_path` insufficient; a wrong requested
revision failing; hub cache root not being `HF_HOME`; D-B4B-002 closed in the
log; padding index read from the model and not hardcoded; no hardcoded hidden
size; the orthography adapter position-agnostic; helpers in the integration
layer; exactly one registered `VerifiedPositionProfile` and no family-wide
permission; unverified backbones failing loud; positions passed exactly once;
nothing positional added into `z`; the wrapper deriving on omission; the probe omitting `position_ids` on the gradient path; and
the three regression guards.

Torch-gated: the RoBERTa rule reproducing the measured `2,3,4,5,6,1,1,1`;
unequal-length batches numbered independently; an unverified backbone rejected at
construction; and the wrapper supplying `[[2,3,4,5,1]]` automatically.

**Added by the scope/safety hardening.** `VERIFIED_POSITION_FAMILIES` no longer
existing; exactly one registered profile, matched on checkpoint + type + class;
profile matching rejecting `roberta-base`, `vinai/phobert-large`, a different
family, a patched model class and a `None` checkpoint; checkpoint identity kept
distinct from revision verification; the wrapper validating rather than trusting
supplied ids; exact integer comparison with no `allclose`/`atol`/`rtol`/`isclose`;
path C using the raw encoder; and the probe checking both the passed ids and the
override rejection. Torch-gated: the PhoBERT profile accepted; three other
checkpoints and a wrong model class rejected; correct supplied ids accepted;
wrong ids, the sequential fallback tensor, and a shape mismatch each failing
loud; and `position_ids` always reaching the encoder.

One pre-existing test needed widening: `test_adapter_does_not_import_the_deterministic_pipeline`
allowed `{__future__, typing, torch, unmark}`, and the profile record adds
`dataclasses`. Its intent — no deterministic-pipeline imports in the neural
module — is unchanged.

**Three documentation-consistency guards** were added last, after two stale
current-state claims survived into this file: Section A kept a pre-repair test
count while Section P was updated, and Section P still listed "only `roberta`
claimed verified" after the scope hardening. The guards assert that Section A's
headline count matches Section P's pytest block, that no phrasing implies
family-wide position permission, and that the locked current state (verdict, 27
checks, D-B3B0-002 open, no training) is still stated. They were verified to fail
on a reintroduced stale count before being kept.

**Adding them changed the count**, from 1867 to 1870 — an audit that records its
own test total moves when a test is added to it. Both sections are updated, and
the guard would have caught the omission.

**Four of my first-draft tests failed on my own mistakes** — `importorskip` placed
after the import it guarded (twice), `ast.unparse` normalising double quotes to
single, and two functions sharing the name `authoritative_position_ids`. All four
were test defects, not implementation defects, and were fixed.

---

## Q. REAL-MODEL RERUN PLAN

The repaired script runs in the **existing** Colab runtime. Nothing requires
deleting it or re-downloading the model.

1. update the repository to the researcher's new committed revision;
2. reuse `.hf-cache`, `.resources-cache`, `.venv-colab` as they are;
3. rerun `scripts/b4b_phobert_adapter_probe.py --revision 01daacda…`;
4. a new timestamped directory appears under `results/b4b/<run_id>/`.

Artifacts: `config.json`, `summary.json`, `report.md`, `position_ids.json` (now
cases **and** helper comparison), `equivalence.json`, `gradients.json`,
`module_modes.json`, `channels.json`, and the new **`provenance.json`** carrying
structured evidence rather than one opaque boolean.

The verifier works offline against the already-cached model.

The rerun must additionally confirm the two hardening repairs: that the wrapper
matched a **verified position profile** (not merely a model family), and that it
**rejects a wrong caller-supplied `position_ids`**. Both are computed checks, and
`position_ids.json` records the matched profile alongside the id comparison.

**Success requires all 27 checks true**, emitting
`B4B_PHOBERT_ADAPTER_INTEGRATION_COMPLETE`. **Removing the failing check is not
success** — the repaired verifier must genuinely prove both tokenizer and model
resolve to `01daacda68afe13d83023d16ec647239e344a1e6`. If any previously passing
neural check regresses, the run is INCOMPLETE.

---

## R. DECISION LOG / PROPOSAL CONSISTENCY

**D-B4B-002: CLOSED** — explicit authoritative `position_ids` required, with the
pre-committed rule, all four observed cases, the padding-position mismatch, the
exact post-repair equivalence, affected code, and future-backbone scope. The
original framing is preserved beneath the resolution.

**D-B4B-002 scope clarified** — the empirical position semantics were validated
for the **`vinai/phobert-base` integration profile**, which does *not* authorise
arbitrary `roberta` checkpoints; every new backbone or checkpoint must validate
its own `inputs_embeds` behaviour before registration. The production wrapper
enforces authoritative ids, and caller-supplied ids are accepted **only** when
exactly equal to the derived ones.

**D-B4B-006: added** — `model.name_or_path` is not revision evidence; cache
snapshot paths and the loaded config commit are; the raw path must be read before
resolving the symlink; `refs/main` is non-authoritative for a pinned revision.
Reproducibility engineering, not a scientific architecture change.

**Proposal updated: NO.** §4.5 already requires position embeddings to be
supplied by the encoder exactly once; this is how that is achieved against a real
model API. PDF stale: **YES**, unchanged from v1.4.

---

## S. BLOCKING ISSUES

**None for this task.** One blocks *completion of B4B*, by design:

**The repaired probe has not run.** The provenance verifier and the automatic
position-id derivation are validated locally against stubs and simulated caches —
which proves the logic, not the integration. Only the rerun can confirm both
against real PhoBERT.

Non-blocking:

1. **`VERIFIED_POSITION_PROFILES` has exactly one member.** Any second backbone
   or checkpoint needs its own measurement; the wrapper refuses it until then.
   That is the intended behaviour, not a limitation to work around — and the
   earlier family-wide scope, which would have quietly waved through
   `roberta-base`, was repaired for exactly this reason.
2. **The local position tests use a `roberta`-shaped stub**, not real PhoBERT.
   They prove the arithmetic, the profile matching and the wiring; the real check
   is the rerun.

3. **A review caught the scope overreach, not a test.** The code and this audit's
   own §S disagreed with each other, and nothing mechanical noticed. The new
   profile tests close that particular gap, but the general limit stands: an
   assertion that matches the code cannot detect that the code claims too much.
4. **Proposal PDF remains stale** (v1.4 source).

---

## T. GIT STATE

`HEAD = 749cc39`.

```
 M docs/spec/decisions.md
 M scripts/b4b_phobert_adapter_probe.py
 M tests/test_neural_adapter.py
 M unmark/modeling/adapter.py
?? docs/audits/016-b4b-real-model-provenance-and-position-repair.md
?? docs/experiments/b4b-phobert-adapter-integration-result.md
?? tests/test_b4b_provenance_and_positions.py
```

Every change is left **unstaged**. **No adapter mathematics changed** by this
hardening: the fusion equation, gate, initialisation, channel contracts and
parameter formula are exactly as B4A locked them. No `add`, `commit`, `push`, `tag`, `stash`,
`reset`, `checkout` or `restore` was run. **Torch and transformers were not
installed locally; no model weights were downloaded or loaded locally; no network
was accessed; nothing was trained.**

```text
AUDIT FILE WRITTEN: docs/audits/016-b4b-real-model-provenance-and-position-repair.md
OTHER AUDIT MODIFICATIONS MADE: NO
COMMIT CREATED: NO
```
