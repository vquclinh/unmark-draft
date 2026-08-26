# Audit 035 — Torch Resume Test Fixture Repair (034-MAJ1)

**Scope:** ONE verification defect. `tests/test_stage1_resume_state_machine_torch.py` only.
**Date:** 2026-08-26
**Mode:** TEST REPAIR. No production code, script, config, spec or prior audit was modified.

---

## 1. Starting HEAD

```
$ git rev-parse HEAD
55aa4064780b37626bcae7eef83c504a96fcc51f
```

## 2. Starting Working Tree

The uncommitted consolidated repair (Audit 033) plus audits 031–034. Nothing staged;
`git diff --check` exit 0. `docs/audits/035-…md` did not exist.

Production diff fingerprint recorded **before** any edit:

```
$ git diff -- unmark/ scripts/ | md5sum
481566c450b2d1d06a8204150c5be7d1
```

## 3. Exact 034-MAJ1 Root Cause

`UnmarkEncoder.__init__` (`unmark/modeling/adapter.py`) deliberately fails fast rather than deep
inside a run:

```python
self.position_profile = resolve_position_profile(encoder)
self.padding_index    = detect_padding_index(encoder)
```

`resolve_position_profile` matches the **whole** profile — checkpoint, model type and model class —
against `VERIFIED_POSITION_PROFILES`, whose sole entry is
`PHOBERT_BASE_POSITION_PROFILE = ("vinai/phobert-base", "roberta", "RobertaModel")`. The comment on
that structure is explicit that a shared `model_type` is *not* evidence: D-B4B-002 measured one
checkpoint, and D-B3B0-002 is OPEN.

The old fixture exposed only `config.hidden_size`, so it presented
`(None, None, "_FrozenStub")` — no profile matched — and it offered no padding index by any of the
three routes `detect_padding_index` tries. Both calls raised `UnsupportedPositionSemantics`, so
`build_objective()` died at construction and **all 12 tests in the file would ERROR** in the CUDA
environment, before reaching any checkpoint or resume code.

Reproduced against the **real** production functions, extracted by AST from `adapter.py` and executed
without torch:

```
OLD  _FrozenStub      detect_checkpoint -> None   model_family -> None   class -> '_FrozenStub'
                      resolve_position_profile -> RAISES UnsupportedPositionSemantics
                      detect_padding_index     -> RAISES UnsupportedPositionSemantics
```

## 4. Exact Test-File Changes

One file: `tests/test_stage1_resume_state_machine_torch.py` (untracked, part of the Audit 033 change
set, so it remains untracked with modified content).

| Removed | Added |
|---|---|
| `_StubConfig` (only `hidden_size`) | `_RobertaLikeConfig` — `model_type`, `_name_or_path`, `pad_token_id`, `hidden_size` |
| `_FrozenStub(torch.nn.Module)` | `_FrozenBackbone(torch.nn.Module)` with real frozen parameters, an `embeddings` namespace and `get_input_embeddings`; `__name__`/`__qualname__` set to `"RobertaModel"` |
| `_ResumeObjective(torch.nn.Module)` (a hand-rolled objective) | `_NoForwardObjective(Stage1Objective)` — the **real** objective subclassed, with only `forward` poisoned |
| `build_objective` returning the hand-rolled objective | `build_objective` returning a real adapter + real `UnmarkEncoder` + real objective built with `provenance().weights` |
| — | `import types`, `from unmark.stage1.objective import Stage1Objective` |

No test body, assertion, case or scenario was changed. The 12 test functions are byte-identical.

## 5. Why Production Code Was Not Changed

The defect was entirely in the test double. `UnmarkEncoder`'s construction contract is correct and
intentional — failing at construction is the documented design, and the profile is checkpoint-scoped
on purpose because only one checkpoint was empirically measured. Weakening it, or adding a test-only
branch, would have removed a real scientific guard to make a test pass. Nothing in production was
touched, and this is verified mechanically in §12.

The fixture was also **not** repaired by monkeypatching `resolve_position_profile` or
`detect_padding_index`, and `train_run` is not mocked. The stub now satisfies the same contract a
legitimate PhoBERT encoder satisfies.

## 6. Constructor Preconditions Re-Derived

Re-derived from source, not copied from Audit 034 prose:

| Precondition | Production consumer | Satisfied by |
|---|---|---|
| checkpoint identity `"vinai/phobert-base"` | `detect_checkpoint` — tries `name_or_path`/`_name_or_path` on the encoder, then on its config | `config._name_or_path` |
| model family `"roberta"` | `detect_model_family` — `config.model_type` | `config.model_type` |
| model class `"RobertaModel"` | `resolve_position_profile` — `type(encoder).__name__` | `_FrozenBackbone.__name__` / `__qualname__` |
| padding index | `detect_padding_index` — `embeddings.padding_idx`, then `embeddings.word_embeddings.padding_idx`, then `config.pad_token_id` | all three, agreeing at `1` |
| hidden size | `verify_model_contract` — `int(getattr(encoder.config, "hidden_size", 768))` | `hidden_size = 8` (see below) |
| frozen, in eval | `freeze_encoder` (called by `UnmarkEncoder`), then `verify_model_contract`'s `encoder.training` check | real parameters + `self.eval()` |
| no leaked encoder parameter | `verify_model_contract` — `leaked` | encoder params are `word_embeddings.*`/`projection.*`; trainable params are `adapter.*` — disjoint |

Verified against the **real** extracted production functions:

```
NEW  _FrozenBackbone   detect_checkpoint   -> 'vinai/phobert-base'
                       detect_model_family -> 'roberta'
                       model_class         -> 'RobertaModel'
                       resolve_position_profile -> MATCHES vinai/phobert-base/roberta/RobertaModel
                                                   rule=roberta_input_ids_offset
                                                   evidence=D-B4B-002 (real-model B4B probe)
                       detect_padding_index -> 1
```

An additional check confirms the padding index does not depend on the `config.pad_token_id` fallback
alone: with that field removed, `embeddings.padding_idx` still resolves to `1`.

## 7. Comparison With the Known-Working Idiom (`tests/test_stage1.py:745-767`)

The existing `RobertaLike` supplies `model_type="roberta"`, `pad_token_id=1`,
`_name_or_path="vinai/phobert-base"`, an `embeddings` namespace, real parameters, and
`RobertaLike.__name__ = "RobertaModel"`. The repaired fixture matches it on every one of those
points.

**It deliberately differs in one respect, and copying blindly would have introduced a new failure.**
`RobertaLike`'s config has **no `hidden_size`**. That is correct there, because that test never calls
`verify_model_contract`. Here `train_run` calls it *before* the resume block, and it reads:

```python
hidden = int(getattr(encoder.config, "hidden_size", HIDDEN_SIZE))
if hidden == HIDDEN_SIZE and total != ADAPTER_TRAINABLE_PARAMETERS:
    raise TrainerContractViolation(...)
```

With `hidden_size` absent the fallback is **768**, which activates the locked 3 551 232-parameter
gate — and a d=8 adapter has far fewer. `hidden_size = 8` is therefore load-bearing: it keeps that
one clause inactive while every other clause of the contract (no leaked encoder parameter, encoder in
eval) is still enforced against the fixture. This is recorded in the `TINY_HIDDEN` docstring so the
next reader does not "fix" it by aligning with `test_stage1.py`.

## 8. Real `train_run` Still Exercised

The chain is unchanged and unmocked:

```
ValidationPoint (real)
  -> checkpoint_payload            (real, and it now validates points)
  -> save_training_checkpoint      (real torch.save + atomic publish)
  -> load_training_checkpoint      (real torch.load)
  -> train_run(resume=carried)     (real)
       -> verify_checkpoint            (real)
       -> require_resumable_leg        (real)
       -> adapter.load_state_dict(strict=True)          (real)
       -> optimizer.load_state_dict                     (real AdamW from build_optimizer)
       -> require_optimizer_parameter_identity          (real)
       -> require_optimizer_state_device                (real)
       -> DeterministicSampler.from_state               (real; binds corpus_digest)
       -> ValidationPoint.from_dict                     (real — the line that used to raise)
       -> resolve_budget / budget_decision              (real)
```

Symbol occurrence counts in the file: `train_run` 25, `resume_cap` 10, `load_training_checkpoint` 6,
`UnmarkEncoder` 4, `save_training_checkpoint` 3, `build_optimizer` 3, `checkpoint_payload` 2,
`DeterministicSampler` 2, `fresh_adapter` 2. The only `monkeypatch`-like substitution in the file is
`torch.optim.AdamW.step`, restored in a `finally`. Nothing under test is mocked away, and the tests
have **not** degraded into helper-only checks.

The sampler state is still produced by a real `DeterministicSampler` over the same `CHUNKS` the
resume passes to `train_run`, so `from_state`'s `corpus_digest` binding is genuinely satisfied rather
than bypassed.

## 9. No-Update Guards Remain

Three independent guards, unchanged:

1. **Structural** — every case restores at `global_update == cap`, so `while global_update < cap` is
   false on entry and the loop body cannot execute.
2. **Optimizer** — `torch.optim.AdamW.step` is replaced with a function that fails the test if
   called, and restored in `finally`.
3. **Forward** — `_NoForwardObjective.forward` raises, and `_FrozenBackbone.forward` raises
   independently, so an unexpected forward through either path fails loudly.

`evaluate_fn` is additionally asserted to have been called zero times in case A, proving update 0 was
*restored* rather than re-measured.

**No scientific optimizer step can occur in this file.**

## 10. Local Focused Test Result

```
$ .venv/bin/python -B -m pytest -q -p no:cacheprovider -rs \
    tests/test_stage1_resume_state_machine_torch.py \
    tests/test_stage1_resume_state_machine.py \
    tests/test_stage1_training_resume_state.py \
    tests/test_stage1_training_resume.py \
    tests/test_stage1_name_resolution.py

77 passed, 3 skipped in 0.44s

SKIPPED tests/test_stage1_resume_state_machine_torch.py:45  the real train_run half needs torch
SKIPPED tests/test_stage1_training_resume.py:40             the tensor half needs torch
SKIPPED tests/test_stage1_name_resolution.py:80             unmark.stage1.objective needs torch
```

The torch-gated file still SKIPS locally, exactly as expected. **This skip is not runtime proof of
the repair** — see §15.

## 11. Full Lightweight Suite

```
$ .venv/bin/python -B -m pytest -q -p no:cacheprovider
3737 passed, 105 skipped in 131.48s
```

Identical to the pre-repair figures in Audit 034 (3737 / 105), as expected: this change touches only
a file that skips locally.

## 12. Production Diff Status

**UNCHANGED — verified byte-for-byte.**

```
before:  git diff -- unmark/ scripts/ | md5sum  ->  481566c450b2d1d06a8204150c5be7d1
after:   git diff -- unmark/ scripts/ | md5sum  ->  481566c450b2d1d06a8204150c5be7d1
```

`git diff --stat -- unmark/ scripts/ configs/ requirements/` is identical before and after
(442 insertions, 36 deletions across 5 files). No MINOR or INFORMATIONAL finding from Audit 034 was
touched: the `ValidationPoint` score tolerance, `CampaignIdentity`, the clean-tree policy, README,
freeze metadata, the `budget_limited` schema, `pyproject.toml` and torch pinning are all untouched.

## 13. Scientific `optimizer.step` Status

**None executed.** `lr-pilot`, `r-phase1` and `final-main` were not invoked. The torch-gated file
skipped locally, and by §9 it cannot step an optimizer even when it runs.

## 14. Official UIT-VSFC TEST Status

**SEALED / UNUSED.** Not opened, inspected, read, screened, evaluated, mounted, or passed to any
command. No information derived from it.

## 15. Exact CUDA Test That Must Run Next

```
python -B -m pytest -q -rs tests/test_stage1_resume_state_machine_torch.py
```

Expected: **12 passed, 0 skipped, 0 errors.** Any `UnsupportedPositionSemantics` would mean the
fixture repair is incomplete; any `AssertionError` naming `optimizer.step()` or a forward pass would
mean the no-update discipline was broken.

Then, as already required by Audit 034 §23, the four MUST RE-RUN items:
checkpoint/resume, the 20k→40k continuation, artifact handoff, and repository-head provenance.

## 16. Final Verdict

The fixture-construction failure identified by Audit 034 as 034-MAJ1 is repaired: the stub now
satisfies every precondition of the real `UnmarkEncoder` constructor — verified against the actual
production detection functions — without weakening production validation, without monkeypatching the
detectors, and without mocking `train_run`. Production code is byte-identical. The twelve tests still
traverse the real writer → save → load → `train_run` restore chain, and the no-update guards are
intact and strengthened.

No new material issue was discovered.

**This is not a CUDA PASS.** The known fixture-construction failure is repaired; the test must still
execute in authoritative CUDA acceptance.

**TORCH RESUME TEST FIXTURE REPAIR COMPLETE — READY TO COMMIT AND RUN AUTHORITATIVE CUDA ACCEPTANCE**

---

*End of Audit 035.*
