# Audit 001 — G−1 environment policy and harness readiness

| | |
|---|---|
| **Audit id** | 001 |
| **Date (UTC)** | 2026-08-19 |
| **Scope** | The task that cleaned up the accidental local ML environment and restructured the repository into a lightweight-local / heavyweight-Colab workflow |
| **Repository state** | `HEAD = 4f39ccf` ("update proposal"); working tree carries the uncommitted G−1 scaffolding |
| **Type** | Strict read-only audit — no repairs, no installs, no network, no model execution |
| **Phase** | Phase 0 / Gate G−1 |

> This document records the audit findings only. It does not fix anything. Whether a
> follow-up task is opened is the researcher's decision.

---

## A. Verdict

**FAIL — FOLLOW-UP REQUIRED**

The environment/cleanup half of the task is clean and verifiable: no global installs,
`.venv` is repo-scoped and isolated (`include-system-site-packages = false`) with only
pytest/PyYAML in it, none of the six heavy libraries are importable locally, no model
weights or >20 MB files exist anywhere in the checkout, the Hugging Face model cache is
gone, `.gitignore` behaves correctly including the `.gitkeep` negation, dependency
separation is correct, and all 104 tests pass offline in 0.16 s with sockets disabled.

Two code-design defects block sign-off. First, the future-compatibility check fails
outright: `test_unmark_package_never_imports_the_heavy_ml_stack` scans **every** `.py`
file under `unmark/` and rejects the import even when indented, so it bans not just
top-level but *lazy* torch imports anywhere in the package — it will fail the moment
`unmark/modules/`, `unmark/training/` or `unmark/baselines/` (all named in proposal
§8.1) are implemented. Second, `base_signature` is case- and punctuation-sensitive by
design, and simulation confirms that a restorer which capitalises a lowercase
`full_strip` sentence and adds a final stop — highly plausible for a Wikipedia/news
trained model — drives the core base-preservation rate to 0.0 and reports
`ENGINEERING_SMOKE_FAIL` under a check literally named `no_catastrophic_lexical_rewriting`,
when no lexical rewriting occurred.

---

## B. Requirement matrix

| Requirement | Status | Evidence | Notes |
|---|---|---|---|
| No global install | PASS | system `python3 -c "import torch"` → ModuleNotFoundError; `pip3 list --user` shows none of torch/transformers/sentencepiece/safetensors/accelerate | Pre-existing user-site packages (mypy, pandas, huggingface_hub 1.27.0 from Aug 9) untouched |
| Local `.venv` isolated | PASS | `sys.executable` = `<repo>/.venv/bin/python`; `sys.prefix != sys.base_prefix`; `pyvenv.cfg: include-system-site-packages = false` | 32 MB |
| No heavy local dependencies | PASS | torch, transformers, sentencepiece, safetensors, accelerate, huggingface_hub, tokenizers, numpy all report `absent`; `pip list` = iniconfig, packaging, pip, pluggy, Pygments, pytest, PyYAML | — |
| No local model weights | PASS | `find` for `*.safetensors/*.bin/*.pt/*.pth/*.ckpt` → empty; `find -size +20M` → empty; largest non-`.git` file is `unmark-proposal.pdf` (341 KB) | HF hub cache now holds only pre-existing `datasets--AIGuruTinix--ViFinQA` |
| No stray `.venv-colab` / `.hf-cache` | PASS | `find -maxdepth 2` returns only `./.venv` | — |
| Dependency separation | PASS | `dev.txt` = `-r base.txt` + `pytest>=8,<10`; `experiment.txt` = `-r base.txt` + transformers/sentencepiece/safetensors/accelerate | `base.txt` = PyYAML only |
| Transformers pin | PASS | `transformers==4.57.6` in `experiment.txt` | End-to-end load **unverified**; only the 5.15.1 failure was observed directly |
| `torch` omitted from `experiment.txt` | PASS | absent, with an explanatory comment | Intentional, so Colab's CUDA build is reused |
| `.gitignore` | PASS | `check-ignore -q` → `.venv/`, `.venv-colab/`, `venv/`, `.hf-cache/`, `results/**`, `__pycache__/`, `.pytest_cache/` ignored; `results/g_minus1/.gitkeep` **trackable** | See N6 |
| Lightweight imports | PASS | `base_signature('Tôi đang học')` → `Toi dang hoc`; gates module imports with `heavy modules loaded: none`; CLI script imports without torch | `unmark/` non-stdlib imports = `['yaml']` only |
| Future ML-import compatibility | **FAIL** | guard predicate rejects `import torch`, `import torch.nn as nn`, `from torch.utils.data import…`, and indented/lazy variants | See B1 / §E |
| Pinned G−1 config | PASS | parsed YAML: `model_id: nrl-ai/vn-diacritic-vit5-base`, `revision: 30ea5a9e4a0b9436e18915fd4dbb5876eaee7325` | — |
| Deterministic generation config | PASS | `do_sample: false`, `num_beams: 1`, `max_new_tokens: 256`; `validate_config` raises on sampling/beams | Length bound sensible (fine-tuned at input 256) |
| Smoke-test coverage | PASS | all 10 required categories present, 48 cases, unique ids | 6/5/5/8/4/5/4/4/4/3 |
| Record fields | PASS | `outputs`, `deterministic`, `input/output_base_signature`, `base_preserved`, `clean_exact_preserved`, `error`, `latency_ms` + 9 supporting fields | — |
| Result artifacts | PASS | `write_artifacts` emits `config.json`, `cases.jsonl`, `summary.json`, `report.md` | — |
| No automatic scientific acceptance | PASS (with caveat) | 4 checks: model_loaded, core_inference_completed, greedy_decoding_deterministic, no_catastrophic_lexical_rewriting; no accuracy metric, no expected outputs anywhere | Caveat = B2, the rewriting check misfires |
| Offline unit tests | PASS | 104 passed in 0.16 s with `socket.socket` disabled; no network/model/clock/absolute-path deps | Paths are `__file__`-relative |
| README local workflow | PASS (with N3) | lines 29–33 match the required form | `python3.11` binary absent on this machine |
| README Colab workflow | PASS (with N4) | clone → `export HF_HOME="$PWD/.hf-cache"` → `.venv-colab --system-site-packages` → `experiment.txt` → CUDA check → run → `rm -rf .venv-colab .hf-cache` | No command installs `experiment.txt` into local `.venv` |
| No staged changes | PASS | `git diff --cached --stat` empty | — |
| No commits by the audited task | PASS | HEAD `4f39ccf`; reflog's newest entry is that commit | — |
| No commits by the audit | PASS | reflog unchanged | — |

---

## C. Blocking issues

### B1 — blanket ban on ML imports across the whole `unmark/` package

| | |
|---|---|
| **Severity** | High (architectural; blocks G0/G1/G2 implementation) |
| **File** | `tests/test_restore_smoke_utils.py:40-47` (`test_unmark_package_never_imports_the_heavy_ml_stack`); `README.md:185-187` states the same rule as absolute |

**Problem.** The test walks `PACKAGE_ROOT.rglob("*.py")` over the whole `unmark/` package
and asserts no line starts with `import torch` / `from torch` / `import transformers` /
`from transformers`. Because it calls `.strip()` before `.startswith()`, it also rejects
indented, function-local (lazy) imports — the very pattern its sibling test
`test_script_imports_torch_and_transformers_only_lazily` *requires* of
`scripts/g_minus1_restore_smoke.py`.

**Why it matters.** Proposal §8.1 plans `unmark/modules/{channels,fusion,unmark}.py`,
`unmark/training/{stage1_module,stage2_head}.py` and
`unmark/baselines/{restore,align_adapter}.py` — all of which need PyTorch. Verified: every
one of those import forms trips the assertion, lazy or not. The first person to start G1
will see a red suite and will most likely delete the test wholesale, losing the real
protection it provides today.

**Recommended follow-up.** Narrow the scope rather than remove the test. Either

* **(a)** restrict the scan to the subpackages the local G−1 tests import —
  `unmark/orthography/**` and `unmark/gates/**` — via an explicit
  `LIGHTWEIGHT_SUBPACKAGES` allowlist that later phases extend; or
* **(b)** replace the text scan with a behavioural check: import each lightweight module
  in a subprocess and assert `"torch" not in sys.modules`.

Option (b) is stricter where it matters and silent about files it does not cover. The
README sentence *"`unmark/` never imports torch or transformers; the test suite enforces
this"* must be narrowed in the same change.

### B2 — base-signature case/punctuation sensitivity can manufacture a false FAIL

| | |
|---|---|
| **Severity** | High (produces a misleading headline verdict on the first Colab run) |
| **File** | `unmark/orthography/signature.py` (`base_signature`) consumed by `unmark/gates/g_minus1.py` (`engineering_status` → check `no_catastrophic_lexical_rewriting`) |

**Problem.** `base_signature` deliberately preserves case and punctuation, so a restorer
that capitalises a lowercase input or appends a final stop changes the signature and is
scored as `base_preserved = False`. Simulated with a real suite case (`fs_03`):

```text
input   : hom nay thoi tiet o thanh pho ho chi minh rat dep
output  : Hôm nay thời tiết ở Thành phố Hồ Chí Minh rất đẹp.
base_preserved : False
base_diff      : hom→Hom, thanh→Thanh, ho chi minh→Ho Chi Minh, dep→dep.
```

Applied across core cases the core rate falls to 0.0 (threshold 0.9) and the run reports
`ENGINEERING_SMOKE_FAIL`.

**Why it matters.** A ViT5 model fine-tuned on Vietnamese Wikipedia and news is a strong
candidate to restore sentence-initial and proper-noun capitalisation, and the
`full_strip` / `ambiguity_context` / `mixed_script` cases are all written lowercase and
unpunctuated. The gate would then fail under a check named "no catastrophic lexical
rewriting" when no word was rewritten — the opposite of the brief's intent, and exactly
the kind of automatic verdict the task said to avoid. Nothing is *hidden* (`base_diff`
and both signatures are recorded, so a reader can see it), but the headline status and
the per-category table would be wrong.

**Recommended follow-up.** Decide explicitly whether casing and terminal punctuation are
part of the "lexical base". Suggested: keep the strict signature as the recorded
diagnostic, but compute the *engineering* rewriting check on a casefolded,
terminal-punctuation-insensitive variant, and report case-only / punctuation-only
differences as their own labelled category — the existing `whitespace_only_difference`
field already models this pattern correctly. Do **not** simply casefold `base_signature`;
that would weaken the diagnostic the gate exists for.

---

## D. Non-blocking issues

| ID | Severity | File | Problem | Why it matters | Recommended follow-up |
|---|---|---|---|---|---|
| N1 | Low | `configs/restore/nrl_vit5_base.yaml` (`dtype: float32`); `scripts/g_minus1_restore_smoke.py:145-148` | `dtype` is written into the run's `config.json` but never passed to `from_pretrained`; the real dtype is only observed as `parameter_dtype` | Reads like a setting, is a declaration. Changing it in YAML has no effect; `config.json` could carry a `dtype` contradicting `parameter_dtype` | Pass it at load time, or rename to `expected_dtype` and assert |
| N2 | Low | `tests/test_restore_smoke_utils.py:61` | `assert "torch" not in sys.modules or True` is a tautology — always passes | Dead assertion; suggests coverage that does not exist | Delete the line, or make it a real subprocess check |
| N3 | Low | `README.md:29` | Literal command is `python3.11 -m venv .venv`; no `python3.11` exists here (only `/usr/bin/python3.14`) | A copy-paste of the block fails at line 1 with "command not found" | Lead with `python3 -m venv .venv`, note 3.11+ as the supported floor |
| N4 | Low | `README.md:94-98` (Colab "Optional overrides") | That variant omits `HF_HOME`; Colab `%%bash` cells do not inherit exports across cells | Breaks the "one `rm` reclaims everything" claim. Colab-VM only — no local-machine risk | Prefix with `HF_HOME="$PWD/.hf-cache"` as the main block does |
| N5 | Low | `pyproject.toml` | No `requires-python`; nothing declares the supported interpreter range | Code is in fact broadly compatible (§F); undeclared means a future 3.9 CI runner fails with no stated contract | Add `requires-python` once the floor is chosen |
| N6 | Low | `.gitignore:24-26` | Ignores `*.safetensors`, `*.ckpt`, `*.pt` but not `*.bin` or `*.pth` | `pytorch_model.bin` is a real checkpoint name; the pattern set is asymmetric with the artifact search this audit ran | Consider adding `*.pth`; `*.bin` is broader and may want a narrower rule |
| N7 | Informational | `unmark/gates/g_minus1.py:278` | Field named `has_ground_truth`, but the harness holds no ground truth at all (correctly — the brief forbids inventing it). It means "a human could adjudicate this deterministically" | A future reader of `cases.jsonl` may look for labels that do not exist | Rename to e.g. `humanly_adjudicable` |
| N8 | Informational | `configs/restore/nrl_vit5_base.yaml` | `min_core_base_preservation_rate: 0.9` is a magic number | Mitigated — config-visible, commented engineering-only, applies to base preservation not restoration accuracy, so it is not automatic scientific acceptance. Still arbitrary, and it is the threshold B2 trips | Revisit together with B2 |
| N9 | Informational | `~/.cache/huggingface/xet` (92 KB, logs only) | Created by the audited task's run but left in place (outside the authorised deletion scope) | Disclosed at the time; re-confirmed still present | Researcher's call: `rm -rf ~/.cache/huggingface/xet` |

---

## E. Future-compatibility findings

### Verdict on the blanket `torch` / `transformers` prohibition

```text
FAIL — restriction blanket-bans future legitimate ML modules
```

This is **B1**. The claim in the audited task's report ("one test scans every file under
`unmark/` for any torch/transformers import") is accurate, and that is precisely the
problem. Two aggravating details beyond a plain path-scope error:

1. The predicate `line.strip().startswith(...)` applies the ban at **any indentation**, so
   even a correctly-written lazy import inside a function body fails. The package is
   therefore forbidden from using the exact escape hatch the script is required to use.
2. The README states the rule as an unqualified invariant, so narrowing the test also
   requires narrowing the documentation, or the two will disagree.

**Intended scope:** *modules imported by the local G−1 unit tests* — today
`unmark/orthography/**` and `unmark/gates/**` — must be importable with only
`requirements/dev.txt`. Future `unmark/modules/`, `unmark/training/`, `unmark/baselines/`
must be free to import PyTorch.

### Other decisions that may obstruct later phases

* **Gate module cohesion.** `unmark/gates/g_minus1.py` (34 KB) bundles suite definition,
  config validation, record building, summarisation, report rendering and serialisation in
  one G−1-specific module. G0/G1/G2 will each want the summarise/serialise/report
  machinery; expect to extract a shared `unmark/gates/common.py` rather than copy it three
  times.
* **Single shared experiment pin.** `transformers==4.57.6` is a hard `==` pin in the one
  `experiment.txt` shared by all future GPU work. It is correct for the RESTORE
  checkpoint, but PhoBERT-based G1/G2 work will inherit it. If a later phase needs a newer
  transformers, the RESTORE baseline and the UNMARK backbone will need separate
  requirement files rather than one loosened pin — otherwise the "pinned, frozen restorer"
  guarantee in proposal §5.3 quietly weakens.
* **No packaging metadata.** `pyproject.toml` carries only pytest config. Fine now
  (`pythonpath = ["."]` makes `import unmark` work uninstalled); a later `pip install -e .`
  on Colab would need real metadata added.
* **Casing convention resurfaces at G0.** B2's decision will return: the proposal's G0
  pass criterion permits only NFC and tone-placement differences in
  `rec(dec(x)) = canon(x)`. Whatever casing convention is chosen for B2 should be settled
  once and shared with `canon()`, not decided twice.

---

## F. Test evidence

| Command | Result |
|---|---|
| `git status --short` / `git diff --stat` / `git diff --cached --stat` | 1 modified (`README.md`, +187/−1), 7 untracked dirs, **staged diff empty** |
| `git log --oneline -6` / `git reflog -8` | HEAD `4f39ccf update proposal`; no commit from the audited task or the audit |
| `git diff -- . ':!*.lock'` | Only removal is the placeholder line `# unmark-draft`; no unrelated changes |
| `find . -maxdepth 2 -type d \( -name .venv -o -name venv -o -name .venv-colab -o -name .hf-cache \)` | `./.venv` only — 32 MB |
| `.venv/bin/python --version` | Python 3.14.5 |
| `.venv/bin/python -c "import sys; print(sys.executable, sys.prefix)"` | `<repo>/.venv/bin/python`; prefix ≠ base_prefix |
| `.venv/bin/python -m pip list` | iniconfig, packaging, pip, pluggy, Pygments, pytest 9.1.1, PyYAML 6.0.3 |
| import probe × 8 heavy libs | torch, transformers, sentencepiece, safetensors, accelerate, huggingface_hub, tokenizers, numpy — all **absent** |
| `find . -type f \( -name '*.safetensors' -o -name '*.bin' -o -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' \)` | empty |
| `find . -path ./.git -prune -o -path ./.venv -prune -o -type f -size +20M -print` | empty |
| `git check-ignore -v .venv/test .venv-colab/test .hf-cache/test results/g_minus1/example.json results/g_minus1/.gitkeep` | first four ignored; `.gitkeep` matched by the **negation** `!results/**/.gitkeep` (line 19) → `check-ignore -q` exits non-zero → correctly **trackable** |
| `base_signature('Tôi đang học')` | `Toi dang hoc` |
| NFC vs NFD over 15 samples | identical signatures in all 15 |
| Tone / letter / `đ` / case spot-checks | `hoà`≡`hòa`; `Đường`≡`Duong`; `DUONG`≡`ĐƯỜNG`; `Đường`≢`duong`; `café`→`cafe`; `Müller`/`façade`/URLs/emails/emoji/digits unchanged |
| Idempotence + NFC-output over all 48 cases | no violations; no residual Vietnamese mark in any signature |
| Simulated capitalised+punctuated restoration on `fs_03` | `base_preserved=False`; across core cases → **ENGINEERING_SMOKE_FAIL** (rate 0.0 vs 0.9) → B2 |
| Guard predicate vs 5 future ML import forms | all 5 → `TEST FAILS`, including indented/lazy → B1 |
| `require_experiment_dependencies()` live | raises `ExperimentDependenciesMissing`; message contains the required sentence, `requirements/experiment.txt`, `Missing module(s):` |
| `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider` | **104 passed, 0 failed, 0 warnings, 0.16 s** |
| Same suite with `socket.socket` monkeypatched to raise | **104 passed** — genuinely offline |
| grep for network/model/clock/absolute-path deps in `tests/` | only a URL *string* inside a test case; `make_run_id` tested with an injected datetime; all paths `__file__`-relative |
| AST scan of `unmark/` for non-stdlib imports | `['yaml']` only |

### Python version note

No 3.10+/3.11+/3.12+-only constructs are present (`match`-statement greps were false
positives on `for case in …` loop variables). Every module carries
`from __future__ import annotations`, and there is no runtime use of PEP 604 unions, so
the lightweight code is broadly version-agnostic and **Python 3.14.5 locally is harmless**.

This says nothing about ML compatibility: torch/transformers wheels for 3.14 are not a
consideration here precisely because the policy keeps ML off this machine. Colab supplies
its own interpreter and CUDA PyTorch, and that stack remains unverified until G−1 actually
runs there.

---

## G. Git state at audit time

* **Branch:** `main`
* **Staged files:** none (`git diff --cached --stat` empty)
* **Unstaged:** `README.md` (modified, +187/−1)
* **Untracked:** `.gitignore`, `pyproject.toml`, `configs/`, `requirements/`, `results/`, `scripts/`, `tests/`, `unmark/`
* **Modifications during the audit:** no repository file was created, edited or deleted;
  `git status --short` was byte-identical before and after. One filesystem side effect is
  disclosed: running the sanctioned import and pytest commands regenerated four gitignored
  `__pycache__/` directories (`unmark/`, `unmark/gates/`, `unmark/orthography/`,
  `scripts/`). `PYTHONDONTWRITEBYTECODE=1` and `-p no:cacheprovider` were used for the test
  runs, so no `.pytest_cache/` was created.
* **Commits:** none created; no `add`, `commit`, `push`, `tag`, `stash`, `reset` or
  `checkout` was run at any point.

```text
AUDIT MODIFICATIONS MADE: NO
COMMIT CREATED: NO
```

> Note: the two lines above describe the read-only audit itself. This document was written
> to `docs/audits/` afterwards, at the researcher's explicit request.
