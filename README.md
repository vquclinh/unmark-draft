# UNMARK

**Tone-Factored Input Adaptation for Diacritic-Robust Vietnamese Language Understanding**

The working specification is [`unmark-proposal.md`](unmark-proposal.md). The repository is
currently at **Phase 0 / Gate G−1**: nothing from UNMARK itself is implemented yet.

---

## Development environments

This project deliberately splits work across two machines. The split is a policy, not a
convenience — the local checkout must stay small and disposable, and the local machine
must never end up holding model weights.

| | Local machine | Google Colab |
|---|---|---|
| Environment | `.venv/` | `.venv-colab/` |
| Requirements | `requirements/dev.txt` | `requirements/experiment.txt` |
| Purpose | coding, git, configs, Unicode/orthography utilities, unit tests, inspecting result files | Hugging Face model inference, G−1, later GPU gates and training |
| Heavy ML libraries | none | torch (from the runtime), transformers, sentencepiece, safetensors, accelerate |
| Model checkpoints | never downloaded | downloaded at runtime into `.hf-cache/` |

Both environments live **inside the repository** and are disposable. Neither is committed.

### Local machine

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/dev.txt
pytest -q
```

Whatever `python3` your machine provides is fine. The local code is pure standard library
plus PyYAML and carries `from __future__ import annotations` throughout, so it is not
sensitive to the interpreter version — this checkout runs its suite on Python 3.14.

That says nothing about ML compatibility, and it is not meant to: **the interpreter that
has to satisfy torch and transformers is Colab's, not yours.** Local Python is decoupled
from the ML runtime by design, which is exactly why no heavy dependency is installed here.

No model checkpoints and no heavy ML dependencies are needed locally. The test suite runs
in well under a second.

Things that work locally without the model:

```bash
pytest -q                                            # full unit test suite
python scripts/g_minus1_restore_smoke.py --list-cases # inspect the smoke suite
```

Running the real smoke test locally is refused on purpose, with a message pointing at
Colab. Nothing is ever installed automatically.

### Google Colab

Use a **T4 GPU** runtime (Runtime → Change runtime type → T4 GPU).

Colab already ships a CUDA-enabled PyTorch, so the environment is created with
`--system-site-packages` and `requirements/experiment.txt` deliberately does **not** pin
torch — pinning it would trigger a multi-gigabyte reinstall of a CUDA build that is
already present.

`HF_HOME` is redirected into the checkout so the entire download is disposable: after the
experiment, deleting one directory reclaims all of it.

---

## G−1 RESTORE smoke test on Google Colab

Paste this into a Colab cell on a T4 GPU runtime. Replace the clone URL with your own.

```bash
%%bash
set -e

git clone https://github.com/<your-account>/unmark-draft.git
cd unmark-draft

# Keep every Hugging Face download inside the checkout, so cleanup is one rm.
export HF_HOME="$PWD/.hf-cache"

# Reuse Colab's CUDA PyTorch; install only the rest.
python -m venv .venv-colab --system-site-packages
.venv-colab/bin/python -m pip install --upgrade pip
.venv-colab/bin/python -m pip install -r requirements/experiment.txt

# Confirm the GPU is visible from inside the venv.
.venv-colab/bin/python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# Run the gate.
HF_HOME="$PWD/.hf-cache" .venv-colab/bin/python scripts/g_minus1_restore_smoke.py
```

The script prints the selected device, streams each restoration as it happens, then prints
a per-category table and the path to the run directory.

Optional overrides:

```bash
HF_HOME="$PWD/.hf-cache" .venv-colab/bin/python scripts/g_minus1_restore_smoke.py \
    --config configs/restore/nrl_vit5_base.yaml \
    --output-root results/g_minus1
```

`HF_HOME` is set inline on purpose. An `export` in one Colab `%%bash` cell does not carry
into the next cell, and without it the checkpoint lands in the shared Colab cache instead
of `.hf-cache/` — which breaks the one-command cleanup below. Prefix every command that
can load or download the model.

Download the results to your machine before the runtime is recycled — for example
`results/g_minus1/<run_id>/report.md`.

### What the gate produces

Each run writes a timestamped directory `results/g_minus1/<run_id>/` containing:

| File | Contents |
|---|---|
| `config.json` | model id, pinned revision, generation parameters, repeats, device, torch / transformers / Python versions, timestamp |
| `cases.jsonl` | one record per test case: input, all repeated outputs, determinism, base signatures, base preservation, clean-exact preservation, latency, error |
| `summary.json` | counts, determinism / base-preservation / clean-exact rates, rates by category, latency statistics |
| `report.md` | human-readable report ending in a `G-1 Assessment` section |

Two preservation measurements are recorded per case and never merged:

| | Compares | Sensitive to | Used for |
|---|---|---|---|
| `base_preserved` (strict) | `base_signature` | case, punctuation, every word | the record of what the restorer actually changed |
| `rewrite_preserved` (engineering) | `rewrite_signature` | every word; *not* case, *not* a trailing `.!?…` | the `no_catastrophic_lexical_rewriting` check |

A restorer that turns `hom nay troi dep` into `Hôm nay trời đẹp.` has changed formatting,
not vocabulary: strict preservation reads false, lexical preservation reads true, and the
case appears under *Formatting-only differences* in the report. Substituted, inserted or
deleted words fail both.

The script reports an **engineering** status (`ENGINEERING_SMOKE_PASS` /
`ENGINEERING_SMOKE_FAIL`) based only on whether the model loaded, whether ordinary
Vietnamese ran without error, whether greedy decoding was deterministic, and whether the
lexical content survived. It does **not** score restoration accuracy and it does **not**
decide G−1 — that decision stays with the researcher, and `report.md` closes with the
checklist for making it.

### Model / library compatibility note

The G−1 checkpoint is pinned:

```text
nrl-ai/vn-diacritic-vit5-base @ 30ea5a9e4a0b9436e18915fd4dbb5876eaee7325
```

It was serialised by **transformers 4.57.6**, and `requirements/experiment.txt` pins that
version. Transformers 5.x cannot load this checkpoint's `tokenizer.json` and fails with:

```text
TypeError: argument 'vocab': 'dict' object cannot be converted to 'Sequence'
```

This was observed directly, not assumed. The script warns if it detects a mismatched
major version.

---

## Cleanup

Every environment and cache in this project is disposable and repository-local.

Local development tooling:

```bash
rm -rf .venv
```

Colab experiment environment and downloaded model weights:

```bash
rm -rf .venv-colab .hf-cache
```

Nothing here touches shared or system-wide caches.

---

## Repository layout

```text
configs/
  restore/
    nrl_vit5_base.yaml        # LOCKED G-1 model config: checkpoint, revision, decoding
requirements/
  base.txt                    # shared lightweight dependencies
  dev.txt                     # local: base + pytest
  experiment.txt              # Colab: base + transformers==4.57.6 + tokenizer stack
scripts/
  g_minus1_restore_smoke.py   # G-1 runtime entry point (Colab; lazy torch/transformers)
tests/
  test_orthography_signature.py
  test_restore_smoke_utils.py
unmark/
  orthography/
    signature.py              # base_signature (strict) + rewrite_signature (engineering)
  gates/
    g_minus1.py               # smoke suite, config validation, records, summary, report
results/
  g_minus1/                   # run artifacts (git-ignored except .gitkeep)
```

### Which modules must stay lightweight

A named set of modules must be importable with `requirements/dev.txt` alone, because the
offline test suite imports them:

```text
unmark.orthography.signature
unmark.gates.g_minus1
```

`tests/test_restore_smoke_utils.py` enforces this *behaviourally*: it spawns a clean
subprocess from `.venv`, imports each listed module, and asserts that none of torch,
transformers, sentencepiece, safetensors or accelerate ended up in `sys.modules`. The list
is explicit, so adding a module to it is a deliberate act.

This is a rule about **those modules**, not about the package. Later phases will add
genuinely ML-shaped code — `unmark/modules/`, `unmark/training/`, `unmark/baselines/` in
proposal §8.1 — and **those modules are free to import PyTorch normally**. They simply do
not belong on the lightweight list, and the local suite will not import them.

`scripts/g_minus1_restore_smoke.py` is a separate case: it does use torch and
transformers, but only inside the model path, so the CLI itself still imports on a machine
without them.
