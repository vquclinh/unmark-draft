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

## Orthography core (B1A)

The deterministic layer UNMARK's input channels are built from. Pure standard
library; no model, no word list, no network.

```python
from unmark.orthography import canon, decompose, recompose

parts = decompose("Đường Nguyễn Huệ")
parts.base_text            # 'Duong Nguyen Hue'
recompose(parts) == canon("Đường Nguyễn Huệ")   # True
```

### Three channels, two granularities

| Channel | Granularity | States |
|---|---|---|
| base | character | the letters with every Vietnamese diacritic removed |
| tone | **syllable** | `UNMARKED`, `SAC`, `HUYEN`, `HOI`, `NGA`, `NANG` |
| letter diacritic | **character** | `NONE`, `BREVE`, `CIRCUMFLEX`, `HORN`, `STROKE`, `NA` |

The granularities differ for a reason (proposal §4.3): a syllable carries exactly one
tone, but may carry several letter diacritics on different characters — `được` is
`đ`+stroke, `ư`+horn, `ợ`+horn, with one nặng tone for the whole syllable. A per-syllable
letter channel would be lossy and would break reconstruction.

`NONE` and `NA` are not the same. `NONE` is a letter that could carry a Vietnamese letter
diacritic and does not; `NA` means the channel does not apply at all — space, digit,
punctuation, emoji.

### Clean lexical tone vs observable tone

This distinction is the point of the whole project, so it is enforced at the type level
with two separate enums.

| | `Tone` (lexical) | `ObservedTone` (deployable) |
|---|---|---|
| Values | `NGANG`, `SAC`, `HUYEN`, `HOI`, `NGA`, `NANG` | `UNMARKED`, `SAC`, `HUYEN`, `HOI`, `NGA`, `NANG` |
| Means | the syllable's true tone | what is visible in the string |
| Available at inference | no | yes |

**`UNMARKED` is not `NGANG`.** A syllable with no tone mark is either a genuine *ngang*
or a syllable whose mark was stripped, and the orthography cannot tell which — the sixth
Vietnamese tone is written with no mark at all (proposal §1.2). So:

```python
decompose("ma").syllables[0].observed_tone   # ObservedTone.UNMARKED
decompose("ma").syllables[0].lexical_tone    # None  -- genuinely unknowable

decompose("ma", source_is_clean=True).syllables[0].lexical_tone   # Tone.NGANG
```

`lexical_tone` is `None` for unmarked syllables unless the caller explicitly asserts the
source text is clean. A *visible* mark settles the lexical tone either way. This keeps the
H4 oracle policy (genuine *ngang* vs `MISSING`) implementable later without letting the
inference-time path ever assume it.

### The invariant

```text
recompose(decompose(x)) == canon(x)
canon(canon(x)) == canon(x)
```

`canon` is a reconstruction target, not a comparison form: it applies NFC and nothing
else — no whitespace collapsing, no case folding, no punctuation rewriting. (Contrast
`base_signature` in the same package, which does collapse whitespace because it serves the
G−1 diagnostic.)

### Canonical tone placement

Vietnamese admits two accepted positions for a tone mark over a vowel cluster. UNMARK
fixes one as **its canonical convention**, so that `canon` is a function of the word and
experiments are reproducible:

```python
canon("hòa")  == "hoà"     # nucleus-based: the glide `o` does not take the tone
canon("thúy") == "thuý"
canon("khỏe") == "khoẻ"
canon("qùa")  == "quà"     # in `qu-` the u belongs to the onset
canon("mùa")  == "mùa"     # `u` is not a glide before `a`
canon("được") == "được"    # in `ươ` the tone goes on the second vowel
```

This is a project canonicalisation convention adopted for reproducibility. It is **not**
a claim that the other convention is wrong or that this is the sole official Vietnamese
orthography — both occur in real text. The decision, the rule and its rationale are
recorded in [`docs/spec/orthography.md`](docs/spec/orthography.md) (D-001).

The rule works on syllable structure — onset digraphs (`qu-`, `gi-`), the medial glide,
letter diacritics marking the nucleus, then coda presence — not on a word list. Only the
*position* of a tone mark within its own vowel cluster ever changes: letters, case,
punctuation, whitespace, digits, URLs and e-mail addresses are untouched, and `ă â ê ô ơ ư`
never move off their base letter.

`TonePlacement.PRESERVE` remains available as an explicit **diagnostic** mode that leaves
tone marks where the input put them; it is no longer the default.
`TonePlacement.TRADITIONAL` is not implemented and raises.

### Deferred: Vietnamese-candidate eligibility (GAP-2)

Proposal §4.3 decides candidacy by matching "the Vietnamese syllable inventory after
stripping", but that inventory is not enumerated in the proposal and does not exist here.
Every alphabetic span therefore reports `Eligibility.UNDECIDED` rather than being guessed
from a word list — `ban`, `AI`, `machine` are observationally ambiguous at this layer.
Resolution is deferred to the B3 / input-policy stage; see `docs/spec/orthography.md`
(D-002). Whatever rule is adopted must stay a pure function of the *stripped* form so that
clean and corrupted input get identical labels, and a test enforces that the code never
decides from the presence of diacritics.

### G0 round-trip checker

```bash
.venv/bin/python scripts/g0_orthography_check.py --self-check
.venv/bin/python scripts/g0_orthography_check.py --input corpus.txt --max-samples 100000
```

Writes `results/g0/<run_id>/` with `config.json`, `summary.json`, `failures.jsonl` and
`report.md`, separating Unicode-normalisation differences from tone-placement collapsing
so neither hides inside the other. It reports at most
`ORTHOGRAPHY_CORE_READY_FOR_G0_CORPUS_CHECK` and never `G0 PASS`: G0 needs ≥100K sentences
of real Vietnamese, no corpus ships with this repository, and nothing here downloads one.

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
  g0_orthography_check.py     # G0 round-trip checker (local, no corpus ships here)
tests/
  test_orthography_signature.py
  test_orthography_decompose.py
  test_restore_smoke_utils.py
unmark/
  orthography/
    marks.py                  # mark inventories, tone/letter channel states
    units.py                  # base-char + combining-mark grouping (shared)
    placement.py              # nucleus-based canonical tone placement
    models.py                 # CharacterUnit, SyllableSpan, DecomposedText
    canonical.py              # canon() = NFC + fixed tone placement
    decompose.py              # decompose() / recompose()
    signature.py              # base_signature (strict) + rewrite_signature (engineering)
  gates/
    g_minus1.py               # smoke suite, config validation, records, summary, report
docs/
  spec/orthography.md         # orthography decisions (D-001 placement, D-002 eligibility)
  audits/                     # persisted audits
results/
  g_minus1/                   # run artifacts (git-ignored except .gitkeep)
  g0/                         # run artifacts (git-ignored except .gitkeep)
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
