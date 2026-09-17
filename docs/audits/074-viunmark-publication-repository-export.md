# Audit 074 - ViUnMark Publication Repository Export

**Date:** 2026-09-17  
**Scope:** create a clean, self-contained publication export at
`release/viunmark-paper/` without destructively changing the research
repository.

## 1. Starting Snapshot

```text
HEAD                  368fad497111a30acf0407fdb0b0f2f60315b8d4
branch                main
git status --short    (empty)
git diff --stat       (empty)
```

The starting HEAD includes the committed P0 recovery work from Audit 073. No
commit, push, model training, model selection, validation read, test read,
scoring, tuning, or recalibration was performed.

```text
NEW_TRAINING=NO
MODEL_SELECTION=NO
VALIDATION_READ=NO
TEST_READ=NO
TEST_TUNING=NO
TEST_RECALIBRATION=NO
POST_TEST_RETUNING=NO
RESEARCH_REPOSITORY_DESTRUCTIVELY_CLEANED=NO
PUBLICATION_EXPORT_CREATED=YES
```

## 2. Inventory Method

Repository reconnaissance used:

* `git ls-files` over all tracked files;
* `find` directory traversal, excluding no tracked category during inspection;
* import/AST inspection of Python modules;
* ripgrep scans for old identifiers, absolute paths, parent-repo imports, and
  external-service references;
* focused inspection of `docs/spec/viunmark-final-system-v1.json`,
  `docs/spec/viunmark-historical-aliases-v1.json`,
  `docs/audits/073-final-viunmark-public-facing-system-recovery.md`,
  `unmark/viunmark/`, and ViUnMark tests.

Tracked files inspected: **327**. Filesystem traversal also observed local
generated/cache material, including `.venv`, `.pytest_cache`, `__pycache__`, and
resource caches; these were classified as generated/cache and were not exported.

## 3. Classification Totals

Machine-readable inventory:

`release/viunmark-paper/docs/repository-migration-manifest.json`

Classification totals:

```text
BINARY_OR_EXTERNAL_ASSET                 2
CORE_PUBLIC_CODE                        36
DATASET_OR_DATA_DERIVATIVE              1
EXPERIMENTAL_NOT_SELECTED               8
GENERATED_OR_CACHE                      7
HISTORICAL_MACHINE_READABLE_ARTIFACT    2
PUBLIC_CONFIG                           7
PUBLIC_TEST                            86
RESEARCH_ONLY_PROVENANCE              178
```

Action totals:

```text
COPY_EXACT    2
OMIT         42
PORT         51
SUMMARIZE   232
```

## 4. Public Export Shape

The export contains **54 files** and no symlinks:

```text
release/viunmark-paper/.gitignore
release/viunmark-paper/README.md
release/viunmark-paper/artifacts/historical/configs/viunmark-final-system-v1.json
release/viunmark-paper/artifacts/provenance/historical_aliases.json
release/viunmark-paper/artifacts/provenance/reproduction_manifest.json
release/viunmark-paper/artifacts/results/uit_vsfc/README.md
release/viunmark-paper/configs/uit_vsfc/evaluation.json
release/viunmark-paper/configs/uit_vsfc/paper.json
release/viunmark-paper/configs/uit_vsfc/training.json
release/viunmark-paper/docs/data.md
release/viunmark-paper/docs/method.md
release/viunmark-paper/docs/provenance.md
release/viunmark-paper/docs/repository-migration-manifest.json
release/viunmark-paper/docs/reproduction.md
release/viunmark-paper/docs/results.md
release/viunmark-paper/notebooks/ViUnMark_Reproduction.ipynb
release/viunmark-paper/pyproject.toml
release/viunmark-paper/requirements.txt
release/viunmark-paper/scripts/build_representations.py
release/viunmark-paper/scripts/predict.py
release/viunmark-paper/scripts/prepare_uit_vsfc.py
release/viunmark-paper/scripts/reproduce_uit_vsfc.py
release/viunmark-paper/scripts/run_diagnostic.py
release/viunmark-paper/scripts/score_predictions.py
release/viunmark-paper/scripts/train_readouts.py
release/viunmark-paper/scripts/verify_assets.py
release/viunmark-paper/src/viunmark/__init__.py
release/viunmark-paper/src/viunmark/adapters.py
release/viunmark-paper/src/viunmark/cli.py
release/viunmark-paper/src/viunmark/config.py
release/viunmark-paper/src/viunmark/corruption.py
release/viunmark-paper/src/viunmark/diagnostics/__init__.py
release/viunmark-paper/src/viunmark/diagnostics/complementarity.py
release/viunmark-paper/src/viunmark/diagnostics/decision_geometry.py
release/viunmark-paper/src/viunmark/diagnostics/gain_factorization.py
release/viunmark-paper/src/viunmark/diagnostics/pooling_bridge.py
release/viunmark-paper/src/viunmark/diagnostics/pooling_comparison.py
release/viunmark-paper/src/viunmark/diagnostics/scale_pathway_preflight.py
release/viunmark-paper/src/viunmark/evaluation.py
release/viunmark-paper/src/viunmark/fusion.py
release/viunmark-paper/src/viunmark/heads.py
release/viunmark-paper/src/viunmark/losses.py
release/viunmark-paper/src/viunmark/orthography.py
release/viunmark-paper/src/viunmark/provenance.py
release/viunmark-paper/src/viunmark/readouts.py
release/viunmark-paper/src/viunmark/system.py
release/viunmark-paper/src/viunmark/training.py
release/viunmark-paper/tests/test_cli.py
release/viunmark-paper/tests/test_diagnostics.py
release/viunmark-paper/tests/test_fusion.py
release/viunmark-paper/tests/test_method.py
release/viunmark-paper/tests/test_provenance.py
release/viunmark-paper/tests/test_public_surface.py
release/viunmark-paper/tests/test_training.py
```

No cache directories remain in the export.

## 5. Public Dependency Closure

The public package namespace is `viunmark` under `src/viunmark/`. It has no
runtime imports from `unmark.*`, research scripts, audit infrastructure, or
historical campaign runners.

Self-contained public code covers:

* ViUnMark-Gate and ViUnMark-Scale method configuration;
* seven tone rows and five letter rows, with non-applicable channel sentinels
  outside both tables;
* learned tone/letter embeddings, `Linear(3d,d) + LayerNorm`, gate
  `Linear(3d,d)`, gate weight `0`, gate bias `logit(0.01)`;
* `FIRST_TOKEN`, `MASKED_MEAN`, and `CONCAT = [FIRST_TOKEN ; MASKED_MEAN]`;
* robust MLP architecture and initialization/optimizer policy facts;
* unweighted CE and sqrt-inverse-frequency CE rule;
* five-head branch arithmetic means, fixed branch weights, and no parent-bias
  stacking;
* UIT-VSFC dataset-specific calibration as provenance/config, not a method
  default;
* scoring sealed prediction artifacts separately from prediction generation.

The generic method remains calibration-neutral.

## 6. Omitted Or Summarized Material

Omitted or summarized categories:

* historical audits and decisions: summarized in public docs/provenance;
* old Colab cells and research notebooks: omitted; one new orchestrator notebook
  was authored;
* Stage-I training runners, Stage-II campaign infrastructure, monitoring,
  baseline restore code, and old diagnostic runners: omitted from runtime;
* proposal PDF/zip and external assets: omitted;
* results placeholders/caches and generated Python caches: omitted;
* dataset-like fixtures: not part of the publication dependency closure.

## 7. Historical Artifacts And Provenance

Copied exactly:

* `docs/spec/viunmark-final-system-v1.json` ->
  `artifacts/historical/configs/viunmark-final-system-v1.json`;
* `docs/spec/viunmark-historical-aliases-v1.json` ->
  `artifacts/provenance/historical_aliases.json`.

Known external artifact SHA-256 identities are recorded in
`artifacts/provenance/reproduction_manifest.json` with
`present_in_public_repo: false`. No external file was fabricated.

Old research identifiers are absent from normal code/docs/CLI. They remain only
in historical/provenance files, `docs/provenance.md`, the migration manifest
where original source paths must be recorded, and tests that enforce the rule.

## 8. Public Entry Points

Primary:

```text
scripts/reproduce_uit_vsfc.py
```

Supporting:

```text
scripts/verify_assets.py
scripts/prepare_uit_vsfc.py
scripts/build_representations.py
scripts/train_readouts.py
scripts/predict.py
scripts/score_predictions.py
scripts/run_diagnostic.py
viunmark CLI module
```

Diagnostic command names:

```text
scale-preflight
pooling-bridge
pooling-comparison
decision-geometry
complementarity
gain-factorization
adapted-only
phobert-readout
viunmark
```

No old diagnostic/system identifiers are exposed as public commands.

## 9. Notebook

New notebook:

```text
notebooks/ViUnMark_Reproduction.ipynb
```

It is an orchestrator. It records environment info, configurable data/asset/output
roots, package installation, asset verification, frozen reproduction verification,
and a diagnostic call. It contains no duplicate model, loss, fusion, corruption,
or scoring implementation and no personal Drive path.

## 10. Scientific Invariants Preserved

Preserved exactly:

* final system: ViUnMark;
* public pathways: ViUnMark-Gate, ViUnMark-Scale, native PhoBERT pathway;
* readouts: Gate Robust Readout, Scale Unweighted Readout, Scale Weighted
  Readout, PhoBERT Readout;
* exactly five selected heads per branch;
* arithmetic mean of logits within each branch;
* no best-seed selection;
* final raw branch weights: PhoBERT `0.500`, Gate `0.250`, Scale Unweighted
  `0.125`, Scale Weighted `0.125`;
* parent biases never stacked;
* UIT-VSFC final calibration: class index `1` additive bias `+1.25`;
* Adapted-Only Fusion remains diagnostic/ablation, with UIT-VSFC diagnostic
  calibration `+0.75` only;
* method/protocol recovery remains distinct from bit-exact historical execution
  recovery.

## 11. Unresolved Details Preserved

Not filled in:

* robust-MLP cross-entropy reduction;
* low-level RNG and batch mechanics;
* historical state-dict layout details;
* native PhoBERT fields not established by its protocol;
* numeric result tables, because the research repository contains only frozen
  evidence identities, not committed result-table records;
* software license.

`PUBLIC_CODE_LICENSE = UNRESOLVED`.

## 12. Validation

From `release/viunmark-paper/`:

```text
python -m compileall src scripts                 PASS
env PYTHONPATH=src python -c "import viunmark"   PASS
env PYTHONPATH=src python -m viunmark.cli --help PASS
env PYTHONPATH=src python scripts/verify_assets.py --config configs/uit_vsfc/paper.json PASS
python -m json.tool notebooks/ViUnMark_Reproduction.ipynb PASS
env PYTHONPATH=src pytest -q                     15 passed
```

Scans:

```text
symlink scan                                     0 symlinks
cache-dir scan                                  0 cache dirs after cleanup
parent-repo import scan under src/viunmark       0 hits
absolute personal path scan                      0 public hits outside tests/manifest guard strings
old-name scan                                   hits only in historical/provenance, migration manifest, and public-surface test token list
```

Original research repository regression:

* sandboxed full run: `5056 passed, 273 skipped, 7 failed`; the 7 failures were
  all `tests/test_stage1_parallel.py` multiprocessing forkserver socket binds
  blocked by sandbox permissions;
* escalated full rerun: `5063 passed, 273 skipped`.

No torch installation was performed. Existing environment capabilities were used.

## 13. Final State

Final `git status --short` before writing this audit showed:

```text
?? release/
```

Final `git diff --stat` before writing this audit was empty because all release
files were untracked at that point.

Expected final status after this audit:

```text
?? docs/audits/074-viunmark-publication-repository-export.md
?? release/
```

The author should review the release tree, the unresolved license, external
asset publication plan, and result-table publication plan before seeding a public
repository.
