# Audit 075 - ViUnMark Publication Reproduction Hardening

## Starting Provenance

- Command date: 2026-09-17.
- Starting HEAD: `d82a8dd1303b398fa182bbb7089bd9930a1d62f0`.
- Starting `git status --short`: clean.
- Starting `git diff --stat`: empty.
- Requirement for clean starting tree: satisfied.

Explicit controls:

- `NEW_TRAINING=NO`
- `MODEL_SELECTION=NO`
- `VALIDATION_READ=NO`
- `TEST_READ=NO`
- `TEST_TUNING=NO`
- `TEST_RECALIBRATION=NO`
- `POST_TEST_RETUNING=NO`

No commit and no push were performed.

## Scope

This audit treated `release/viunmark-paper/` from Audit 074 as the starting
publication layout. Work was limited to reproduction hardening: asset contracts,
CLI truthfulness, README/notebook correctness, result-artifact handling, tests,
and publication blocker documentation.

No historical scientific files outside the public export were modified.

## Files Changed

Modified release files:

- `release/viunmark-paper/README.md`
- `release/viunmark-paper/artifacts/provenance/reproduction_manifest.json`
- `release/viunmark-paper/docs/reproduction.md`
- `release/viunmark-paper/docs/results.md`
- `release/viunmark-paper/notebooks/ViUnMark_Reproduction.ipynb`
- `release/viunmark-paper/scripts/reproduce_uit_vsfc.py`
- `release/viunmark-paper/scripts/score_predictions.py`
- `release/viunmark-paper/scripts/verify_assets.py`
- `release/viunmark-paper/src/viunmark/cli.py`
- `release/viunmark-paper/tests/test_cli.py`
- `release/viunmark-paper/tests/test_public_surface.py`

Added release files:

- `release/viunmark-paper/artifacts/results/uit_vsfc/missing_result_artifacts.json`
- `release/viunmark-paper/docs/publication-readiness.md`
- `release/viunmark-paper/scripts/render_results_table.py`
- `release/viunmark-paper/src/viunmark/assets.py`
- `release/viunmark-paper/tests/test_asset_contract.py`

Added research audit file:

- `docs/audits/075-viunmark-publication-reproduction-hardening.md`

## Reproduction Modes Supported

The high-level runner `scripts/reproduce_uit_vsfc.py` was narrowed to advertise
only implemented behavior:

- `--mode frozen`
- `--stage verify`
- `--stage diagnostics`

Removed from the advertised high-level runner surface:

- `retrain-readouts`
- `prepare`
- `representations`
- `train-readouts`
- `predict`
- `score`
- `all`

Reason: these stages were placeholders or require unavailable external assets
and unrecovered historical execution choices. The public runner no longer
pretends that they are implemented.

The lower-level scripts remain present with explicit boundaries:

- `verify_assets.py`: implemented; verifies the frozen model asset contract.
- `run_diagnostic.py`: implemented public diagnostic dispatch.
- `predict.py`: intentionally stops until external frozen assets are supplied.
- `score_predictions.py`: implemented only for sealed prediction artifacts.
- `train_readouts.py`: intentionally stops until author-reviewed public policy
  choices are supplied.
- `build_representations.py` and `prepare_uit_vsfc.py`: utility placeholders
  retained as public entry points but not advertised as full reproduction by the
  high-level runner.

## Asset Contract

The public frozen-model contract now requires exactly 22 external model assets:

- 2 Stage-I checkpoints:
  - `ViUnMark-Gate`, SHA-256
    `6773fbb59c7381ba8ddaa944302124a124f5b8a5cb0a5dbb1a5063f3db4a2a91`,
    selected update `3500`.
  - `ViUnMark-Scale`, SHA-256
    `a32c0167817d457d5067c2a351f2d1b73b26229033f03727f43e2d79f59ef685`,
    selected update `8000`.
- 20 Stage-II readout heads:
  - 5 `Gate Robust Readout` heads.
  - 5 `Scale Unweighted Readout` heads.
  - 5 `Scale Weighted Readout` heads.
  - 5 `PhoBERT Readout` heads.

For every readout head, the public manifest records:

- relative expected path under the configured asset root;
- SHA-256;
- asset role;
- selected seed;
- selected boundary.

The asset verifier behavior is now:

- `--manifest-only`: prints expected layout and exits `0` without requiring
  files.
- normal verification: checks file presence and SHA-256 for all 22 assets.
- missing assets: exits nonzero with a concise stderr summary and detailed JSON.
- SHA mismatch: exits nonzero with expected and actual SHA-256 in JSON.

Observed no-assets behavior in this environment:

- `python scripts/verify_assets.py --config configs/uit_vsfc/paper.json`
  exited `1` with `22 missing, 0 sha256 mismatched`, as expected because
  checkpoints are not shipped.

## Result-Artifacts Status

The committed research repository was searched for authoritative
machine-readable paper result files. Exact frozen result JSON/CSV files were not
present in Git.

No values were reconstructed or hand-entered.

Added:

- `artifacts/results/uit_vsfc/missing_result_artifacts.json`

This file records known SHA-256 identities from the authoritative final-system
provenance and marks each canonical numeric result source as unavailable in the
current repository. It explicitly requires author-supplied canonical artifacts
before publication result tables are rendered.

Added:

- `scripts/render_results_table.py`

This utility renders tables only from an author-approved machine-readable result
artifact. It does not read labels and does not calculate paper metrics.

## README Command Audit

README commands now separate:

- frozen asset contract inspection with `--manifest-only`;
- real frozen asset verification;
- high-level frozen verification through `reproduce_uit_vsfc.py`;
- sealed prediction scoring;
- diagnostics.

Parser/import smoke tests were added for README script commands.

The README now states that readout retraining is not advertised by the current
high-level runner because the remaining public implementation choices need
author review.

## Notebook Audit

Notebook inspected:

- `release/viunmark-paper/notebooks/ViUnMark_Reproduction.ipynb`

Status:

- Orchestrator only.
- No duplicated model implementation.
- No historical research code.
- No personal paths.
- Uses configurable `DATA_ROOT`, `ASSET_ROOT`, `OUTPUT_ROOT`, and `CONFIG`.
- Calls public scripts.
- Checks asset manifest before the real asset verification.
- The real verification cell stops clearly if assets are unavailable.
- Does not download unverified assets.
- Does not score before prediction sealing.

Structural validation:

- `python -m json.tool notebooks/ViUnMark_Reproduction.ipynb`: pass.

## Diagnostic Executability Matrix

| Analysis | Status | Notes |
| --- | --- | --- |
| `scale-preflight` | `FULLY_EXECUTABLE` | Validates public ViUnMark-Scale configuration. |
| `pooling-bridge` | `FULLY_EXECUTABLE` for public interface | Reports supported same-forward pooling bridge behavior. |
| `pooling-comparison` | `EXECUTABLE_WITH_EXTERNAL_ASSETS` for numerical reproduction | Public command exists; matched numerical reproduction needs frozen assets. |
| `decision-geometry` | `EXECUTABLE_WITH_EXTERNAL_ASSETS` | Geometry helpers exist; paper-scale analysis needs frozen representations/logits. |
| `complementarity` | `EXECUTABLE_WITH_EXTERNAL_ASSETS` | Public command exists; numerical complementarity needs frozen logits. |
| `gain-factorization` | `EXECUTABLE_WITH_EXTERNAL_ASSETS` | Public command exists; complete numerical factors need frozen evidence. |
| `adapted-only` | `REPORT_ONLY_FROM_FROZEN_ARTIFACT` until assets/results are supplied | No new metric reconstruction. |
| `phobert-readout` | `REPORT_ONLY_FROM_FROZEN_ARTIFACT` until assets/results are supplied | No historical selection rerun. |
| `viunmark` | `EXECUTABLE_WITH_EXTERNAL_ASSETS` | Final fusion semantics are implemented; frozen reproduction requires assets. |

## License And Third-Party Status

- Project software license: no root repository license was found.
- `PUBLIC_CODE_LICENSE=UNRESOLVED`.
- PhoBERT: public config records the `vinai/phobert-base` model identity; final
  publication should confirm license/reference language against the upstream
  model card.
- UIT-VSFC: raw data is not redistributed; users supply it under its own terms.
- Checkpoints: not shipped; redistribution channel and license terms remain an
  author action.
- No third-party binary assets were copied into the public export in this audit.

## Publication Blocker Matrix

Created:

- `release/viunmark-paper/docs/publication-readiness.md`

It covers:

- source code;
- README;
- notebook;
- configs;
- unit tests;
- Stage-I checkpoints;
- 20 Stage-II heads;
- dataset;
- official result records;
- diagnostics;
- software license;
- citation metadata;
- PhoBERT attribution;
- checkpoint redistribution.

Primary blockers before public release:

- `PUBLIC_CODE_LICENSE=UNRESOLVED`.
- External frozen model assets are not shipped.
- Canonical numeric paper result artifacts are not present in current Git.
- Checkpoint redistribution terms/channel need author decision.
- Citation metadata is pending.

## Public Test Results

Commands run from `release/viunmark-paper/`:

- `python -m compileall src scripts`: pass.
- `env PYTHONPATH=src python -c 'import viunmark; print(viunmark.__name__)'`:
  pass.
- `env PYTHONPATH=src python -m viunmark.cli --help`: pass.
- `env PYTHONPATH=src python scripts/verify_assets.py --config configs/uit_vsfc/paper.json --manifest-only`:
  pass.
- `env PYTHONPATH=src python scripts/reproduce_uit_vsfc.py --help`: pass.
- `env PYTHONPATH=src python scripts/run_diagnostic.py --analysis scale-preflight`:
  pass.
- `python -m json.tool notebooks/ViUnMark_Reproduction.ipynb`: pass.
- `env PYTHONPATH=src pytest -q`: `24 passed in 1.16s`.

Expected failure check:

- `env PYTHONPATH=src python scripts/verify_assets.py --config configs/uit_vsfc/paper.json`:
  exits `1` because all 22 external assets are absent; failure message is
  actionable and includes detailed missing-asset records.

One initial import smoke command attempted to print `viunmark.__version__` and
failed because the package does not expose `__version__`. This was a smoke
command mistake, not a package import failure; the corrected import smoke passed.

## Hygiene Scan Results

Parent-repository import scan:

- No `from unmark` or `import unmark` imports in public `src/` or `scripts/`.
- No public runtime import from parent research packages was found.

Symlink scan:

- No symlinks under `release/viunmark-paper/`.

Generated/cache scan:

- No `__pycache__`, `.pyc`, or `.pytest_cache` remains under the release tree.

Old-name scan:

- Public-surface hits were limited to approved exception zones:
  - `artifacts/provenance/historical_aliases.json`;
  - `docs/provenance.md`;
  - `docs/repository-migration-manifest.json`;
  - `tests/test_public_surface.py`.
- No old research identifiers were found in README, normal public docs, public
  module names, CLI commands, or notebook cells outside those exceptions.

Absolute/personal path scan:

- No `/mnt/`, `/home/`, `/content/drive`, or `MyDrive` paths were found in
  public files except the test file that asserts notebook path hygiene.

## Original Repository Regression

First sandboxed run:

- `pytest -q`: failed only in `tests/test_stage1_parallel.py`.
- Failure cause: sandbox blocked multiprocessing forkserver socket bind with
  `PermissionError: [Errno 1] Operation not permitted`.
- Summary: `7 failed, 5056 passed, 273 skipped in 151.34s`.

Escalated rerun outside the sandbox:

- `pytest -q`: `5063 passed, 273 skipped in 171.29s`.

The release hardening did not regress the original research test suite.

## Final Release Tree

```text
release/viunmark-paper
release/viunmark-paper/.gitignore
release/viunmark-paper/README.md
release/viunmark-paper/artifacts
release/viunmark-paper/artifacts/historical
release/viunmark-paper/artifacts/historical/configs
release/viunmark-paper/artifacts/historical/configs/viunmark-final-system-v1.json
release/viunmark-paper/artifacts/historical/results
release/viunmark-paper/artifacts/provenance
release/viunmark-paper/artifacts/provenance/historical_aliases.json
release/viunmark-paper/artifacts/provenance/reproduction_manifest.json
release/viunmark-paper/artifacts/results
release/viunmark-paper/artifacts/results/uit_vsfc
release/viunmark-paper/artifacts/results/uit_vsfc/README.md
release/viunmark-paper/artifacts/results/uit_vsfc/missing_result_artifacts.json
release/viunmark-paper/configs
release/viunmark-paper/configs/uit_vsfc
release/viunmark-paper/configs/uit_vsfc/evaluation.json
release/viunmark-paper/configs/uit_vsfc/paper.json
release/viunmark-paper/configs/uit_vsfc/training.json
release/viunmark-paper/docs
release/viunmark-paper/docs/data.md
release/viunmark-paper/docs/method.md
release/viunmark-paper/docs/provenance.md
release/viunmark-paper/docs/publication-readiness.md
release/viunmark-paper/docs/repository-migration-manifest.json
release/viunmark-paper/docs/reproduction.md
release/viunmark-paper/docs/results.md
release/viunmark-paper/notebooks
release/viunmark-paper/notebooks/ViUnMark_Reproduction.ipynb
release/viunmark-paper/pyproject.toml
release/viunmark-paper/requirements.txt
release/viunmark-paper/scripts
release/viunmark-paper/scripts/build_representations.py
release/viunmark-paper/scripts/predict.py
release/viunmark-paper/scripts/prepare_uit_vsfc.py
release/viunmark-paper/scripts/render_results_table.py
release/viunmark-paper/scripts/reproduce_uit_vsfc.py
release/viunmark-paper/scripts/run_diagnostic.py
release/viunmark-paper/scripts/score_predictions.py
release/viunmark-paper/scripts/train_readouts.py
release/viunmark-paper/scripts/verify_assets.py
release/viunmark-paper/src
release/viunmark-paper/src/viunmark
release/viunmark-paper/src/viunmark/__init__.py
release/viunmark-paper/src/viunmark/adapters.py
release/viunmark-paper/src/viunmark/assets.py
release/viunmark-paper/src/viunmark/cli.py
release/viunmark-paper/src/viunmark/config.py
release/viunmark-paper/src/viunmark/corruption.py
release/viunmark-paper/src/viunmark/diagnostics
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
release/viunmark-paper/tests
release/viunmark-paper/tests/test_asset_contract.py
release/viunmark-paper/tests/test_cli.py
release/viunmark-paper/tests/test_diagnostics.py
release/viunmark-paper/tests/test_fusion.py
release/viunmark-paper/tests/test_method.py
release/viunmark-paper/tests/test_provenance.py
release/viunmark-paper/tests/test_public_surface.py
release/viunmark-paper/tests/test_training.py
```

## Final Status Snapshot

`git status --short` before writing this audit showed only release-tree edits
and additions. This audit file itself is added by this task and will appear in
the final status.

Tracked-file `git diff --stat` before writing this audit:

```text
 release/viunmark-paper/README.md                   |  32 +-
 .../provenance/reproduction_manifest.json          | 345 ++++++++++++++++++++-
 release/viunmark-paper/docs/reproduction.md        |   7 +-
 release/viunmark-paper/docs/results.md             |   4 +
 .../notebooks/ViUnMark_Reproduction.ipynb          |   7 +
 .../viunmark-paper/scripts/reproduce_uit_vsfc.py   |  10 +-
 .../viunmark-paper/scripts/score_predictions.py    |  21 ++
 release/viunmark-paper/scripts/verify_assets.py    |  28 +-
 release/viunmark-paper/src/viunmark/cli.py         |  22 +-
 release/viunmark-paper/tests/test_cli.py           |  52 ++++
 .../viunmark-paper/tests/test_public_surface.py    |  13 +-
 11 files changed, 495 insertions(+), 46 deletions(-)
```

Untracked additions before this audit:

- `release/viunmark-paper/artifacts/results/uit_vsfc/missing_result_artifacts.json`
- `release/viunmark-paper/docs/publication-readiness.md`
- `release/viunmark-paper/scripts/render_results_table.py`
- `release/viunmark-paper/src/viunmark/assets.py`
- `release/viunmark-paper/tests/test_asset_contract.py`

## Final Statement

- `REPRODUCTION_HARDENING_COMPLETE=YES`
- `PUBLICATION_EXPORT_CREATED=YES`
- `RESEARCH_REPOSITORY_DESTRUCTIVELY_CLEANED=NO`
- `NEW_TRAINING=NO`
- `MODEL_SELECTION=NO`
- `VALIDATION_READ=NO`
- `TEST_READ=NO`
- `TEST_TUNING=NO`
- `TEST_RECALIBRATION=NO`
- `POST_TEST_RETUNING=NO`
