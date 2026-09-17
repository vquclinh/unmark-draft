# ViUnMark

ViUnMark is a Vietnamese diacritic-robust sentiment system. The public method has
three pathways: ViUnMark-Gate, ViUnMark-Scale, and a native PhoBERT pathway. The
final system averages exactly five selected heads per branch, fuses raw logits
with fixed weights, then applies the UIT-VSFC reproduction calibration once.

## Install

```bash
pip install -e .
```

The core package is stdlib-only. Install `torch` only for tensor modules or
readout training experiments.

## Dataset And Assets

Raw UIT-VSFC data and model checkpoints are not included. Set paths in
`configs/uit_vsfc/paper.json`; frozen asset identities are listed in
`artifacts/provenance/reproduction_manifest.json`.

## Quick Reproduction

```bash
python scripts/verify_assets.py --config configs/uit_vsfc/paper.json
python scripts/reproduce_uit_vsfc.py --config configs/uit_vsfc/paper.json --mode frozen
```

Frozen prediction generation requires the external checkpoint/result assets
identified in the manifest. Prediction generation and scoring are separate:
`scripts/predict.py` does not read labels, and `scripts/score_predictions.py`
scores an already sealed prediction artifact.

## Readout Retraining

The recovered policy supports protocol-level Stage-II/readout retraining. It is
not a claim of bit-exact historical retraining; the public runner stops until an
author-reviewed executable policy is supplied.

## Diagnostics

```bash
python scripts/run_diagnostic.py --analysis scale-preflight
python scripts/run_diagnostic.py --analysis gain-factorization
```

Available analyses are `scale-preflight`, `pooling-bridge`,
`pooling-comparison`, `decision-geometry`, `complementarity`,
`gain-factorization`, `adapted-only`, `phobert-readout`, and `viunmark`.

## Expected Results

Macro-F1 is primary and Accuracy is secondary. Numeric result artifacts are not
committed in the research repository; this export records the SHA-256 identities
of frozen evidence and reserves `artifacts/results/uit_vsfc/` for released
machine-readable result tables.

## Structure

`src/viunmark/` contains the public package, `scripts/` contains entry points,
`configs/` contains UIT-VSFC configuration, `artifacts/` contains provenance and
historical machine-readable records, `docs/` contains compact method and
reproduction notes, and `tests/` checks public behavior.

## Citation

Citation metadata is pending the paper's final bibliographic details.

## Acknowledgements

The method uses the `vinai/phobert-base` model identity recorded in the public
config and the UIT-VSFC dataset supplied by the user under its own terms.
