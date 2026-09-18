# Reproduction

The official entry point is `scripts/reproduce_uit_vsfc.py`.

`--mode frozen` validates the public config and external asset manifest. Full
prediction reproduction requires released frozen assets whose SHA-256 identities
are recorded in `artifacts/provenance/reproduction_manifest.json`.

Readout retraining is not exposed by the high-level runner in this release
candidate. The protocol is recovered at policy level, but historical bit-exact
execution is not fully recovered, so an executable retraining path needs
author-reviewed public implementation choices before it is advertised.

Prediction and scoring are separate. Prediction generation must not read labels;
scoring consumes a sealed prediction artifact and a user-supplied label file.
