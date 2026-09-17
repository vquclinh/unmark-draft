# Reproduction

The official entry point is `scripts/reproduce_uit_vsfc.py`.

`--mode frozen` validates the public config and external asset manifest. Full
prediction reproduction requires released frozen assets whose SHA-256 identities
are recorded in `artifacts/provenance/reproduction_manifest.json`.

`--mode retrain-readouts` is reserved for public readout retraining. The protocol
is recovered at policy level, but historical bit-exact execution is not fully
recovered, so the current script stops before making material training choices.

Prediction and scoring are separate. Prediction generation must not read labels;
scoring consumes a sealed prediction artifact and a user-supplied label file.
