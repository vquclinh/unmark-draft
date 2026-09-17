# Provenance

The public code is derived from the recovered final-system specification and the
clean method layer in the research repository. Historical aliases are preserved
only in `artifacts/provenance/historical_aliases.json` and exact historical
machine-readable files under `artifacts/historical/`.

The important boundary is:

```text
method/protocol recovery != bit-exact historical execution recovery
```

Unresolved historical details include robust-MLP cross-entropy reduction, some
low-level RNG and batch mechanics, historical state-dict layout details, and
native PhoBERT fields not established by its protocol.

Historical aliases, for provenance only, map names such as UNMARK-A, V2-SCF,
R4-W-MLP, R6-U-MLP, R6-W-MLP, D1-D4, COMP-D1, COMP-D2, SYS1, SYS2-1, SYS2-2,
and OPT-stage labels to the public terminology used by the package.
