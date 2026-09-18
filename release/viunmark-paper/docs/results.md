# Results

Macro-F1 is the primary metric and Accuracy is secondary. Result tables should
separate clean `FULL`, corrupted-condition results, corrupted-condition average,
and All-6 average.

The research repository records SHA-256 identities for frozen UIT-VSFC evidence
but does not contain committed numeric paper-result records. This export
therefore does not fabricate result tables. Machine-readable result files should
be placed under `artifacts/results/uit_vsfc/` when released.

Use `scripts/render_results_table.py` to render human-readable tables from an
author-approved machine-readable result artifact. The renderer does not read
labels or recalculate metrics.
