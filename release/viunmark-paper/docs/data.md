# Data

UIT-VSFC raw data is not included. Users provide a dataset directory through
`configs/uit_vsfc/paper.json`.

Expected tabular fields for scoring are `sample_id` and `label`. Prediction
artifacts use `sample_id`, `condition`, and `prediction`. Labels are mapped as
`negative -> 0`, `neutral -> 1`, and `positive -> 2`.

Splits and sample IDs must be stable across prediction and scoring. The public
code does not tune on test labels and does not silently recalibrate after
scoring.
