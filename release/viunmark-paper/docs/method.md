# Method

ViUnMark combines ViUnMark-Gate, ViUnMark-Scale, and native PhoBERT readouts.
The adapted pathways use a frozen PhoBERT encoder, seven tone rows, five letter
rows, zero-vector handling for non-applicable channels, `Linear(3d,d) +
LayerNorm` fusion, and a gate initialized with zero weights and `logit(0.01)`
bias.

Readouts are `FIRST_TOKEN`, `MASKED_MEAN`, and `CONCAT = [FIRST_TOKEN ;
MASKED_MEAN]` from the same encoder forward. The robust MLP is
`LayerNorm(d) -> Linear(d,256) -> GELU -> Dropout(0.1) -> Linear(256,n)`.

For deployment, each branch is the arithmetic mean of exactly five head logits.
The final raw branch weights are PhoBERT `0.500`, ViUnMark-Gate `0.250`, Scale
Unweighted `0.125`, and Scale Weighted `0.125`. Generic method calibration is
the identity; UIT-VSFC reproduction adds class-index `1` bias `+1.25` once.
