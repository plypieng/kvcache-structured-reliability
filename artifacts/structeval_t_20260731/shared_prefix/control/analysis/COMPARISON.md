# Shared-prefix failure-set versus control-set comparison

The two sets contain 23 tasks each and are matched by
input/output format and nearest FP16 generated length.

| Measure | KIVI-4-induced parse failures | Matched non-regression controls | Difference | Ratio |
|---|---:|---:|---:|---:|
| All top-1 disagreement rate | 0.515% | 0.247% | +0.269% | 2.09x |
| Structural-marker disagreement rate | 0.647% | 0.167% | +0.480% | 3.87x |
| Other-token disagreement rate | 0.474% | 0.271% | +0.203% | 1.75x |

## Interpretation boundary

The control set is matched by input/output format and FP16 generated length, but it is not a randomized or semantic match. The comparison is descriptive and does not establish that structural positions cause parse failures.
The shared-prefix trace identifies changed next-token decisions under an
identical prefix; it does not identify a specific cache position, layer, Key,
or Value as the cause.
