# Shared-prefix next-token sensitivity

FP16 and KIVI receive the same prompt and the same previously generated
FP16 tokens. A disagreement therefore occurs before free-running outputs
can diverge, although it does not identify a causal cache entry or layer.

| Measure | Result |
|---|---:|
| Selected parse-regression tasks | 23 |
| Teacher-forced positions | 7762 |
| Tasks with a top-1 disagreement | 23 |
| FP16 top-1 matches replay target | 7758 / 7762 (99.948%) |
| KIVI top-1 matches replay target | 7720 / 7762 (99.459%) |
| All top-1 disagreements | 40 (0.515%) |
| Structural-marker disagreements | 12 / 1854 (0.647%) |
| Other-token disagreements | 28 / 5908 (0.474%) |

## Interpretation boundary

The trace isolates next-token changes under an identical prefix, but does not attribute a change to a specific cache position, layer, key, or value.
The evaluator artifact stores decoded output text rather than original token IDs.
The trace retokenizes that text; any FP16 replay mismatch is reported above
rather than silently treated as an effect of quantization.
The selected tasks are KIVI-4-induced parse regressions, so these rates
must not be generalized to all StructEval-T tasks.
