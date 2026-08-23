# Shared-prefix next-token sensitivity

FP16 and KIVI receive the same prompt and the same previously generated
FP16 tokens. A disagreement therefore occurs before free-running outputs
can diverge, although it does not identify a causal cache entry or layer.

| Measure | Result |
|---|---:|
| Selected parse-regression tasks | 23 |
| Teacher-forced positions | 7706 |
| Tasks with a top-1 disagreement | 10 |
| FP16 top-1 matches replay target | 7704 / 7706 (99.974%) |
| KIVI top-1 matches replay target | 7685 / 7706 (99.727%) |
| All top-1 disagreements | 19 (0.247%) |
| Structural-marker disagreements | 3 / 1792 (0.167%) |
| Other-token disagreements | 16 / 5914 (0.271%) |

## Interpretation boundary

The trace isolates next-token changes under an identical prefix, but does not attribute a change to a specific cache position, layer, key, or value.
The evaluator artifact stores decoded output text rather than original token IDs.
The trace retokenizes that text; any FP16 replay mismatch is reported above
rather than silently treated as an effect of quantization.
The selected tasks are KIVI-4-induced parse regressions, so these rates
must not be generalized to all StructEval-T tasks.
