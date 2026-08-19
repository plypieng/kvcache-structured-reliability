# Shared-prefix KIVI-4 mechanism trace

This artifact replays the stored FP16 output for each of the 23 tasks where
FP16 parsed and KIVI-4 did not. FP16 and KIVI-4 receive the same prompt and the
same previous output tokens at every position. This prevents an earlier token
change from creating a different prefix before the next-token decisions are
compared.

## Frozen configuration

| Item | Value |
|---|---|
| Model | `mistralai/Mistral-7B-Instruct-v0.2` |
| Model revision | `41b61a33a2483885c981aa79e0df6b32407ed873` |
| KIVI revision | `67aba607a1deaeb18b70ae796ab25d05a08b3345` |
| Conditions | FP16 and KIVI-4 (K4/V4) |
| KIVI group size | 32 |
| FP16 residual length | 128 tokens |
| Selection | All 23 FP16-only parse transitions under KIVI-4 |
| Seed | 42 |
| Server run | 2026-08-19 09:35:49--09:57:45 JST |
| Hardware | NVIDIA RTX A6000 |

## Result boundary

The trace localizes next-token sensitivity under an identical prefix. It does
not identify which cache position, layer, Key, or Value caused a disagreement.
The 23 tasks were selected because they regressed, so the observed disagreement
rates cannot be generalized to all StructEval-T tasks or compared causally
without a matched non-regression control set.

The evaluator artifact stores decoded text, not the original generated token
IDs. The trace therefore retokenizes that text. Four of 7,762 replay positions
do not reproduce the stored target as FP16 top-1; these are reported rather
than attributed to quantization.

See `analysis/SUMMARY.md`, `analysis/task_summary.csv`, and
`analysis/top1_disagreements.csv` for the validated outputs. Verify the raw
traces with:

```bash
shasum -a 256 -c artifacts/structeval_t_20260731/shared_prefix/SHA256SUMS
```
