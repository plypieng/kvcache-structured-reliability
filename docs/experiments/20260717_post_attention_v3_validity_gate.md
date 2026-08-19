# Post-Attention KIVI-Style Validity Gate

Date: 2026-07-17
Status: Completed; validity gate passed

## Purpose

Validate the corrected cache update order before any 50- or 100-task rerun.
This gate compares FP16 against uniform K2/V2 on exactly the same ten frozen
StructEval-T JSON tasks.

This is a deterministic causal-study protocol using the official prompt wrapper
and extraction/scoring logic. It is not the official full StructEval inference
protocol because generation is greedy, capped at 1024 tokens, and includes
explicit END_CODE and repeated-tail stopping.

## Fixed protocol

- Model: `Qwen/Qwen2.5-7B-Instruct`
- Manifest: `json_10_stratified_seed42.json`
- Families: two each from Text, CSV, XML, YAML, and TOML to JSON
- Cache revision: `kivi-post-attention-finalize-v3`
- Residual length: 128 tokens
- Quantization group size: 32
- Quantizer: packed affine min/max, K per-channel across token groups, V
  per-token across channel groups
- Generation: greedy, seed 123, maximum 1024 new tokens
- Protection: none
- Evaluator-path leakage: false

## Validation completed before launch

- Local tests: 47 passed
- A6000 tests: 47 passed
- Tiny Qwen logit boundary matches full-precision current-residual attention
- Explicit key-boundary and value-overflow trace matches the post-attention
  reference sequence
- Local pretrained 0.5B integration smoke test completed with v3 storage and
  atomic checkpoint metadata
- Local and remote core source SHA-256 values match

| File | SHA-256 |
|---|---|
| `pretrained_kv_quantization.py` | `bdc4ac7d3def205c4786b268b20b6e17be210bd3b0d8bd12aaffcac763c54679` |
| `real_quantized_cache.py` | `86c04aad8dd86626d041dc900379c236714ee7715440ee0d04bbc745ad5118af` |
| `analyze_paired_structeval.py` | `e1ca4cc666a29b91cf526bbd87058a216a6743ddd1599274e6ffe73337c7cc34` |

## Server state

- Host: `plypieng@100.114.7.72`
- Queue PID: `8252`
- Log: `toy_kv_experiments/logs/post_attention_gate_v3_20260717_0350.nohup.log`
- FP16 run ID: `b76a2d3b6a204faf9b889218a836bf2f`
- K2/V2 run ID: `4312e08fb62e4892af793ad6a30c15fe`
- FP16 output: `toy_kv_experiments/results/post_attention_gate/post_attention_v3_json10_fp16_20260717_035141.json`
- K2/V2 output: `toy_kv_experiments/results/post_attention_gate/post_attention_v3_json10_K2_V2_20260717_035141.json`

Both FP16 and K2/V2 completed 10/10. Every task was atomically stored under the
corresponding `.json.checkpoints/tasks/` directory. The queue's final assertions
confirmed paired task IDs, the v3 cache revision, non-empty run fingerprints,
and no evaluator-path leakage.

## Early paired sanity check

Task `100530` completed in both modes:

- FP16: 241 generated tokens, 13.62 seconds, parse success, all required paths.
- K2/V2: 241 generated tokens, 354.22 seconds, parse success, all required paths.
- The extracted JSON and every leaf value are exactly identical.
- K2/V2 persistent storage: 9,372,160 bytes versus 30,851,072 FP16-equivalent
  bytes, or 3.292x compression and 4.861 effective stored bits per scalar after
  payload, metadata, and residuals.

This confirms the first corrected boundary run is behaviorally plausible. The
large runtime difference is expected from reconstructing packed blocks before
standard attention and is not a fused-kernel performance result.

The sanity check also found that the original v3 FP16 rows left generic cache
storage fields at zero. The reporting layer was corrected for future runs to
record physical persistent bytes for FP16, fake quantization, and packed
storage. The current gate remains valid for behavioral comparison; its FP16
bytes can be reconstructed exactly from sequence length and model dimensions.

The reporting-only patch was deployed after the K2 process had imported the
original gate source. The completed K2 result therefore remains tied to the
recorded `bdc4...` source hash. Its checkpoint directory must not be resumed
with the updated source; the fingerprint guard will correctly reject that
mixed-code resume. New Stage A runs use the updated accounting source and a new
fingerprint.

## Completed result

| Metric | FP16 | K2/V2 | Paired change |
|---|---:|---:|---:|
| JSON parse success | 10/10 | 8/10 | -2 tasks |
| Full required-path success | 4/10 | 2/10 | -2 tasks |
| Mean required-path rate | 0.517 | 0.332 | -0.184 |
| Mean StructEval structural score | 0.613 | 0.426 | -0.187 |
| Exact output-text matches | - | 5/10 | - |

The 95% paired bootstrap interval for the mean structural-score change is
`[-0.500, 0.098]`, and the exact McNemar p-value for parse success is `0.5`.
This ten-task gate therefore shows a substantial observed K2/V2 effect but is
not a statistically conclusive estimate of its population effect.

Across tasks, K2/V2 stored an average of 4.912 effective bits per cache scalar
after packed payload, affine metadata, and the FP16 residual were included. The
mean observed compression ratio relative to an FP16 cache was 3.278x, ranging
from 2.899x to 3.815x depending on sequence length.

The reference implementation took 6,376.9 seconds for K2/V2 versus 149.4
seconds for FP16 across the ten tasks. This approximately 42.7x slowdown comes
from reconstructing packed blocks before ordinary attention and is not evidence
about a fused KIVI kernel's production performance.

## Critical failure inspection

Both compression-induced parse failures were the two YAML-input tasks:

| Task | FP16 envelope | K2/V2 envelope | Inner JSON |
|---|---|---|---|
| `180528` | `<|BEGIN_CODE|> ... <|END_CODE|>` | `<code>|BEGIN_CODE| ... <code>|END_CODE|` | Valid, all paths, exactly equal to FP16 |
| `180548` | `<|BEGIN_CODE|> ... <|END_CODE|>` | `<code> ... </code>` | Valid, all paths, exactly equal to FP16 |

Thus, the official-style extraction failures were not caused by missing braces,
truncation, or incorrect JSON payloads. K2/V2 changed the output-control tokens,
so the extractor received text beginning with `<code>` and rejected it before
JSON parsing. If the substring from the first `{` through the final `}` is
parsed diagnostically, both payloads are valid, contain every required path,
and are exactly equal to their paired FP16 payloads.

This diagnostic must not replace the official-style score: obeying the required
output protocol is part of structured generation. It does, however, localize
the failure mechanism and provides direct motivation for unequal protection of
protocol-control and structure-critical tokens.

The remaining non-identical outputs show that K2/V2 is not only an envelope
perturbation. One Text task improved from score 0.20 to 0.86, another fell from
0.73 to 0.20, and one XML task retained the same structural score while changing
some values. Five tasks matched FP16 exactly at the raw output-text level.

By input family in this small sample:

- TOML: both tasks exactly preserved and scored 1.0.
- CSV: both remained parseable but had no required-path coverage under either
  FP16 or K2/V2.
- XML: both remained parseable with unchanged structural scores; one changed
  content values.
- Text: both remained parseable, with one structural gain and one loss.
- YAML: both payloads were preserved exactly, but their control envelopes were
  corrupted and therefore failed official-style extraction.

The apparent YAML concentration is a two-task observation, not a supported
input-family conclusion.

## Gate decision

The validity gate passed. K2/V2 quality is poor on two official-style outputs,
but no implementation, provenance, pairing, extraction, scoring, or leakage
defect was found. The corrected JSON-100 Stage A matrix may proceed.

Local copies of the frozen results:

- `toy_kv_experiments/results/post_attention_gate/post_attention_v3_json10_fp16_20260717_035141.json`
- `toy_kv_experiments/results/post_attention_gate/post_attention_v3_json10_K2_V2_20260717_035141.json`
- `toy_kv_experiments/results/post_attention_gate/post_attention_v3_json10_K2_V2_paired_20260717_035141.json`
