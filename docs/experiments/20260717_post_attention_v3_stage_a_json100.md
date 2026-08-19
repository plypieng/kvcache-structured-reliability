# Corrected Stage A JSON-100 Matrix

Date started: 2026-07-17 10:19 JST  
Status: Stopped intentionally after 19/100 FP16 tasks

The queue was stopped after preserving its atomic checkpoints because the
experiment scope changed from a JSON-only causal study to the complete official
StructEval benchmark. No compressed condition had started. These partial rows
remain valid pilot data, but they must not be mixed with the full-benchmark
matrix or reported as an official StructEval score.

## Purpose

Characterize the structural reliability and measured persistent-cache storage
of the corrected `kivi-post-attention-finalize-v3` cache before selecting one
operating point for the poster's fidelity-allocation experiment.

This is a deterministic causal-study protocol with the official StructEval
prompt wrapper and JSON extraction/scoring logic. It is not an official full
StructEval benchmark run because decoding is greedy, generation is capped at
1024 new tokens, and explicit END_CODE and repeated-tail stopping are enabled.

## Frozen protocol

- Model: `Qwen/Qwen2.5-7B-Instruct`
- Manifest: `json_100_stratified_seed42.json`
- Tasks: 20 each from Text, CSV, XML, YAML, and TOML to JSON
- Cache revision: `kivi-post-attention-finalize-v3`
- Residual length: 128 tokens
- Quantization group size: 32
- Quantizer: packed affine min/max
- Keys: per-channel quantization across token groups
- Values: per-token quantization across channel groups
- Generation: greedy, seed 123, maximum 1024 new tokens
- Protection: none
- Protection signal source: prompt-visible, unused in uniform conditions
- Evaluator-path leakage: disabled
- Resume: atomic per-task checkpoints with fingerprint validation

## Matrix

| Condition | Purpose |
|---|---|
| FP16 | Paired reference and model-capability baseline |
| K8/V8 | High-fidelity packed-cache control |
| K8/V4 | Lower value fidelity with high-fidelity keys |
| K4/V8 | Lower key fidelity with high-fidelity values |
| K4/V4 | Moderate uniform compression |
| K2/V2 | Aggressive failure-probe setting |

K4/V8 was added before launch so K8/V8, K8/V4, K4/V8, and K4/V4 form a
complete 2x2 comparison of key and value precision. This permits estimation of
key, value, and interaction effects rather than relying on a one-sided
ablation.

## Server record

- Host: `plypieng@100.114.7.72`
- Queue PID: `11599`
- Run label: `20260717_101900`
- Log: `toy_kv_experiments/logs/post_attention_stage_a_v3_20260717_101900.nohup.log`
- Result directory: `toy_kv_experiments/results/post_attention_stage_a/`
- Runner: `toy_kv_experiments/server/run_post_attention_stage_a_json100.sh`
- Test result immediately before launch: `47 passed`

Local and server hashes matched immediately before launch for the Stage A
runner, A6000 runner, quantization source, cache source, and frozen manifest.

The FP16 condition completed 19 tasks before the scope change. Its run ID is
`31881a4506a64bbba257a0ee8475f4bf` and the fingerprint is
`a5286ece40c65af021a0cd247edbc9fabf1bc88f09404f91a6e6d8c0aefc617e`.
The last preserved checkpoint is `0018_000519_K16_V16`.

## Expected duration

FP16 should finish in roughly 20-30 minutes. The corrected K2/V2 gate averaged
about 10.6 minutes per task in the unfused reference implementation. If the
other compressed conditions have similar overhead, the complete six-condition
queue may require approximately three to five days. Per-task checkpointing
allows safe recovery from interruption.

This runtime is an implementation cost of reconstructing packed blocks before
ordinary attention. It must not be reported as the latency of a fused KIVI
kernel.

## Analysis after completion

For every compressed condition, pair rows against FP16 by task ID and report:

- parse success and compression-induced parse failures;
- required-path coverage and full-path success;
- StructEval structural score;
- FP16-reference leaf path/value diagnostics;
- paired bootstrap confidence intervals and exact McNemar tests;
- transitions by Text, CSV, XML, YAML, and TOML input family;
- output-envelope failures separately from malformed or truncated JSON;
- exact output matches, structural gains, and structural regressions;
- measured persistent bytes, metadata, residual bytes, compression ratio, and
  effective stored bits per scalar;
- runtime only as reference-implementation overhead.

The operating point for Stage B should have enough degradation to permit a
recovery test without complete collapse. Stage B must compare allocation
methods at equal measured persistent-cache bytes.
