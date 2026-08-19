# Poster Pilot: All-Format StructEval-100

Prepared and launched: July 17, 2026  
Model: `Qwen/Qwen2.5-7B-Instruct`  
Purpose: August 3 poster evidence

## Scope

This is a paired pilot rather than an official StructEval leaderboard run. The
frozen manifest contains:

- 100 unique tasks;
- 25 tasks from each of T-generation, T-conversion, V-generation, and
  V-conversion;
- all 44 official input/output task types;
- all 18 normalized output formats;
- fixed sampling seed 42 and a checked source-dataset hash.

Manifest:
`toy_kv_experiments/data/structeval_full/manifests/allformats_100_category_balanced_seed42.json`

## Frozen Generation Protocol

- greedy decoding, temperature 0;
- official StructEval prompt and extraction wrapper;
- 2,048 generated-token poster cap;
- stop when `<|END_CODE|>` is completed;
- no repeated-tail heuristic;
- residual length 128 and K/V group size 32;
- real packed affine reference cache, revision
  `kivi-post-attention-finalize-v3`;
- no protection in this first characterization matrix.

The 2,048-token cap makes this a poster pilot rather than a website-comparable
score. Every cap hit remains in the result and is reported; no task is dropped.

## Uniform Characterization Matrix

1. FP16
2. K8/V8
3. K8/V4
4. K4/V8
5. K4/V4

The factorial 4/8-bit conditions identify whether key or value fidelity is the
more useful intervention target. A protection experiment will be selected only
after this matrix identifies an operating point with nontrivial degradation.

## Active Run

- Run label: `20260717_123435`
- Queue PID: `19261`
- Log:
  `toy_kv_experiments/logs/poster_pilot_allformats100_20260717_123435.nohup.log`
- Results:
  `toy_kv_experiments/results/poster_pilot_allformats100/`
- First checkpoint: task `180529`, YAML to JSON, FP16, 389 generated tokens,
  official end-marker stop, 19.55 seconds.

### Status snapshot: July 21, 2026, 14:16 JST

- FP16: complete, 100/100.
- K8/V8: complete, 100/100.
- K8/V4: complete, 100/100, finished July 21 at 06:25 JST.
- K4/V8: paused before launch.
- K4/V4: paused before launch.

The parent queue was stopped while K8/V4 ran, then terminated after K8/V4
completed. No additional condition started. The poster workflow now moves from
uniform characterization to the proposed fidelity-allocation experiment.

K8/V4 is currently much slower than FP16 because this experiment uses an
unfused reference implementation that repeatedly reconstructs the quantized
cache. This runtime is useful for implementation diagnosis but is not a claim
about the speed of an optimized low-bit KV-cache kernel.

## Completed Results

The completed FP16 and K8/V8 files passed the local integrity check: exact
manifest order, 100 unique rows, complete required fields, one run fingerprint,
and the intended 25/25/25/25 category balance.

| Condition | Sum of task times | Mean task time | Mean stored cache | Effective bits/scalar | Mean compression |
|---|---:|---:|---:|---:|---:|
| FP16 | 0.705 h | 25.37 s | 61.645 MB | 16.000 | 1.000x |
| K8/V8 | 27.184 h | 978.64 s | 37.078 MB | 9.847 | 1.629x |
| K8/V4 | 61.942 h | 2229.92 s | 32.002 MB | 8.137 | 1.977x |

The effective storage includes affine quantization metadata and the FP16
residual, so it is higher than the nominal payload precision. K8/V8 and K8/V4
each reached the 2,048-token cap on one task; FP16 reached it on none. K8/V4's
61.9-hour runtime reflects the unfused reference implementation and must not be
presented as the expected latency of optimized KV-cache quantization.

### Official StructEval textual evaluation

The official non-renderable evaluator can score the 50 textual tasks without a
VQA model. These are valid scores for this frozen pilot subset, but they are not
official leaderboard scores for the complete 2,035-task benchmark.

| Condition | T-generation | T-conversion | Equal category average |
|---|---:|---:|---:|
| FP16 | 69.44 | 79.76 | 74.60 |
| K8/V8 | 69.44 | 82.96 | 76.20 |
| K8/V4 | 71.36 | 82.96 | 77.16 |

Paired analysis shows that K8/V8 changed six of the 50 generated textual
outputs, but only one task changed evaluator score. On task `170536`, XML to
JSON, FP16 preserved valid JSON syntax but used extra wrapper objects, producing
a key-validation score of 0 and final score 0.2. K8/V8 reproduced the requested
array/object structure, giving key-validation 1 and final score 1.0. The other
49 task scores were unchanged. Therefore, the +1.60 average is a single-sample
decoding effect and does not support a claim that K8/V8 is generally better.

K8/V4 changed 19 of the 50 textual generations relative to FP16, but only three
task scores changed. Task `170536` improved from 0.2 to 1.0, task `001014`
(Text to TOML) improved from 0.0 to 1.0, and task `001839` (Text to YAML)
worsened from 1.0 to 0.48. Relative to K8/V8, K8/V4 improved one score, worsened
one score, and left 48 unchanged. The higher aggregate is therefore caused by a
small number of decoding changes and is not evidence that lower value precision
is intrinsically better.

Local evaluation artifacts:

- `toy_kv_experiments/results/poster_pilot_allformats100/20260717_123435_FP16_poster_t_eval/`
- `toy_kv_experiments/results/poster_pilot_allformats100/20260717_123435_K8_V8_poster_t_eval/`
- `toy_kv_experiments/results/poster_pilot_allformats100/20260717_123435_K8_V4_inference_poster_t_eval/`

## Preserved Full Benchmark

The complete official FP16 reproduction was paused to prioritize the poster.
Its 266 completed checkpoints remain under:

`toy_kv_experiments/results/structeval_full_official/qwen2_5_7b_official_v1_FP16_inference.json.checkpoints/`

The full run can resume from task 267 after the poster queue. Its rows must not
be pooled with this capped pilot.

## Evaluation Boundary

Non-renderable and rendering stages can use the checked StructEval evaluator.
The 50 visual tasks per condition additionally require GPT-4.1-mini VQA. Across
five conditions this is 250 visual-judge task calls. The server still requires
`OPENAI_API_KEY` before that stage can be completed.
