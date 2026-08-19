# Fidelity Development Set: Textual StructEval-20 K4/V4

Launched: July 21, 2026, 14:41 JST  
Model: `Qwen/Qwen2.5-7B-Instruct`  
Purpose: identify a degraded low-bit operating point before testing fidelity allocation

## Frozen Development Scope

The development manifest is a deterministic, outcome-independent subset of the
existing all-format StructEval-100 poster manifest:

`toy_kv_experiments/data/structeval_full/manifests/text20_from_allformats100_seed42.json`

- 20 unique non-renderable tasks;
- 10 T-generation and 10 T-conversion tasks;
- 15 input/output task types;
- CSV, JSON, TOML, XML, and YAML outputs;
- seed 42;
- parent-manifest hash recorded in the subset manifest.

Selecting only from the parent manifest allows paired reuse of completed FP16,
K8/V8, and K8/V4 outputs. No model result was used to choose a task.

## Existing Paired Baselines

| Condition | T-generation | T-conversion | Equal category average |
|---|---:|---:|---:|
| FP16 | 77.20 | 73.40 | 75.30 |
| K8/V8 | 77.20 | 81.40 | 79.30 |
| K8/V4 | 77.20 | 81.40 | 79.30 |
| K4/V4 | 77.20 | 81.40 | 79.30 |

The higher quantized averages are driven by sampled decoding changes and are not
evidence that quantization improves the model. The development question is
whether uniform K4/V4 introduces enough degradation to evaluate protection.

## K4/V4 Protocol

- real packed affine KIVI-style reference cache;
- key bits 4, value bits 4;
- residual length 128;
- group size 32;
- no structure-token protection;
- official StructEval prompt and extraction;
- greedy decoding, seed 123;
- stop at `<|END_CODE|>`;
- no loop detector;
- 2,048 generated-token cap;
- checkpoint resume enabled.

## Completed Run

- Run label: `20260721_144147`
- Wrapper PID at launch: `34046`
- Result:
  `toy_kv_experiments/results/fidelity_dev_text20/20260721_144147_K4_V4_inference.json`
- Log:
  `toy_kv_experiments/logs/fidelity_dev_text20_k4v4_20260721_144147.nohup.log`
- K8/V4 runtime for the same 20 tasks: 1.50 hours.
- Completed: July 21, 2026, 16:07 JST.
- K4/V4 summed task time: 1.41 hours.
- Mean persistent cache: 16.58 MB.
- Mean effective storage: 6.61 bits per scalar.
- Mean compression relative to FP16 cache: 2.44x.
- Output-cap hits: 0.

The runner validates the completed artifact and then executes the official
non-renderable StructEval evaluator automatically.

## Paired Finding

K4/V4 changed 10 of 20 generations relative to FP16, but only three official
task scores changed:

- `170536`, XML to JSON: 0.20 to 1.00;
- `001830`, Text to YAML: 1.00 to 0.81;
- `000504`, Text to JSON: 0.72 to 0.91.

Relative to K8/V4, K4/V4 improved one task by 0.19, worsened one by 0.19, and
left 18 scores unchanged. The equal category average therefore remains 79.30.
K4/V4 materially changes decoding and compresses the persistent cache, but this
subset does not exhibit aggregate degradation. It is not yet a suitable
operating point for demonstrating recovery from structure-aware protection.
