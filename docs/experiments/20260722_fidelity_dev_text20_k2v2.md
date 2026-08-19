# Fidelity Development Set: Textual StructEval-20 K2/V2

Launched: July 22, 2026, 09:03 JST  
Model: `Qwen/Qwen2.5-7B-Instruct`  
Purpose: locate a degraded uniform-quantization baseline for fidelity allocation

## Paired Scope

This run uses the same frozen, outcome-independent manifest as FP16, K8/V8,
K8/V4, and K4/V4:

`toy_kv_experiments/data/structeval_full/manifests/text20_from_allformats100_seed42.json`

- 20 unique non-renderable tasks;
- 10 T-generation and 10 T-conversion tasks;
- 15 input/output task types;
- CSV, JSON, TOML, XML, and YAML outputs.

## Protocol

- real packed affine KIVI-style reference cache;
- key bits 2, value bits 2;
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

- Run label: `20260722_090349`
- Completed: July 22, 2026, 14:07 JST
- Result:
  `toy_kv_experiments/results/fidelity_dev_text20/20260722_090349_K2_V2_inference.json`
- Log:
  `toy_kv_experiments/logs/fidelity_dev_text20_k2v2_20260722_090349.nohup.log`

The artifact passed validation and was evaluated with the official textual
StructEval evaluator. The 20 tasks required 7,484 summed task-seconds
(2.08 task-hours).

## Validated Outcome

| Measure | FP16 | K2/V2 |
|---|---:|---:|
| Mean official textual score | 0.7530 | 0.7175 |
| Mean persistent-cache storage | 1.000 | 0.288 |
| Paired task outcomes | - | 13 unchanged, 3 better, 4 worse |
| New parser failures | 0 | 1 |

The four regressions were Text-to-YAML tasks `001830` and `001818`,
Text-to-CSV task `000219`, and JSON-to-XML task `051714`. Task `001818`
changed from a valid FP16 output with score 1.00 to an unparsable K2/V2
output with score 0.00. The three improvements do not imply that quantization
improves the model: they occurred on tasks already unstable under FP16 and
must be treated as paired variability.

This is the first tested setting with enough degradation to study fidelity
allocation. It is a development-set result, not an official full StructEval
leaderboard score.
