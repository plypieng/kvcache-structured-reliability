# Full StructEval Matrix Protocol

Prepared: July 17, 2026  
Model: `Qwen/Qwen2.5-7B-Instruct`  
Dataset: official StructEval, 2,035 tasks

Current execution state:

- FP16 launch: July 17, 2026 at 10:40 JST
- Paused for the poster pilot: July 17, 2026 at 12:31 JST
- Preserved progress: 266 / 2,035 completed tasks
- Original server PID: `12586` (no longer active)
- Run ID: `4061d47723bf40b391639dad06102324`
- Run fingerprint: `6d43d55c650effcad0ee570bf2e37fd0bb34aa5a0797ccdac12865a00a8f1966`
- Log: `toy_kv_experiments/logs/structeval_full_official_FP16_20260717_104017.nohup.log`
- Checkpoints: `toy_kv_experiments/results/structeval_full_official/qwen2_5_7b_official_v1_FP16_inference.json.checkpoints/`

## Decision

The main matrix will use the complete StructEval dataset rather than the
JSON-only subset. A result is called benchmark-comparable only after all three
stages are complete:

1. inference on every official task with the official prompt and decoding
   settings;
2. official rendering and non-renderable parsing;
3. official structural and GPT-4.1-mini visual evaluation.

Local JSON parsing metrics are diagnostics and are not substitutes for the
official score.

## Official scope

| Leaderboard category | Definition | Tasks |
|---|---|---:|
| StructEval-T generation | Text input to a non-renderable structure | 250 |
| StructEval-T conversion | Structured input to another non-renderable structure | 700 |
| StructEval-V generation | Text input to a renderable structure | 650 |
| StructEval-V conversion | Structured input to a renderable structure | 435 |
| **Total** | 18 output formats and 44 task types | **2,035** |

The reported overall score is the unweighted mean of the four category means,
not the task-weighted mean over 2,035 rows.

Published Qwen2.5-7B-Instruct reference values are 59.21, 62.18, 53.28, and
61.43 for the four categories, with an overall value of 59.03. These are the
baseline validation target, not values that our run may copy or assume.

References:

- <https://tiger-ai-lab.github.io/StructEval/>
- <https://arxiv.org/abs/2505.20139>
- <https://github.com/TIGER-AI-Lab/StructEval>

## Frozen inference protocol

- Dataset ID set: all 2,035 official IDs, in official file order.
- Model: `Qwen/Qwen2.5-7B-Instruct`.
- Prompt: official `<|BEGIN_CODE|>` / `<|END_CODE|>` wrapper.
- Decoding: greedy, temperature 0.0.
- Output limit: no benchmark-specific cap; generation may use the remaining
  positions in the model's 32,768-token context window.
- Early stopping: model EOS only. END-tag and repeated-tail heuristics are
  disabled because they are not part of the published inference setup.
- Resume: one atomic checkpoint per task with run-fingerprint validation.
- Uniform cache conditions: FP16, K8/V8, K8/V4, K4/V8, K4/V4, and K2/V2.
- Quantized cache: residual length 128, group size 32, packed affine storage,
  and post-attention finalization revision `kivi-post-attention-finalize-v3`.
- Protection: none in the uniform characterization matrix.

The paper used vLLM for open-source baselines. Our cache intervention requires a
manual Hugging Face decoding loop. Greedy outputs should be close, but this
runtime difference is why the complete FP16 reproduction is a mandatory gate.

## Evaluation protocol

Run the official renderer and evaluator on a copy of each completed inference
file. The formulas are:

```text
StructEval-T = 0.2 * syntax/render success + 0.8 * required-path score
StructEval-V = 0.2 * render success + 0.1 * keyword score + 0.7 * VQA score
```

Use GPT-4.1-mini for the visual VQA stage, matching the accepted paper. The
rescoring path is pinned to the upstream `litellm` branch at commit `7781339`.
This current official branch contains stricter renderer fixes, so small
StructEval-V differences from historical website values may reflect renderer
version as well as model inference. Rendering executes model-generated HTML,
framework, and Python code. The server therefore runs the rendering stage under
Bubblewrap with a read-only root filesystem, per-run writable scratch, sanitized
environment variables, and isolated PID/IPC namespaces. The judge API key is
not present inside the renderer sandbox.

After evaluation, run:

```bash
python -m toy_kv_experiments.summarize_structeval_official \
  path/to/evaluation.json \
  --out path/to/summary.json
```

The summary is valid only when `official_scope_complete` is `true`.

## Execution gate

1. Run all 2,035 FP16 generations.
2. Render and score FP16 with the official evaluator.
3. Compare its four category values with the published Qwen2.5-7B-Instruct row.
4. Investigate protocol or runtime differences if the discrepancy is material.
5. Only then start compressed full-set conditions.

The full compressed matrix script requires `ALLOW_LONG_MATRIX=1` so it cannot
be launched accidentally before this gate.

Each condition is accepted only after the artifact checker confirms exactly
2,035 unique task IDs in the exact official dataset order, the expected
250/700/650/435 category counts, all required official fields, and the requested
K/V bit widths. A separate matrix
evaluation script scores every accepted condition and produces one four-column
leaderboard table. Completed score artifacts are reused by default to avoid
repeating paid visual-judge calls.

## Evaluator validity gate

The pinned evaluator passed its 22 upstream tests. A deterministic smoke run
then rendered and evaluated one safe example for every official output format
inside the Bubblewrap sandbox:

- render success: 18/18;
- evaluation rows: 18/18;
- browser and framework renderers: Angular, Canvas, HTML, Markdown, Mermaid,
  React, SVG, Vega, and Vue;
- local execution renderers: Matplotlib, LaTeX, TikZ, and Typst;
- non-renderable parsers: CSV, JSON, TOML, XML, and YAML.

Smoke artifact:
`toy_kv_experiments/results/structeval_evaluator_smoke/20260717_isolated/`

The server does not currently have `OPENAI_API_KEY` configured. This is the
remaining prerequisite for the 1,085 GPT-4.1-mini visual-judge calls. It does
not block inference, isolated rendering, or non-renderable StructEval-T scoring.

## Runtime consequence

The corrected K2/V2 JSON pilot averaged about 637.7 seconds per task in the
unfused reference implementation. At that rate, one 2,035-task compressed
condition takes about 360.5 hours, or 15 days. Five compressed conditions would
take roughly 75 days serially on one A6000, before rendering and VQA.

Therefore the full matrix requires either a fused quantized-attention path,
multiple GPUs, or a much later completion date. The full FP16 reproduction can
start now; the current unfused implementation is suitable for correctness
studies but not for completing the six-condition benchmark by August 3.

## Entry points

- FP16 reproduction:
  `toy_kv_experiments/server/run_structeval_full_official_fp16.sh`
- Guarded full matrix:
  `toy_kv_experiments/server/run_structeval_full_official_matrix.sh`
- Official rendering/evaluation:
  `toy_kv_experiments/server/evaluate_structeval_official.sh`
- Full-condition official evaluation matrix:
  `toy_kv_experiments/server/evaluate_structeval_full_official_matrix.sh`
- Inference/score artifact validator:
  `toy_kv_experiments/check_structeval_official_artifact.py`
- Final matrix table builder:
  `toy_kv_experiments/summarize_structeval_matrix.py`
- Evaluator environment setup:
  `toy_kv_experiments/server/setup_structeval_eval_env.sh`
- Isolated renderer:
  `toy_kv_experiments/server/render_structeval_isolated.sh`
- Leaderboard summary:
  `toy_kv_experiments/summarize_structeval_official.py`
