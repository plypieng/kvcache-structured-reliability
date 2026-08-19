# IEICE First-Draft Evidence Freeze

Date: July 29, 2026

This note records the evidence used in:

- `manuscripts/ieice_shinetsu_2026/manuscript.tex`
- `manuscripts/ieice_shinetsu_2026/manuscript.pdf`

## Official KIVI Baseline Validation

Source:

`toy_kv_experiments/results/official_kivi_longbench15_20260727/comparison.json`

SHA-256:

`a90fedb407fa4da023707d91f3c244f437ba884eb85dea56a95ec681020f4c49`

Reported scope and results:

- model: Mistral-7B-Instruct-v0.2;
- 15 English LongBench tasks and 3,550 examples per condition;
- paper FP16 mean: 43.54;
- paper KIVI-4 mean: 43.53;
- reproduced FP16 mean: 44.95;
- reproduced KIVI-4 mean: 44.88;
- paper KIVI-4 minus FP16: -0.01;
- reproduced KIVI-4 minus FP16: -0.08.

Allowed claim:

The reproduced run shows the same average quality-preservation trend as the
KIVI paper. The absolute scores are not an exact numerical reproduction.

## Structured-Output Pilot

Frozen evidence:

`docs/evidence/20260723/FREEZE.md`

SHA-256:

`c9746ecdfca1bba09111630a3ebbf0698bdd4e8a075285c802b2f8466249f203`

K2/V2 inference artifact:

`toy_kv_experiments/results/fidelity_dev_text20/20260722_090349_K2_V2_inference.json`

SHA-256:

`4457a3ea122e24f8262294fbf698d2918bac975361d441610d3794d2516a6aaa`

Reported scope and results:

- model: Qwen2.5-7B-Instruct;
- frozen 20-task textual StructEval development subset;
- paired FP16, K8/V8, K8/V4, K4/V4, and K2/V2 conditions;
- mean StructEval scores: FP16 0.753, K8/V8 0.793, K8/V4 0.793,
  K4/V4 0.793, and K2/V2 0.7175;
- paired regressions relative to FP16: 0, 0, 1, and 4 for K8/V8, K8/V4,
  K4/V4, and K2/V2, respectively;
- K2/V2: 13 unchanged, 3 better, 4 worse, and 1 new parse failure;
- mean persistent-cache size for K2/V2: 0.288 of FP16;
- task `001818`: FP16 score 1.00 to K2/V2 score 0.00 with a new parse
  failure;
- after code-marker extraction, the K2/V2 output for task `001818` contains
  an extra bare `yaml` line before the document root, which the official YAML
  parser rejects.

Allowed claim:

Average benchmark preservation does not guarantee that every structured output
remains valid. This pilot identifies a development case; it does not establish
a general failure rate or validate structure-aware protection.

## First-Draft Artifacts

- `manuscript.tex` SHA-256:
  `86aa49c5244368464571b7356d0d2c646fa5a19d8e30e96e2d45312bffb49202`
- `manuscript.pdf` SHA-256:
  `627691080dabe768307f9cb6582c515b1edfe2dfec5893feddf1f01585a34402`
- PDF format: one A4 page.
