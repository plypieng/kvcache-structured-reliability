# KV-Cache Quantization and Structured-Output Reliability

This repository contains the reproducibility harness and frozen evidence for
the study **KV-Cache Quantization and Structured-Output Reliability: A Paired
Evaluation on StructEval-T**.

The current study evaluates the official KIVI implementation with
Mistral-7B-Instruct-v0.2 under three Key and Value cache settings:

- FP16;
- KIVI-4 (4-bit keys and values);
- KIVI-2 (2-bit keys and values).

The complete textual StructEval track contains 950 paired tasks. The repository
also includes the LongBench reproduction harness used to check the official
KIVI implementation before applying it to structured outputs.

## Current boundary

The completed evidence is an evaluation study. Structure-aware mixed-fidelity
protection is a research hypothesis and is not presented as an evaluated method.

## Reproduce the analysis

```bash
python3 -m toy_kv_experiments.verify_experiment_freeze
python3 -m pytest -q
python3 -m toy_kv_experiments.analyze_official_structeval_pairs \
  --fp16 artifacts/structeval_t_20260731/fp16/evaluation.json \
  --kivi4 artifacts/structeval_t_20260731/kivi4/evaluation.json \
  --kivi2 artifacts/structeval_t_20260731/kivi2/evaluation.json \
  --output-dir artifacts/structeval_t_20260731/analysis
```

See [the experiment freeze](docs/reproducibility/EXPERIMENT_FREEZE_20260731.md)
for exact model revisions, KIVI source revision, protocol boundaries, hashes,
and server commands.

## Repository scope

Downloaded datasets, model weights, raw checkpoints, notebooks, generated
slides, and server logs are deliberately excluded. Frozen official evaluator
outputs and their SHA-256 checksums are included because they are the evidence
used by the paired analysis.
