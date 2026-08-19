# Official KIVI on StructEval-T

## Purpose

This experiment tests whether the quality-preservation behavior reproduced on
LongBench also holds for machine-readable outputs. It uses the frozen official
KIVI model implementation rather than the project's generic Qwen KIVI-style
cache.

## Frozen Scope

- Dataset: complete StructEval-T track
- Tasks: 950
- T-generation: 250
- T-conversion: 700
- Rendered or visual tasks: 0
- Output formats: CSV, JSON, TOML, XML, and YAML
- Manifest:
  `toy_kv_experiments/data/structeval_full/manifests/structeval_t_complete_950.json`
- Dataset SHA-256: `d4d838145fbe285d0c446b449470512c0c4d3922422894c77aa6183913751bda`
- Manifest SHA-256: `fdcc2880b57f84b03e08a2c1846e2873ffdd6d6d052c27249bb93e4dd1e8a58a`

## Model and Conditions

- Model: `mistralai/Mistral-7B-Instruct-v0.2`
- Model revision: `41b61a33a2483885c981aa79e0df6b32407ed873`
- KIVI commit: `67aba607a1deaeb18b70ae796ab25d05a08b3345`
- Group size: 32
- FP16 residual length: 128 tokens
- Conditions, in order:
  1. FP16
  2. Official KIVI-4, K4/V4
  3. Official KIVI-2, K2/V2

## Generation and Evaluation

- The official StructEval output-format instruction is appended to each query.
- Mistral's official chat template is applied.
- Decoding is deterministic greedy decoding.
- Generation stops at `<|END_CODE|>`, EOS, or 2,048 generated tokens.
- Every condition uses the same ordered tasks and seed.
- Evaluation uses StructEval's non-renderable parser and required-path scorer.
- The headline T score is the unweighted mean of T-generation and T-conversion.

This is a complete StructEval-T evaluation with the official evaluator. It is
not a full StructEval leaderboard submission because the controlled greedy
decoding protocol differs from StructEval's inference API defaults.

## Audit Gates

- A6000 idle before launch
- Frozen KIVI commit and clean tracked worktree
- Mistral model already cached
- Dataset and manifest hashes verified
- 950 unique tasks in exact source order
- No rendered tasks
- Prompt maximum: 1,004 tokens
- Minimum remaining context: 31,764 tokens
- FP16, KIVI-4, and KIVI-2 generation/conversion smoke tests completed
- Official parser/path evaluation completed for each smoke condition
- Local protocol tests: 59 passed

## Running Queue

- Server: `plypieng@192.168.10.113`
- Run root:
  `/home/plypieng/official_baselines/results/kivi_structeval_t_20260729_170807`
- Current sequence: preflights, full FP16, full KIVI-4, full KIVI-2, comparison
- Preliminary runtime estimate: 40-48 hours plus evaluation overhead

Check once:

```bash
toy_kv_experiments/server/watch_a6000_kivi_structeval_t.sh
```

Follow continuously:

```bash
FOLLOW=1 INTERVAL=30 \
  toy_kv_experiments/server/watch_a6000_kivi_structeval_t.sh
```

## Expected Artifacts

Each condition writes:

- `run_metadata.json`
- resumable `inference.jsonl`
- completed `inference.json`
- `eval/evaluation.json`
- `eval/summary.json`

After all three conditions finish, the queue writes:

- `comparison.json`
- `comparison.md`
- `COMPLETE`
