# Toy KV Experiments

Small character-level GPT experiments for studying attention, KV cache, and KV-cache quantization.

This project is intentionally small. It is for understanding mechanisms, not for training a strong language model.

## Quick Start

Generate a synthetic structured-text dataset:

```bash
python toy_kv_experiments/make_data.py
```

Train a tiny GPT:

```bash
python toy_kv_experiments/train.py --max-steps 600
```

Generate without KV cache:

```bash
python toy_kv_experiments/generate.py --prompt '{"name":' --mode no-cache
```

Generate with KV cache:

```bash
python toy_kv_experiments/generate.py --prompt '{"name":' --mode cache
```

Generate with fake quantized KV cache:

```bash
python toy_kv_experiments/generate.py --prompt '{"name":' --mode cache --quantize --bits 4
```

Inspect KV-cache memory estimates:

```bash
python toy_kv_experiments/inspect_cache.py
```

Download public dataset samples and source metadata:

```bash
python toy_kv_experiments/download_public_data.py --max-rows 200
```

Download related papers:

```bash
python toy_kv_experiments/download_papers.py
```

Use a public text sample for training:

```bash
python toy_kv_experiments/train.py --data toy_kv_experiments/data/public/tinystories_train.txt --max-steps 600
```

## What This Teaches

- `Q,K,V` are computed from hidden states in self-attention.
- During autoregressive inference, previous `K,V` tensors are stored and reused.
- KV-cache quantization happens after `K,V` are computed.
- Fake quantization simulates quantization error but does not save real memory.
- Structure-aware fidelity allocation can be studied by protecting selected token positions.

## StructEval Sampling

StructEval stores tasks in contiguous task-family blocks. Prefix sampling makes
the first 50 JSON tasks all Text-to-JSON and the first 100 tasks Text-to-JSON
plus CSV-to-JSON. The pretrained-model runner now uses deterministic stratified
sampling by default:

```bash
python -m toy_kv_experiments.pretrained_kv_quantization \
  --structeval-jsonl toy_kv_experiments/data/structeval_full/structeval_test.jsonl \
  --output-type JSON \
  --structeval-limit 100 \
  --structeval-sampling stratified \
  --structeval-seed 42
```

For the 250 JSON-output tasks, `LIMIT=50` selects 10 tasks and `LIMIT=100`
selects 20 tasks from each of Text, CSV, XML, YAML, and TOML. Selected rows are
shuffled reproducibly and result files record the input family, sampling mode,
seed, stratum, and selection index.

This is balanced coverage of every **input family in the JSON-output subset**,
not every StructEval output format. The repository's local scoring path currently
implements JSON only. Full-benchmark experiments across all conversion pairs
must use StructEval's output-specific official evaluators.

Use `--structeval-sampling head` only to reproduce legacy prefix-sampled runs.
Do not compare legacy and stratified aggregates unless the selected task IDs
are identical.

The frozen paired manifest is
[`json_100_stratified_seed42.json`](data/structeval_full/manifests/json_100_stratified_seed42.json).
It includes the source-file hash as well as task IDs, so a run fails rather
than silently changing its benchmark if the dataset changes.

## KIVI-Style Reference Cache

`real-blockwise` is now an asymmetric KIVI-style storage and fidelity reference:

- keys use affine min/max quantization across token groups;
- values use affine min/max quantization across channel groups per token;
- only 2, 4, 8, and 16 bits are valid persistent storage widths;
- prompt attention receives the original full-precision K/V;
- old K/V is stored packed with scale and minimum metadata after prefill;
- completed key groups and value overflow are sealed before decoding attention,
  matching the update order in KIVI Algorithm 1;
- normal Hugging Face attention still receives dequantized tensors.

It is therefore suitable for controlled numerical-fidelity and persistent-byte
experiments, but not for latency or peak-GPU-memory claims. The cache does not
use KIVI's fused CUDA attention kernels.

Run the frozen 100-task KIVI-style reference on the A6000 with:

```bash
BITS=4 RESIDUAL_LENGTH=128 KV_GROUP_SIZE=32 \
  bash toy_kv_experiments/server/run_kivi_reference_json100.sh
```

Use 2, 4, or 8 for `BITS`. The result records the manifest, source hash,
full-precision prefill policy, KIVI group size, and payload/metadata/residual
storage breakdown.

For structure protection, `prompt-visible` is the default signal source: it
extracts identifiers from source code or schema text included in the prompt.
`oracle-required-paths` is retained only for explicitly labeled upper-bound
analysis because it reads StructEval evaluator annotations.

Use the paired analyzer only after both files were run on the same frozen
manifest:

```bash
python -m toy_kv_experiments.analyze_paired_structeval \
  --fp16 path/to/fp16.json \
  --candidate path/to/k4v4.json \
  --out toy_kv_experiments/results/paired_fp16_vs_k4v4.json
```

It rejects mismatched task IDs and reports compression-induced failures only for
tasks that FP16 handled successfully, plus per-family results, paired bootstrap
intervals, and an exact McNemar test for parse success.
