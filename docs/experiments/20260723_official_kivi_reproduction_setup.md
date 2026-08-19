# Official KIVI Reproduction Setup

Prepared: July 23, 2026  
Corrected after generation preflight: July 27, 2026  
Server: `plypieng@100.114.7.72`  
GPU: NVIDIA RTX A6000, 49,140 MiB

## Purpose

This environment is isolated from the StructEval experiments. It reproduces
the official KIVI implementation on Mistral-7B-Instruct-v0.2 before we compare
our own cache implementation with the paper. Existing `KIVI-style` StructEval
results remain separate and must not be relabeled as official KIVI.

## Frozen Source and Runtime

- Repository: `https://github.com/jy-yuan/KIVI`
- Server checkout: `/home/plypieng/official_baselines/KIVI-paper`
- Commit: `67aba607a1deaeb18b70ae796ab25d05a08b3345`
- Commit date: June 16, 2024
- Conda environment: `kivi-paper`
- Python: 3.10
- PyTorch: 2.1.2+cu121
- Transformers: 4.36.2
- FlashAttention: 2.5.6
- Datasets: 2.16.1
- CUDA compiler and headers: 12.1
- GPU compilation target: SM 8.6

This is the last main-branch commit before the ICML 2024 publication period.
Its `requirements.txt` and `pyproject.toml` agree on PyTorch 2.1.2 and
Transformers 4.36.2.

## Why the Setup Was Corrected

The first setup used the November 2025 repository head and its newer
`pyproject.toml`: PyTorch 2.4.1 and Transformers 4.43.1. Imports and a packed
GEMV smoke test passed, but real Mistral generation failed twice:

1. Transformers inserted an empty `DynamicCache`, while KIVI expected its
   custom tuple cache.
2. After suppressing that conversion, KIVI called the older rotary-embedding
   interface `seq_len=...`, which Transformers 4.43 no longer accepts.

The Mistral KIVI file was introduced in April 2024 and still targets
Transformers 4.36.2. The correct response is therefore a paper-era source and
runtime, not patches to the KIVI model mathematics. The failed 2025-head
environment remains at `/home/plypieng/official_baselines/KIVI` and
`kivi-official` as audit evidence.

## Installation

The preparation script creates the isolated environment, checks out the exact
source revision, installs paper-era dependencies, installs the matching
FlashAttention wheel, and compiles the official `kivi_gemv` CUDA extension:

```bash
cd ~/kvcache
bash toy_kv_experiments/server/prepare_official_kivi_repro_env.sh
```

PyTorch 2.1 bundles a pybind11 conversion expression that CUDA 12.1 cannot
parse while compiling an extension. The preparation script verifies that exact
header expression and replaces it with its equivalent implicit conversion
before compiling `kivi_gemv`. This build-only dependency patch does not alter
KIVI source, weights, quantization, or runtime arithmetic.

The validation gate must report:

```text
torch 2.1.2+cu121
transformers 4.36.2
flash_attn 2.5.6
evaluator dependency gate: OK
```

## Reproduction Target

- Model: `mistralai/Mistral-7B-Instruct-v0.2`
- Model revision:
  `41b61a33a2483885c981aa79e0df6b32407ed873`
- Benchmark: the 15 non-E LongBench tasks in KIVI `docs/long_bench.md`
- LongBench revision:
  `f72191f71cd6fcd0da8a54f0915078efda579449`
- Baseline: FP16
- KIVI condition: K4/V4, group size 32, FP residual length 128
- Decoding: greedy, official task-specific generation limits
- Primary quality comparison: per-task score and 15-task mean

The KIVI table reports an FP16 mean of 43.54 and KIVI-4 mean of 43.53 for this
model. Reproduced values need not be bit-identical, but deviations must be
explained from recorded source, model, dataset, prompt, and library versions.

## Source Audit Boundary

The published table contains 15 tasks, while the paper-era launcher lists only
eight non-E tasks. Its Mistral chat-template condition also checks a substring
that does not match the actual model name.

The harness therefore:

- runs the exact 15 tasks shown in the published table;
- preserves official prompt formatting by default with
  `PROMPT_MODE=frozen-source`;
- preserves middle truncation, generation lengths, greedy decoding, the
  SAMSum stop rule, and official LongBench metrics;
- checkpoints every generated example and records complete provenance.

`PROMPT_MODE=published-intent` corrects the Mistral substring and applies the
tokenizer chat template. It is a protocol sensitivity analysis, not the
default paper reproduction.

Because the 15-task set is reconstructed from the table, this is a
source-audited reproduction rather than byte-for-byte execution of the
unmodified launcher.

## Run Sequence

1. Run FP16 and KIVI-4 on one Qasper example.
2. Compare the two artifacts and verify that both evaluators finish.
3. Run all 15 tasks in FP16.
4. Run the same examples with KIVI-4.
5. Compare both runs with the paper table.
6. Only then port the official KIVI path to the frozen StructEval protocol.

```bash
cd ~/kvcache

nohup bash toy_kv_experiments/server/queue_official_kivi_reproduction.sh \
  > ~/official_baselines/results/kivi_longbench_queue.nohup.log 2>&1 &

WATCH=1 bash toy_kv_experiments/server/watch_official_kivi_reproduction.sh
```

The queue refuses to start when another reproduction holds its lock or when
`nvidia-smi` reports an existing compute process.

## Claim Boundary

- **Official KIVI reproduction:** paper-era checkout, compiled CUDA extension,
  supported Mistral model, and LongBench protocol described above.
- **KIVI-style reference:** our packed affine cache with KIVI grouping
  directions and FP residual, dequantized into ordinary Hugging Face
  attention.
- Existing StructEval evidence uses the second path. It is suitable for paired
  quality and persistent-storage analysis, but not for official KIVI latency,
  throughput, or peak-memory claims.
