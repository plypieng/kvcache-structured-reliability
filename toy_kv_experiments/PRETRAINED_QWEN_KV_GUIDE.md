# Pretrained Qwen KV-Cache Quantization Guide

This track uses a real pretrained model instead of the TinyGPT toy model.

Model:

- Hugging Face ID: `Qwen/Qwen2.5-0.5B-Instruct`
- Local path: `toy_kv_experiments/models/qwen2_5_0_5b_instruct`
- Main script: `toy_kv_experiments/pretrained_kv_quantization.py`
- Notebook: `notebooks/track3_qwen_pretrained_kv_quantization.ipynb`

## Why This Track Exists

The TinyGPT notebook teaches the mechanism of attention, KV cache, and quantization in a small controlled model. This Qwen track shows the same idea on a real instruction model:

1. Load a pretrained causal language model.
2. Run normal instruction-style inference.
3. Inspect the real Hugging Face `DynamicCache`.
4. Generate manually token-by-token instead of using `model.generate()`.
5. Modify the KV cache during decoding.
6. Compare full KV, fake INT8 KV, fake INT4 KV, and fake INT2 KV.

## Download Or Refresh The Model

```bash
hf download Qwen/Qwen2.5-0.5B-Instruct \
  --local-dir toy_kv_experiments/models/qwen2_5_0_5b_instruct
```

The script loads with `local_files_only=True`, so after the model is downloaded it can run without contacting Hugging Face.

## Simple Inference And KV Quantization Smoke Test

```bash
python toy_kv_experiments/pretrained_kv_quantization.py \
  --max-new-tokens 80 \
  --bits 16 8 4 2
```

Expected interpretation:

- `16`: no KV quantization, used as the full-precision reference.
- `8`: fake INT8 KV cache. Usually close to full precision.
- `4`: fake INT4 KV cache. Often still usable on simple prompts.
- `2`: fake INT2 KV cache. Often damages generation strongly.

## StructEval JSON Smoke Test

```bash
python toy_kv_experiments/pretrained_kv_quantization.py \
  --structeval-jsonl toy_kv_experiments/data/structeval_full/structeval_test.jsonl \
  --output-type JSON \
  --structeval-limit 5 \
  --max-new-tokens 512 \
  --bits 16 8 4 2 \
  --out toy_kv_experiments/results/qwen_structeval_json_smoke.json
```

The script reports:

- `json_parse_success`: whether the generated response can be parsed as JSON.
- `json_required_path_rate`: how many StructEval-required JSON paths are present.
- `required_match_rate`: simple text substring matching, kept for comparison.

For this research direction, `json_parse_success` is more important than normal text similarity because invalid JSON can break a downstream parser or tool call completely.

## What Is Fake Quantization?

The current implementation simulates low-bit KV storage by doing:

```text
float KV -> quantized integer grid -> dequantized float KV
```

The tensors remain floating point in memory, so this does not save real memory yet. It is useful because it injects the numerical error that low-bit KV storage would introduce. This lets us study how cache distortion affects generation before implementing packed low-bit kernels.

## Current Quantization Axes

Qwen cache tensors have shape:

```text
[batch, kv_heads, tokens, head_dim]
```

The educational KIVI-style default is:

- keys: `per-channel`, meaning the scale is shared across token positions for each channel.
- values: `per-token`, meaning the scale is shared across head dimensions for each token.

This is the main place where the script connects pretrained inference to the KIVI-style idea.
