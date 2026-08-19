# Schema-Path UEP-KV: First Fidelity-Allocation Experiment

## Current Evidence

Our StructEval JSON experiments show a useful boundary:

- Full precision and 8-bit KV cache are stable for `Qwen2.5-7B-Instruct` on the JSON subset.
- 4-bit KV cache collapses: outputs often become long loops or malformed/truncated JSON.
- KIVI-style residual cache and blockwise fake quantization did not rescue 4-bit by themselves.
- JSON-syntax-token protection was too sparse. It protected punctuation and control markers, but not the prompt-side field names and required schema paths that StructEval actually scores.

This suggests the next experiment should not only protect `{`, `}`, `:`, and `,`. It should protect tokens that carry the output contract itself, such as `novel`, `author`, `birth_year`, `characters`, and other required path segments.

## Proposed Idea

Working name: **Schema-Path UEP-KV**.

The idea is to treat KV-cache compression as an unequal-error-protection problem:

> Under a fixed memory budget, not every cached token has the same failure cost. Tokens that encode schema paths, required fields, parser boundaries, or output constraints should receive higher KV fidelity than ordinary content tokens.

For StructEval JSON tasks, the benchmark already provides `raw_output_metric`, which lists required JSON paths. We can extract path terms from it and mark matching prompt/cache token positions as constraint-critical.

Example:

```text
Required path: novel.author.birth_year
Protected terms: novel, author, birth_year, birth, year
```

Then compare:

```text
No UEP:
  all older KV positions -> 4-bit or 6-bit

Schema-Path UEP:
  ordinary older KV positions -> 4-bit or 6-bit
  schema/path-critical positions -> 8-bit
  recent residual window -> FP16

Budgeted Schema-Path UEP:
  choose only the highest-priority schema/path positions that fit
  a target average-bit budget, e.g. 6.5 bits per KV scalar
```

The budgeted version is now the cleaner research formulation:

```text
Given a cache sequence x_1,...,x_n and a bit budget B,
choose a protected set S that maximizes structural importance
while keeping the average KV precision below B.
```

In the first implementation, all selected schema/path candidates have the same extra cost, so the budgeted policy reduces to selecting a fixed number of candidate positions. This is deliberately simple and inspectable for B4. Later, the score can be refined using parser state, attention, entropy, or path depth.

The current implementation also supports a simple weighted policy:

```text
parent keys receive lower score
leaf required fields receive higher score
control/syntax markers receive fixed structural score
under a target budget, select the highest-scoring older cache positions
```

This makes the idea closer to coding-theoretic unequal protection: the protected set is not only "tokens matching the schema," but the subset with larger estimated structural failure cost.

## Why This Is Different From Existing Work

- **KIVI** studies asymmetric KV quantization and shows keys should be quantized per-channel while values should be quantized per-token. Our implementation keeps that KIVI-inspired axis choice, but asks a different question: which token positions are dangerous to compress for structured validity?
- **KV-AdaQuant** studies different bit allocation for keys and values, motivated by key sensitivity. Our experiment can combine with that by trying `K8/V4` and `K4/V8`, but our main novelty is path/schema-aware position selection.
- **Don't Waste Bits** uses adaptive token-wise precision based on token importance, inspired by Huffman-style variable allocation. Our idea is narrower and more formal: importance is derived from output constraints and required schema paths, not only learned or generic saliency.
- **CodeComp** protects structurally critical code tokens using program-analysis priors. Our related but distinct direction is parser/schema/path-aware protection for structured outputs such as JSON, tool calls, and later code.
- **Constrained decoding** enforces valid next-token choices during generation. Our method instead protects the model's internal memory of the requested structure. These are complementary, not identical.

## Experiment Prepared in Code

Inference sweep:

```bash
bash toy_kv_experiments/server/run_constraint_uep_structeval_jobs.sh
```

Default sweep:

```text
1. K4/V4 no protection
2. K4/V4, schema/path tokens at 8-bit
3. K4/V4, JSON syntax + schema/path tokens at 8-bit
4. K6/V6 no protection
5. K6/V6, schema/path tokens at 8-bit
```

The important comparison is not only whether 4-bit becomes perfect. The better test is whether schema/path UEP improves parse success, required-path rate, max-token-hit rate, or final StructEval score at the same base precision.

Budget-only analyzer:

```bash
python -m toy_kv_experiments.analyze_schema_path_uep_budget \
  --structeval-jsonl toy_kv_experiments/data/structeval_full/structeval_test.jsonl \
  --output-type JSON \
  --limit 20 \
  --base-bits 4 \
  --protected-bits 8 \
  --residual-length 64 \
  --protection-mode constraint-paths \
  --official-structeval-prompt
```

This uses only the tokenizer. It does not run model inference.

## Budget Feasibility

Measured on the local StructEval JSON subset with the Qwen tokenizer:

| Setting | Data | Mean protected token fraction | Mean estimated KV bits | Compression vs FP16 |
| --- | --- | ---: | ---: | ---: |
| K4/V4 + path tokens at 8-bit + residual 128 | first 20 JSON tasks | 22.5% | 7.28 | 2.20x |
| K4/V4 + path tokens at 8-bit + residual 64 | first 20 JSON tasks | 22.5% | 6.14 | 2.61x |
| K4/V4 + path tokens at 8-bit + residual 64 + target 6.5 | first 20 JSON tasks | 21.8% selected / 22.5% candidates | 6.14 | 2.61x |
| K4/V4 + path tokens at 8-bit + residual 128 | all 250 JSON tasks | 19.9% | 8.95 | 1.81x |
| K4/V4 + path tokens at 8-bit + residual 64 | all 250 JSON tasks | 19.9% | 6.94 | 2.32x |
| K4/V4 + path tokens at 8-bit + residual 64 + target 6.5 | all 250 JSON tasks | 8.3% selected / 19.9% candidates | 6.50 | 2.47x |
| K4/V4 + path tokens at 8-bit + residual 32 | all 250 JSON tasks | 19.9% | 5.87 | 2.73x |
| K4/V4 + path tokens at 8-bit + residual 32 + target 6.0 | all 250 JSON tasks | 18.1% selected / 19.9% candidates | 5.81 | 2.76x |

Weighted priority comparison on all 250 JSON tasks:

| Setting | Order | Selected tokens | Structural-score coverage | Mean estimated KV bits |
| --- | --- | ---: | ---: | ---: |
| residual 64 + target 6.5 | prefix | 8.3% | 40.0% | 6.50 |
| residual 64 + target 6.5 | score | 8.3% | 46.9% | 6.50 |
| residual 32 + target 6.0 | prefix | 18.1% | 93.8% | 5.81 |
| residual 32 + target 6.0 | score | 18.1% | 97.3% | 5.81 |

Interpretation:

- The path-protection mask is not tiny: it protects about 20-23% of prompt tokens, because StructEval prompts repeatedly mention required field names and output paths.
- The recent FP16 residual window dominates the budget for short prompts. Therefore, residual length must be treated as an explicit memory-budget variable, not just a KIVI implementation detail.
- Residual 128 is useful for comparison with earlier runs. Residual 64 is the better first feasibility point because it still keeps recent tokens in full precision while reaching about 2.3-2.6x estimated compression.
- Target-average-bit control is important for fairness. It prevents the method from quietly protecting too many tokens on short examples and gives the advisor a clear constrained-optimization view.
- Score-order allocation improves structural-score coverage without changing the average-bit budget. This is a cleaner research story than prefix-order protection.
- A successful result should not be framed as "maximum compression." It should be framed as a reliability-preserving allocation policy under a fixed budget.

## First Inference Smoke: Weighted UEP at Residual 64

Completed on the A6000 with `Qwen2.5-7B-Instruct`, official StructEval JSON prompt/extraction, first 5 JSON tasks, KIVI-style fake quantization axes, blockwise cache update, `K4/V4`, residual length 64, block size 128.

| Setting | Parse success | Full required paths | Max-token hits | Mean tokens | Mean protected score |
| --- | ---: | ---: | ---: | ---: | ---: |
| K4/V4, no protection | 0/5 | 0/5 | 3/5 | 684.2 | 0.0 |
| K4/V4, schema/path 8-bit, target 6.5, score order | 0/5 | 0/5 | 3/5 | 663.6 | 1.0 |
| K4/V4, schema/path 8-bit, target 6.5, score order, residual 128 | 0/5 | 0/5 | 3/5 | 657.0 | 0.676 |

Interpretation:

- The weighted policy did select the intended high-score schema/path positions. The protected-score fraction was 1.0 in this small run.
- The generated text was already globally corrupted, for example repeated quote/bracket fragments and invalid special-token-like strings. This is stronger than a local missing-field error.
- Therefore, this specific negative result should not be interpreted as "schema/path UEP cannot help." It shows that `K4/V4 + residual 64` is too harsh for this model/task setup: global cache degradation dominates before selective protection can recover JSON structure.
- Increasing the residual window to 128 did not change the outcome on this small smoke set. Therefore, the next useful boundary is not more prompt-side path protection at `K4/V4`.
- The next experiment should test `K6/V6`, `K8/V4`, and `K4/V8` with the same official StructEval protocol. If a moderate-precision setting is stable, UEP can be tested as a reliability improvement under a realistic budget instead of trying to rescue a fully collapsed 4-bit run.
- A complementary track is to combine cache-side UEP with constrained decoding or post-checking, because constrained decoding directly prevents invalid JSON tokens while UEP protects the internal memory of the schema.

## Expected Outcomes

Strong positive result:

- `K4/V4 + schema/path 8-bit` recovers some parse success or path score over plain `K4/V4`.
- `K6/V6 + schema/path 8-bit` clearly improves over plain `K6/V6`.
- Residual 64 performs close to residual 128, showing that the reliability gain is from path-aware protection rather than simply from a large FP16 residual.
- Target-6.5 UEP improves over an unprotected setting with similar estimated average bits.
- Score-order UEP improves over prefix-order UEP at the same average-bit budget.

Still useful result:

- 4-bit remains too unstable, but 6-bit becomes the realistic boundary.
- Path protection reduces looping/max-token hits even before it fully restores JSON correctness.
- The best reliable setting is closer to 6-8 average bits, which is still a valid B4 result because it identifies a stability boundary for structured generation.

Negative result:

- Schema/path protection does not improve any metric. Then the likely failure source is not prompt-side schema memory alone; we should test attention/entropy-based token importance or real packed KV-cache kernels next.
- The target-budgeted variant protects too few useful positions. Then we should replace the current prefix order with a stronger priority score, for example path depth, parser state, or attention-to-schema terms.

## Why This Is Feasible for B4

- The experiment does not require training.
- It uses the existing Qwen2.5-7B setup on the A6000.
- It reuses StructEval's official prompt/extraction and JSON path metric.
- The budget analyzer can be run locally or on the server before using GPU time.
- It is easy to explain mathematically as a constrained allocation problem:

```text
minimize memory cost
subject to structural failure probability staying low

or:

allocate higher fidelity to positions with larger structural failure cost
```

## Literature Anchors

- KIVI: asymmetric 2-bit KV-cache quantization, key per-channel and value per-token. https://arxiv.org/abs/2402.02750
- KV-AdaQuant: more bits for keys, fewer for values. https://arxiv.org/abs/2502.15075
- Don't Waste Bits: token-wise adaptive precision inspired by Huffman-style bit allocation. https://arxiv.org/html/2604.04722v1
- CodeComp: structural KV-cache compression for agentic coding with static program-analysis priors. https://arxiv.org/abs/2604.10235
- StructEval: benchmark for structured outputs, including JSON path-style structural metrics. https://arxiv.org/abs/2505.20139
- JSONSchemaBench: structured-output and constrained-decoding benchmark using real-world JSON schemas. https://arxiv.org/abs/2501.10868
- Unequal error protection: information can have different reliability requirements, so more important information receives stronger protection. https://users.metu.edu.tr/bnakib/aux/c/some_fundamental_limits_of_unequal_error_protection.pdf

## Current Caution

The codebase now has two cache-compression paths:

1. **Fake cache quantization**: tensors remain floating point, but quantization error is injected.
2. **Real blockwise cache storage**: evicted old KV blocks are stored persistently as integer tensors plus scales, while the recent residual window remains floating point.

The real-cache path is available as:

```bash
CACHE_QUANTIZATION_MODE=real-blockwise \
  bash toy_kv_experiments/server/run_kivi_block_structeval_jobs.sh
```

The precision-boundary sweep is:

```bash
bash toy_kv_experiments/server/run_real_cache_precision_boundary_jobs.sh
```

Current verified status:

- Local Qwen2.5-0.5B smoke test runs with `real-blockwise`.
- A6000 Qwen2.5-7B StructEval smoke test runs with `real-blockwise`.
- Persistent evicted blocks use packed `uint8` storage for 2-bit, 4-bit, and 8-bit settings. Numeric 6-bit currently uses int8-backed storage with 6-bit quantization levels, so it is useful for error behavior but not exact 6-bit packed memory accounting.
- Attention still receives dequantized floating K/V tensors because the Hugging Face attention kernels expect floating tensors. Therefore, this is real persistent KV storage, but not yet a fused low-bit CUDA attention kernel.

For fake-cache runs, the correct claim remains:

> We are testing whether schema/path-aware KV fidelity allocation improves structured-output reliability under simulated KV quantization error.

For real-cache runs, the stronger but still precise claim is:

> We are testing structured-output reliability using persistent low-bit KV-cache storage with dequantization at attention time.

We should not yet claim production-speed memory saving until we implement or integrate fused low-bit attention kernels.
