# KV-Cache Compression for Structured Output: Literature and Implementation Audit

Date: 2026-07-10

## Executive verdict

The current code is useful as an educational prototype and as evidence that
structured generation can be sensitive to KV perturbation. It is **not yet an
exact KIVI implementation or a controlled systems comparison**. Current scores
should therefore be described as results from our own symmetric blockwise cache,
not as KIVI benchmark results.

The thesis idea also needs a narrower novelty statement. "Protect
structure-critical tokens" alone is no longer sufficient because recent work
already studies structure-aware code-cache retention, role-aware mixed precision,
token-wise bit allocation, and rate-distortion allocation. A more defensible
question is:

> Under a fixed measured KV-storage budget and identical decoding constraints,
> can schema-conditioned fidelity allocation preserve the model's semantic
> choices among valid structured outputs better than uniform, recency, random,
> and attention-based allocation?

This formulation separates two mechanisms:

- Constrained decoding controls which next tokens are syntactically legal.
- KV-cache fidelity controls the model's probability ranking among legal choices.

The second mechanism can affect field selection, enum choice, nesting, values,
tool choice, arguments, and required-path coverage even when syntax is guaranteed.

## Corrections made in this audit

The experiment runner previously took a prefix of StructEval. Because the data is
stored in contiguous family blocks, the first 50 JSON tasks were all Text-to-JSON
and the first 100 contained only two families.

The runner now defaults to deterministic stratified sampling:

- `LIMIT=50`: 10 each from Text, CSV, XML, YAML, and TOML to JSON.
- `LIMIT=100`: 20 each from Text, CSV, XML, YAML, and TOML to JSON.
- Rows are shuffled reproducibly with seed 42.
- The result records input type, sampling mode, seed, stratum, and selection index.
- The same task IDs must be reused for every precision configuration.

This balances all input families in the **JSON-output subset**. It is not full
coverage of StructEval's 44 conversion pairs. The local scorer currently supports
JSON only; a full-benchmark run must call the official output-specific evaluators.

Two additional protocol errors were corrected:

- JSON-path wildcards now match StructEval semantics: `*` expands lists, not
  dictionary values.
- Scripts named as real/KIVI-style residual experiments now default to
  `real-blockwise`, not fake repeated or fake blockwise quantization.

Historical prefix-sampled and fake-quantization files remain useful as development
history, but must not be pooled with new stratified real-cache runs.

### Implementation status after this audit

The repository now has a KIVI-style persistent-storage reference:

- affine min/max codes with explicit scale and minimum metadata;
- four valid INT2 codes rather than the previous three-level symmetric format;
- key token-group and value channel-group semantics;
- full-precision prefill attention followed by compressed cache storage;
- KIVI-style key and value residual behavior;
- a frozen, hash-checked 100-task JSON manifest; and
- result grouping by treatment configuration and task-set fingerprint.

Constraint-path protection now defaults to identifiers extracted from the prompt's
visible source/schema text. The former `raw_output_metric` path signal is still
available only as explicitly labeled `oracle-required-paths` analysis.

It still dequantizes blocks before normal Hugging Face attention. It is a
controlled storage/fidelity reference, not a fused KIVI kernel and not a
throughput or peak-memory baseline.

## Critical implementation findings

| Priority | Finding | Why it matters | Required correction |
|---|---|---|---|
| P0 | Real-blockwise quantization is active during prompt prefill, while fake modes first run full-precision prefill. | Real-versus-fake comparisons change prompt hidden states and first-token logits, not only stored cache fidelity. | Define one prefill policy and apply it to every compared mode. Add a prefill-logit parity test. |
| P0 | Quantization is symmetric absmax RTN, whereas KIVI uses asymmetric min/max quantization with scale and zero-point. | The error distribution and low-bit behavior differ materially from KIVI. | Implement affine asymmetric grouped quantization and store zero-points. |
| P0 | Nominal 2-bit mode uses only `-1, 0, +1`. | One of four packed codes is unused, so the result is not comparable to four-level INT2. | Use all `2^b` affine codes and test code utilization. |
| P1 | `block_size` controls eviction but is not an independent quantization group size. Same-bit runs become scale groups. | Protection changes group boundaries and can improve neighboring tokens through rescaling, confounding fidelity allocation. | Separate eviction block size, K token-group size, V channel-group size, and precision-run storage. |
| P1 | The nominal residual length can retain between `R` and `R+B-1` floating tokens. | Memory and fidelity differ from the configured value and from KIVI's recent window. | Enforce exactly `R` recent floating-point tokens and test every sequence length around block boundaries. |
| P1 | Full historical K/V tensors are dequantized and concatenated before every attention call. | Persistent packed bytes may decrease, but peak GPU memory and throughput do not represent a fused low-bit system. | Call this a storage-fidelity reference. Do not claim latency or peak-memory gains until attention consumes packed cache directly. |
| P1 | Constraint terms are derived from StructEval evaluator paths. | This can leak evaluation annotations that may not be available at deployment. | Derive protection only from prompt-visible schemas, tool definitions, grammar state, or online model signals. Keep oracle results separately labeled as an upper bound. |
| P1 | Budget accounting uses nominal average bits rather than measured persisted bytes. | Scale, zero-point, padding, residual, and fragmentation overhead can reverse apparent comparisons. | Optimize under measured payload plus metadata bytes. Report K, V, scales, zero-points, residual, padding, and allocator peak separately. |
| P1 | A desired protection mask is recomputed, but already packed blocks cannot be upgraded. | Reported desired allocation can differ from persisted allocation. | Log the actual precision assigned when each block is sealed and score that immutable allocation. |
| P1 | `protected_score_fraction` can exceed one because its numerator and denominator cover different position sets. | The metric is mathematically invalid. | Compute both over the same eligible persisted positions. |
| P1 | Result summarization can aggregate different bit settings from one file and can select incompatible runs as the same configuration. | Aggregate scores may not correspond to one treatment. | Group by full configuration and task-ID manifest, including model revision, task count, cap, stop policy, block/group sizes, seed, and code revision. |
| P2 | Numeric 3/5/6/7-bit modes occupy one byte per value. | They measure numerical sensitivity, not the claimed storage rate. | Restrict memory claims to truly packed 2/4/8-bit modes or implement exact packers. |
| P2 | The local generation protocol differs from official StructEval generation. | Scores may use official extraction but not an official end-to-end inference protocol. | Label the protocol precisely; record temperature, token cap, stopping rules, model revision, dataset hash, package versions, hardware, and code revision. |

## Literature map

This is a critical map of work that changes our design or novelty boundary. It
is not a claim that every KV-cache paper has been enumerated.

### 1. Quantizer and cache baselines

| Work | Mechanism relevant to us | Consequence for our implementation |
|---|---|---|
| [KIVI, ICML 2024](https://proceedings.mlr.press/v235/liu24bz.html) | Asymmetric grouped quantization; keys per-channel, values per-token; recent FP residual; fused kernels. | This is the minimum reference implementation we must reproduce before saying "KIVI-style" in a result table. |
| [KVQuant](https://arxiv.org/abs/2401.18079) | Pre-RoPE keys, per-channel calibration, nonuniform codebooks, sparse outliers, sink protection. | Shows that axes alone are insufficient and that outliers and RoPE location matter. |
| [GEAR](https://arxiv.org/abs/2403.05527) | Quantized backbone plus sparse outliers and low-rank residual correction. | A relevant rescue baseline if a corrected affine K4 cache still fails. |
| [SKVQ](https://arxiv.org/abs/2405.06219) | Channel reordering, clipped dynamic asymmetric grouping, sinks, and recent FP window. | Provides a stronger low-bit scalar baseline than our current absmax RTN. |
| [ZipCache, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/7e57131fdeb815764434b65162c88895-Abstract-Conference.html) | Attention-derived saliency and mixed token precision with channel-separable value quantization. | Use as an attention/saliency allocation baseline at the same measured byte budget. |
| [Coupled Quantization, NeurIPS 2024](https://papers.neurips.cc/paper_files/paper/2024/hash/05d6b5b6901fb57d2c287e1d3ce6d63c-Abstract-Conference.html) | Joint codebooks for channel tuples. | Represents a vector-quantization branch, not a scalar blockwise extension. |
| [QAQ](https://arxiv.org/abs/2403.04643) | Per-token K/V bits estimated from attention error, with sparse outliers and CPU fallback. | Relevant adaptive baseline, but not reproducible with irreversible GPU-only blocks. |
| [AQUA-KV](https://arxiv.org/abs/2501.19392) | Predict current-layer K/V from reconstructed neighboring-layer states and quantize the residual. | Correction-based branch requiring learned adapters and cross-layer access. |
| [KV-AdaQuant](https://arxiv.org/abs/2502.15075) | Key-favored precision such as K4/V2, supported by sensitivity analysis. | Directly relevant to our observed key-precision cliff; K and V should not be assumed equally sensitive. |
| [KVTuner](https://arxiv.org/abs/2502.04420) | Per-layer mixed K/V precision selected by offline Pareto search. | A high-priority equal-budget baseline before inventing token protection. |
| [TurboQuant, ICLR 2026](https://openreview.net/forum?id=tO3ASKZlok) | Rotation and distribution-aware scalar codebooks targeting distortion and inner products. | A strong quantizer baseline, but it does not directly target structured validity. |
| [The Risk of KV Cache Compression](https://arxiv.org/abs/2607.01520) | Theoretical task-dependent minimax compression risk under causal attention. | Supports avoiding universal "near-lossless" claims and measuring task-conditional failures. |

### 2. Adaptive fidelity, allocation, and novelty threats

| Work | Overlap with our idea | Remaining distinction |
|---|---|---|
| [Don't Waste Bits](https://arxiv.org/abs/2604.04722) | Learned token-wise choice among 2, 4, 8, and FP16 using entropy, rarity, attention variance, and confidence. | Our method cannot claim novelty from unequal token precision alone. Schema-conditioned signals and constrained-generation outcomes must be the distinction. |
| [RateQuant](https://arxiv.org/abs/2605.06675) | Explicit rate-distortion allocation for mixed-precision KV heads. | "Reliability allocation under a budget" alone is not a clean novelty claim. We need a structured semantic utility and prompt-visible constraints. |
| [KVTuner](https://arxiv.org/abs/2502.04420) | Sensitivity-aware layer allocation. | Layerwise allocation is an essential baseline, not our contribution. |
| [TriAxialKV](https://arxiv.org/abs/2605.17170) | Role-, recency-, and modality-aware INT2/INT4 cache precision. | Role-aware fidelity is already studied; formal schema state and semantic constraint retention are narrower. |
| [InfoKV](https://arxiv.org/abs/2606.26875) | Entropy and forward influence score token importance for eviction. | Use its score as a non-structural importance baseline. |
| [RotateKV](https://arxiv.org/abs/2501.16383) and [KVarN](https://arxiv.org/abs/2606.03458) | Rotation and variance normalization improve low-bit numerical robustness. | These should be considered quantizer improvements before attributing failures to allocation policy. |

### 3. Structure-aware KV work

| Work | Overlap with our topic | Implication |
|---|---|---|
| [CodeComp](https://arxiv.org/abs/2604.10235) | Uses code property graphs and static program structure for KV compression. | Direct precedent against a broad "first structure-aware KV method" claim. Our experiments must include non-code schemas and fixed-byte semantic validity. |
| [Graph-KV](https://arxiv.org/abs/2506.07334) | Graph-derived token relationships guide cache selection. | Graph/structure importance propagation is not novel by itself. |
| [StructKV](https://arxiv.org/abs/2604.06746) and [ArborKV](https://arxiv.org/abs/2605.22106) | Structure-aware retention/organization for long-context cache. | Reinforces that the contribution must be a precise constraint-conditioned allocation problem, not generic structure awareness. |
| [The Pitfalls of KV Cache Compression](https://arxiv.org/abs/2510.00231) | Shows uneven degradation of instruction behavior under compression. | Motivates conditional failure analysis, but should be cited as a preprint rather than an accepted ICLR paper. |

### 4. Structured-output evaluation and constrained decoding

| Work | Role in our study | Important boundary |
|---|---|---|
| [StructEval, TMLR 2026](https://arxiv.org/abs/2505.20139) | Main benchmark for conversion into structured formats and required-content evaluation. | We must use family-balanced task manifests and official output-specific scorers. |
| [JSONSchemaBench](https://arxiv.org/abs/2501.10868) | Tests JSON-schema constrained-decoding engines. | Useful for decoder compliance and overhead, not sufficient for semantic correctness. |
| [Berkeley Function-Calling Leaderboard, ICML 2025](https://proceedings.mlr.press/v267/patil25a.html) | Evaluates function/tool selection and arguments. | Strong second benchmark for valid-but-semantically-wrong outputs. |
| [PICARD, EMNLP 2021](https://aclanthology.org/2021.emnlp-main.779/) | Incrementally rejects invalid SQL tokens during decoding. | Demonstrates that grammar validity can be guaranteed independently of KV fidelity. |
| [SynCode](https://arxiv.org/abs/2403.01632), [DOMINO](https://arxiv.org/abs/2403.06988), and [XGrammar](https://arxiv.org/abs/2411.15100) | Grammar-constrained decoding systems. | Under these decoders, punctuation protection may be redundant; evaluate semantic decisions among allowed tokens. |
| [CRANE](https://arxiv.org/abs/2502.09061) | Studies interactions between constrained decoding and reasoning. | Warns that hard constraints can alter generation quality, so constraints need their own control condition. |

### 5. Coding-theoretic framing

The current cache method is lossy source representation under a memory budget.
It is **not** yet an unequal-error-protection code because it adds no redundant
channel code, defines no noisy channel, and provides no decoding error exponent
or recovery guarantee.

Use coding theory as disciplined motivation, not as a theorem claim:

- [Shannon's rate-distortion theory](https://ieeexplore.ieee.org/document/5311476)
  motivates choosing a representation under a rate budget and task-weighted
  distortion.
- [Entropy-constrained vector quantization](https://web.stanford.edu/class/ee398a/handouts/papers/Chou%20-%20EC%20VQ.pdf)
  motivates operational rate allocation including code lengths and metadata.
- [Unequal error protection](https://arxiv.org/abs/0803.2570) is an analogy for
  unequal importance, but its formal guarantees do not transfer to this cache.
- [Huffman coding](https://doi.org/10.1109/JRPROC.1952.273898) motivates unequal
  resource use from symbol statistics, but Don't Waste Bits already uses this
  intuition for token precision.
- [Constrained systems](https://cmrr-star.ucsd.edu/static/book/book_pdf/chapter1.pdf)
  provides language for valid sequence sets, while grammar-constrained decoding
  is the mechanism that actually enforces the set during LLM generation.

A precise empirical objective is:

> Maximize average structured utility over a paired task set, subject to a hard
> upper bound on measured persistent KV bytes.

Utility should include parse success, full required-path success, mean path
coverage, and task-specific semantic correctness. It should not be replaced by
average tensor MSE alone.

## Revised experimental protocol

### Stage 0: establish a credible cache baseline

1. Implement KIVI-compatible affine quantization using every code, explicit
   scale/zero-point metadata, correct K and V grouping, exact recent-window
   semantics, and a controlled prefill policy.
2. Add unit tests for zero-range tensors, all code levels, group boundaries,
   residual boundaries, prefill logits, and quantize/dequantize error.
3. Initially report only truly packed 2/4/8-bit storage.
4. Report persistent bytes by component. Separately measure peak CUDA memory and
   time; do not infer either from packed payload bytes.

### Stage 1: paired JSON-output benchmark

1. Freeze one 100-task manifest: 20 each of Text, CSV, XML, YAML, and TOML to
   JSON, seed 42.
2. Run FP16, affine K8/V8, K8/V4, K4/V4, K4/V2, and selected layerwise policies
   on exactly those task IDs.
3. Report absolute score and compression-induced failure conditional on FP16
   success. Separate baseline model failures from quantization failures.
4. Use paired bootstrap intervals and McNemar tests for binary outcomes.

### Stage 2: separate syntax from semantics

Run the same tasks in a 2-by-2 design:

| Cache | Unconstrained decoding | Grammar-constrained decoding |
|---|---|---|
| FP16 | Baseline syntax and semantics | Constraint-engine effect |
| Compressed | Total compression degradation | Semantic degradation after syntax is guaranteed |

For constrained runs, log decoder intervention rate, masked probability mass,
allowed-set size, unconstrained top-token validity, and selected-token rank before
masking. Never derive a constraint from the reference answer.

### Stage 3: test allocation, not quantizer accidents

At exactly equal measured bytes, compare:

1. Uniform precision.
2. Random protected positions.
3. Recent-window and sink protection.
4. Attention/saliency allocation such as ZipCache or InfoKV signals.
5. Layerwise K/V allocation such as KVTuner.
6. Oracle schema-path protection, labeled only as an upper bound.
7. Deployable prompt/schema-derived protection.

Only after the corrected uniform baseline is stable should GEAR-style correction,
rotation/normalization, or learned token controllers be added.

### Stage 4: broaden beyond JSON

The full StructEval file contains 44 input-output pairs and output-specific
renderable/non-renderable evaluation. A 100-task cross-format sample can be
stratified approximately across pairs, but cannot be scored by the current
JSON-only path. Integrate the official evaluators first, then freeze and publish
the task manifest.

## Recommended thesis positioning

Working title:

**Structure-Conditioned KV-Cache Fidelity Allocation Under a Hard Memory Budget**

More application-specific alternative:

**Schema-Conditioned KV-Cache Fidelity Allocation for Semantically Reliable Constrained Generation**

Avoid these claims until stronger evidence exists:

- exact KIVI reproduction;
- first structure-aware KV compression method;
- formal unequal-error-protection guarantee;
- real GPU memory or throughput gain from persistent packed bytes;
- general near-lossless low-bit structured generation.

The next defensible contribution is smaller but stronger: a corrected cache
baseline, a paired and balanced structured-output protocol, and evidence about
whether prompt-visible schema information improves semantic structured utility at
the same measured storage rate.
