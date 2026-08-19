# Research Progress Audit: KV-Cache Reliability for Structured Output

Date: 2026-07-17

## Remediation update

The audit findings were implemented before starting another large queue:

- Cache revision `kivi-post-attention-finalize-v3` now returns the current
  full-precision residual to attention and seals eligible K/V only after the
  model forward completes.
- An explicit KIVI-style state trace and a tiny Qwen model logit-boundary test
  protect the update-order invariant.
- The allocation budget now charges key fidelity per full token group and value
  fidelity per token using packed payload, scale/minimum metadata, and FP16
  residual storage.
- Every task is written through an atomic checkpoint with fingerprint-checked
  resume support and a machine-readable status file.
- Results now record source hashes, model/tokenizer identity, package versions,
  hardware, decoding settings, task duration, actual persistent bytes,
  effective stored bits, and compression relative to FP16.
- Paired analysis now includes FP16-reference JSON leaf path-value agreement.
  This diagnostic is explicitly not presented as ground truth or an official
  StructEval metric.
- Oracle evaluator paths are blocked by default and require an explicit CLI
  override. Normal runs are labeled `prompt-schema-lexical-protection` or
  `uniform-cache-fidelity`.
- The complete test suite passes locally and on the A6000: `47 passed`.
- A paired FP16/K2 JSON-10 validity gate was started on the A6000. JSON-100 is
  prepared but remains blocked until every gate disagreement is inspected.

## Executive verdict

The project has progressed from an educational fake-quantization notebook into a
useful experimental prototype with frozen task manifests, packed affine cache
storage, official-style JSON extraction and scoring, and paired statistical
analysis. That is real progress.

The project is not ready for the planned full rerun. The current
`kivi-seal-before-attention-v2` cache does not follow the update order in the
official KIVI implementation. It seals a completed residual block before the
current attention operation, while official KIVI uses the full-precision block
for the current attention operation and seals it afterward. Consequently, no
compressed result currently in the repository should be presented as a valid
KIVI reproduction.

The strongest defensible B4 thesis direction remains viable, but it should be
narrowed to an empirical and budget-controlled question:

> Under identical decoding and measured persistent-cache budgets, can
> prompt-visible schema information identify K/V cache groups whose higher
> fidelity improves structured-output reliability relative to uniform, random,
> and recency-based allocation?

This is more defensible than claiming a formal reliability guarantee or a new
general-purpose KV quantizer.

## Overall research status

| Component | Status | Audit judgment |
|---|---|---|
| Transformer and KV-cache understanding | Strong learning progress | Sufficient to conduct the B4 study with continued paper-level verification. |
| Toy model and fake quantization | Complete as education | Useful history, not evidence for the final method. |
| Real packed cache | Substantial prototype | Affine 2/4/8-bit storage and byte accounting are useful, but update order must be corrected. |
| StructEval data pipeline | Mostly sound for JSON output | Frozen nested manifests and balanced input families are good. It is only a five-pair JSON-output subset. |
| StructEval scoring | Sound for JSON non-renderable scoring | Mirrors parse validity, required-path coverage, and the official 0.2/0.8 score. |
| Official benchmark comparability | Not achieved | Our inference uses greedy decoding, a token cap, and custom stopping, unlike the official inference runner. |
| Statistical analysis | Good foundation | Paired bootstrap and McNemar code exists, but corrected runs are too small. |
| Fidelity-allocation method | Early prototype | Current signal is lexical and prompt-visible, not parser-state or formal-constraint aware. |
| Novelty | Plausible only in a narrow form | Broad token-wise or structure-aware allocation is already covered by related work. |
| Publishable evidence | Not yet available | There is no corrected, paired compressed baseline matrix. |

## P0 findings: resolve before any large rerun

### P0.1 The current residual update order is not KIVI

Current behavior in
[`real_quantized_cache.py`](../toy_kv_experiments/real_quantized_cache.py):

1. Append the current token's K/V to the residual.
2. Seal completed key groups and value overflow.
3. Dequantize the newly sealed blocks.
4. Return the reconstructed cache to attention.

This is implemented in `_decode_update`, where `_seal_key_residual()` and
`_seal_value_residual()` run before the K/V tensors are returned to the model.

The official KIVI implementation performs the operations in this order:

1. Append the current full-precision key.
2. Compute the current `QK^T` contribution using that full-precision key
   residual.
3. Seal the completed key residual after its current use.
4. Compute the current attention output using the full-precision value residual.
5. Seal the oldest value only after its current use.

Evidence:

- [Official KIVI implementation](https://github.com/jy-yuan/KIVI/blob/main/models/llama_kivi.py)
- [KIVI paper, ICML 2024](https://proceedings.mlr.press/v235/liu24bz.html)

The existing test
`test_real_quantized_cache_seals_residuals_before_decode_attention` confirms our
current behavior. It is not a golden test against KIVI and should be replaced.

Required correction:

- Let cache `update()` expose the current full residual to attention.
- Add a post-attention finalization operation that seals every layer after the
  model forward call.
- In the manual generation loop, call finalization immediately after each model
  forward pass.
- Compare a short synthetic trace against an independent reference
  implementation of KIVI Algorithm 1.
- Assign a new algorithm revision. Do not reuse the `v2` label.

### P0.2 There is no corrected end-to-end baseline matrix

The result generations currently fall into three incompatible groups:

| Generation | Interpretation |
|---|---|
| Fake/repeated and fake/blockwise files | Numerical sensitivity experiments only. |
| `legacy-materialize-before-seal-v1` files | Historical real-storage experiments using a superseded cache path. |
| `kivi-seal-before-attention-v2` file | Packed real-cache experiment, but the update order is not official KIVI. |

The only `v2` end-to-end run contains 10 tasks at K2/V2. It is useful as a bug
and sensitivity probe, not as a conclusion.

Paired against FP16 on the same 10 tasks:

| Metric | FP16 | K2/V2 v2 | Paired change |
|---|---:|---:|---:|
| Parse success | 10/10 | 8/10 | -2 tasks |
| Full required paths | 4/10 | 2/10 | -2 tasks |
| Mean required-path rate | 0.517 | 0.419 | -0.098 |
| Mean StructEval score | 0.613 | 0.495 | -0.118 |

The 95% paired bootstrap interval for the score change is `[-0.434, 0.148]`,
and the exact McNemar p-value for parse success is `0.5`. With only 10 tasks,
the observed degradation is not statistically conclusive. Some tasks improved
under K2/V2 while two YAML-input tasks acquired malformed output-control
markers and became unparsable.

Required correction:

- First run a 10-task implementation gate after fixing the update order.
- Only then run the frozen 50- or 100-task matrix.
- Never pool rows from different `cache_algorithm_revision` values.

### P0.3 StructEval does not measure value correctness in this setup

For non-renderable JSON tasks, StructEval computes:

`final score = 0.2 * parse/render validity + 0.8 * required-path coverage`

Required-path coverage checks whether keys or paths exist. It does not verify
that the generated values faithfully match the source. A structurally complete
but factually incorrect conversion can therefore receive a high score.

Consequences:

- Current results support statements about parseability and structural
  coverage, not semantic correctness.
- They do not support claims about correct tool arguments, enum choices, or
  faithful field values.
- If a JSON Schema constrained decoder guarantees syntax and required fields,
  the current metric can saturate even while values are wrong.

Required correction:

- Keep StructEval JSON as the structural validity gate.
- Add content/value fidelity for conversion tasks by normalizing the structured
  source and comparing generated values where deterministic conversion is
  possible.
- Later add a tool-calling benchmark such as BFCL if the thesis claims reliable
  tool arguments.

### P0.4 The allocation budget is not group-aware or byte-exact

The current protection budget counts extra precision per selected token. Real
key quantization groups 32 token positions together. In
`_append_key_blocks()`, one protected position raises the precision of the
entire key group because the group bit width is selected with `max(...)`.

Example:

- Budget model selects one key token for 8-bit protection.
- Real cache promotes all 32 tokens in its key group to 8-bit.
- The nominal budget counts one promoted token, while storage pays for 32.

This is a direct confound for an equal-budget comparison. Frequent JSON syntax
tokens can also cause most key groups to be promoted.

Required correction:

- Allocate key fidelity at key-group granularity.
- Allocate value fidelity at token granularity, because values are grouped over
  channels for each token.
- Optimize against measured payload plus scale/minimum metadata and residual
  bytes, not nominal average bits.
- Log desired positions, actual sealed group widths, and final measured bytes.

## P1 findings: required for a defensible thesis result

### P1.1 The inference protocol is not the official StructEval protocol

Our runner uses:

- greedy decoding with `temperature=0.0`;
- a maximum of 1024 new tokens;
- stopping on `<|END_CODE|>`;
- a repeated-ngram loop detector.

The checked official StructEval inference code calls its model engine with
`temperature=1.0` and `max_tokens=None`. Our prompt wrapper and JSON extraction
match the official repository, but the complete inference procedure does not.

This is not necessarily bad for the causal quantization study. Greedy decoding
removes sampling noise and makes paired treatment comparisons clearer. It must
be labeled correctly:

- `deterministic StructEval-T JSON subset with official-style scoring`, not
  `official StructEval benchmark score`.
- A separate official-protocol replication can be run later if leaderboard
  comparability is required.

### P1.2 The proposed method is lexical, not formally parser-aware

Current prompt-visible protection extracts identifiers with regular
expressions, tokenizes them, and protects matching token pieces. JSON syntax
protection uses token-string tests for braces, brackets, quotes, colons, commas,
and control markers.

It does not currently use:

- an incremental parser state;
- a formal grammar automaton;
- the current allowed-token set;
- parser transition risk;
- JSON Schema required-field state.

Therefore, the implemented method should be called `prompt-schema lexical
protection`, not `formal-constraint-aware protection`.

There is also a causal timing limitation: a generated structural token can only
be protected after the model has already emitted it. Protecting its K/V may help
later decisions, but it cannot prevent the original emission error. Prompt
schema tokens are available early and are the cleaner first target.

### P1.3 Reproducibility metadata is incomplete

The workspace root is not currently a Git repository, and result rows do not
record enough information to reconstruct a run. Missing or incomplete metadata
includes:

- project code commit or source-tree hash;
- exact model revision or weight hash;
- tokenizer revision;
- PyTorch, Transformers, CUDA, and driver versions;
- GPU model in the result file;
- temperature and sampling parameters;
- start/end timestamps and per-task duration;
- deterministic-kernel settings.

The nested manifests and source-dataset hash are good. The same rigor should be
applied to model and code provenance.

### P1.4 Interrupted runs lose all completed rows

Results are written only after `run_structeval_smoke()` returns. The interrupted
K4/V8 run reached 25/50 but produced no usable partial result.

Required correction:

- Write one atomic checkpoint after every task.
- Support resume by task ID.
- Store a run manifest before inference starts.
- Distinguish `running`, `interrupted`, and `complete` status.

### P1.5 Persistent bytes are real, but system claims remain out of scope

The cache stores packed 2/4/8-bit payloads and includes scale, minimum, and
residual bytes. This is useful and should be retained.

Attention still reconstructs the complete floating-point K/V history on every
step. Therefore:

- measured persistent-cache bytes are meaningful;
- peak GPU memory is not represented by those bytes;
- latency and throughput are not comparable to fused KIVI kernels;
- current runtimes mainly measure Python/dequantization overhead.

The K2/V2 smoke run illustrates the distinction. It reports about 3.30x
persistent-cache compression, but approximately 47% of stored bytes are still
the FP16 residual and about 18% are quantization metadata. This is an effective
rate near 4.85 bits per FP16 scalar, not a literal 2-bit end-to-end cache.

### P1.6 Scope is narrower than the project language sometimes suggests

The frozen manifests cover five StructEval-T task pairs:

- Text to JSON
- CSV to JSON
- XML to JSON
- YAML to JSON
- TOML to JSON

They do not cover all 19 StructEval-T pairs, all 44 StructEval task types, code
generation, or tool calling. The final thesis should state this scope exactly.

## What is already credible and reusable

The following work should be preserved:

1. The 10-, 50-, and 100-task JSON manifests are balanced and properly nested.
2. Every treatment can use exactly the same task IDs and dataset hash.
3. JSON extraction mirrors StructEval's marker, fence, and raw-text fallback.
4. JSON wildcard path behavior matches the checked official implementation.
5. The packed affine quantizer uses all `2^B` codes and includes scale/minimum
   metadata in byte accounting.
6. Key groups run across tokens per channel; value groups run across channels
   per token, matching KIVI's central quantization-axis idea.
7. Prompt prefill attention is full precision before compressed storage is
   retained.
8. Paired bootstrap intervals and exact McNemar tests are implemented.
9. `python3 -m pytest -q` passes all 40 tests on 2026-07-17.
10. Two independent FP16 runs produced identical text on all 40 overlapping
    tasks, which is useful evidence that greedy baseline inference is stable.

## Current evidence ledger

| Result | Status | What it can support |
|---|---|---|
| Toy GPT and TinyStories experiments | Educational | Understanding autoregressive inference, cache tensors, and fake quantization. |
| Early 0.5B and 7B fake-quantization StructEval runs | Historical | Motivation and debugging history only. |
| FP16 JSON-100: 97/100 parse, 57/100 full paths, score 0.7138 | Reusable baseline candidate | Baseline model capability, if code/model provenance is reconstructed. |
| Legacy K4/V8, K8/V4, K4/V4 JSON-50 | Exploratory only | Sensitivity hypotheses; not final KIVI comparisons. |
| v2 K2/V2 JSON-10: 8/10 parse, score 0.495 | Exploratory only | Shows that aggressive cache perturbation can alter control markers and task outcomes. |
| Prompt-schema protection analyses | Method-development evidence | Signal coverage and implementation feasibility, not proof of improved reliability. |

## Claim audit

Statements that are currently supportable:

- We built a persistent packed KV-cache fidelity reference for Qwen2.5-7B.
- We reproduced KIVI's affine quantization axes and recent-residual concept, but
  not yet its exact update order or fused execution.
- Structured-output scores can change under cache quantization, including both
  failures and occasional improvements.
- StructEval tasks and quantization settings must be paired because baseline
  task difficulty is highly uneven.
- Metadata and FP16 residual storage materially change the effective rate.

Statements that are not currently supportable:

- We reproduced KIVI.
- K2/V2 significantly reduces structured-output reliability.
- K4/V8 or K8/V4 is superior under the corrected cache.
- Schema/path protection improves reliability.
- The current score is an official full StructEval benchmark score.
- The method preserves semantic correctness or tool-call correctness.
- The method provides a formal or coding-theoretic reliability guarantee.
- Persistent byte reduction gives production memory or throughput gains.

## Novelty assessment

Broad novelty is not defensible. Existing work already covers:

- K/V-specific grouped quantization and residual caches: KIVI.
- Layer-wise and K/V mixed precision: KVTuner and related adaptive methods.
- Token-wise learned precision allocation: Don't Waste Bits.
- Rate-distortion allocation: RateQuant.
- Structure-aware cache retention for code and long contexts: CodeComp,
  StructKV, and ArborKV.
- Exact-output recovery around lossy caches: VeriCache.
- Grammar-constrained generation: PICARD, SynCode, DOMINO, and XGrammar.

The remaining plausible niche is narrower:

> A controlled characterization and simple group-aware allocation method using
> deployable prompt-schema signals, evaluated specifically on structured-output
> failure under equal measured persistent-cache bytes.

This remains a hypothesis, not a verified novelty claim. A final literature
search and comparison table are required before the thesis claim is locked.

For a B4 thesis, a negative result can still be valuable. If prompt-schema
protection does not beat random or recency allocation, the thesis can report
where structured-output failures originate and why intuitive token protection
does not transfer into cache reliability.

## Pre-rerun validity gate

Do not start the 50- or 100-task queue until all items below pass.

1. Replace seal-before-attention with a post-attention finalization path.
2. Add a reference trace test covering prefill, a key boundary, a value
   overflow, and multiple decoding steps.
3. Add a model-level logit parity test for FP16 prefill and a reference
   quantized boundary step.
4. Add atomic per-task checkpointing and resume.
5. Record complete run, model, package, hardware, and decoding metadata.
6. Separate the deterministic causal-study protocol from official benchmark
   replication in names and documentation.
7. Make the protection allocator key-group aware and enforce measured-byte
   budgets.
8. Run FP16 and K2/V2 on the same 10-task manifest as an implementation gate.
9. Manually inspect every disagreement on the 10-task gate.
10. Confirm that no row uses evaluator-only paths unless explicitly labeled
    `oracle`.

## Recommended experiment after the gate

### Stage A: corrected uniform-cache characterization

Use the frozen JSON-100 manifest with greedy decoding and identical stopping:

1. FP16
2. K8/V8
3. K8/V4
4. K4/V4
5. K2/V2

Primary analysis:

- paired change in required-path rate;
- paired change in final structural score;
- compression-induced parse failures conditional on FP16 parse success;
- full-path gains and losses;
- bootstrap confidence intervals;
- results by input family;
- actual bytes per cached token and storage-component breakdown.

The purpose is to find one nontrivial operating point: enough degradation to
recover, but not complete collapse.

### Stage B: allocation study at one operating point

At one fixed measured-byte budget, compare:

1. Uniform precision
2. Random protected key groups/value tokens
3. Recency plus sink protection
4. Prompt-schema lexical protection
5. Optional attention-saliency protection
6. Oracle required-path protection, clearly labeled as an upper bound

Do not implement every recent paper. For a B4 thesis, three strong baselines and
one proposed heuristic are preferable to many incomplete reproductions.

### Stage C: syntax versus semantic behavior

Only after Stage B works, add a constrained-decoding condition:

| Cache | Unconstrained decoding | Grammar/schema constrained decoding |
|---|---|---|
| FP16 | Baseline structure and content | Constraint-engine effect |
| Compressed | Total compression effect | Residual content effect after validity is enforced |

This stage requires a value/content metric; StructEval path existence alone is
not sufficient.

## Immediate decision

The next action is not a large rerun. The next action is to repair and validate
the cache update order, byte-aware allocation, metadata capture, and resume
support. After a corrected 10-task gate passes, the existing A6000 queue can be
reused for the full paired experiment.
