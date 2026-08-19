# Poster Plan for August 3, 2026

> Historical plan prepared July 17. Its broad-matrix schedule is superseded by
> `docs/MASTER_PLAN_20260723_TO_20260803.md` and the July 23 evidence freeze.
> Keep this file for the poster rationale and storyboard, not as the active
> daily schedule.

Prepared: July 17, 2026  
Poster session: Monday, August 3, 2026  
Result freeze: Sunday, July 26, 2026

Current status: the full official FP16 reproduction was started and remains
resumable, but poster execution has been reprioritized to a frozen 100-task
all-format pilot so that a complete paired result can be frozen by July 26.

The full benchmark and the August 3 poster now have different completion
horizons. The poster may use a clearly labeled balanced subset, while every
claim of comparison with the StructEval website must come from the complete
official inference, rendering, and evaluation pipeline.

## Decision

The poster should present one completed and reproducible study rather than the
entire intended thesis. Its central question is:

> How does KV-cache quantization affect structured generation across StructEval
> formats, and where is unequal fidelity allocation worth testing next?

The poster is still valid if the allocation method does not improve the result.
In that case, the contribution is a corrected characterization of where
structured-output failures occur and evidence that intuitive protection is not
sufficient.

Working title:

**Reliability of Structured Generation under KV-Cache Quantization**

Use `KIVI-style cache quantization`, not `KIVI reproduction`. The reference
implementation follows KIVI's quantization axes and residual-cache idea, but it
uses ordinary attention with cache reconstruction rather than fused packed
kernels.

## Minimum Poster Evidence

### Experiment A: corrected uniform-cache characterization for the poster

Use the frozen 100-task all-format manifest, Qwen2.5-7B-Instruct, greedy
decoding, identical prompts and stopping, and the corrected
`kivi-post-attention-finalize-v3` cache.

The manifest contains 25 tasks from each official category, all 44 input/output
task types, and all 18 normalized output formats. This is a paired poster pilot,
not an official StructEval leaderboard score.

| Condition | Role |
|---|---|
| FP16 | Reference output and model capability |
| K8/V8 | High-fidelity quantized control |
| K8/V4 | KIVI-motivated asymmetric setting |
| K4/V4 | Moderate uniform compression |
| K4/V8 | Isolate lower key fidelity from value fidelity |

Run all five conditions on the same 100 tasks. K2/V2 is omitted from the poster
matrix because it is primarily a failure probe and disproportionately expensive
in the unfused reference implementation.

Primary measurements:

- JSON parse success;
- required-path coverage;
- StructEval structural score;
- compression-induced failures conditional on FP16 success;
- FP16-reference leaf path/value agreement, labeled as a diagnostic rather
  than an official benchmark metric;
- measured persistent-cache bytes and effective stored bits;
- runtime reported only as reference-implementation overhead, not speedup.

### Experiment B: one small allocation comparison

Choose one operating point from Experiment A where compression causes some but
not total degradation. On the same 50-task subset and at equal measured cache
bytes, compare:

1. Uniform fidelity.
2. Random protected key groups and value tokens.
3. Recency plus attention-sink protection.
4. Prompt-schema lexical protection.

The prompt-schema method is the proposed B4 heuristic. It should not be called
parser-aware or formally constrained: it uses only schema information visible
in the prompt. An oracle required-path condition may be shown only as a clearly
labeled upper bound and must not be mixed with deployable results.

Experiment B is a preliminary poster result, not a requirement for the poster
to remain presentable.

## Validity Gate Before the Main Runs

The active paired JSON-10 gate must pass the following checks:

- all rows use cache revision `kivi-post-attention-finalize-v3`;
- task IDs, prompts, generation settings, model, and seed are paired;
- no evaluator-only required paths enter the protection signal;
- every FP16/K2 output disagreement is inspected manually;
- observed cache bytes and metadata are plausible;
- extraction and scoring agree with the checked StructEval JSON logic;
- no result is labeled as an official full StructEval benchmark score.

Poor K2/V2 quality does not fail the gate. Only an implementation, provenance,
pairing, extraction, or scoring error fails it.

## Schedule

### July 17-18: close the validity gate

- Finish FP16 versus K2/V2 on JSON-10.
- Run the paired analysis and inspect all ten pairs.
- Fix only correctness or provenance defects.
- Decide `gate passed` or `gate failed` in the experiment note.
- Obtain the official poster dimensions, orientation, language, logo rules,
  printing method, and print-submission deadline.

### July 18-21: run Experiment A

- Queue FP16, K8/V8, K8/V4, K4/V4, and K2/V2 on JSON-100.
- Check the first completed task of every condition before leaving it queued.
- Preserve per-task checkpoints and status files.
- Do not combine any historical fake-quantization, v1, or v2 rows with v3.

### July 21-22: analyze and select the operating point

- Produce the paired metric table with confidence intervals.
- Plot structural score against measured persistent-cache bytes.
- Plot compression-induced failures by source family.
- Select one operating point for allocation; otherwise record why no useful
  operating point exists.

### July 22-25: run Experiment B

- Implement and test random and recency/sink controls if still missing.
- Run all allocation policies at an equal measured-byte budget.
- Inspect representative success, failure, recovery, and regression examples.
- If equal-budget enforcement is not trustworthy by July 24, stop Experiment B
  and present it only as future work.

### July 26: freeze results

- No new quantizer architecture or benchmark condition after this date.
- Freeze result files, source hashes, tables, figures, and exact claims.
- Write a one-paragraph conclusion that remains true under every plotted result.

### July 27-28: build the poster

- Create the poster at the official dimensions.
- Use a three-column reading order: problem, method, evidence.
- Keep one main message per section and no more than four principal figures.
- Include a QR code to a read-only project summary only if permitted.

### July 29: advisor and lab review

- Ask reviewers to identify one unclear claim, one unfair comparison, and one
  unreadable figure.
- Record requested corrections; do not expand the study scope.

### July 30-August 1: revise and proof

- Check terminology, citation metadata, axes, sample counts, and captions.
- Verify that every number can be traced to a frozen result file.
- Print one A4-scale proof and inspect it at arm's length.

### August 2: final production

- Export the final PDF with embedded fonts.
- Check page size, image resolution, margins, and QR code if used.
- Print or submit before the administrative deadline.

### August 3: poster session

- Prepare a 30-second summary, a 2-minute walkthrough, and a 5-minute technical
  explanation.
- Keep the JSON failure example as the entrance for a non-LLM audience.

## Poster Storyboard

### 1. Motivation and question

- Long-context inference stores past keys and values in the KV cache.
- Quantization reduces persistent cache storage but perturbs future attention.
- A small output error can invalidate an otherwise useful structured response.
- Question: which fidelity settings preserve structural reliability under a
  measured memory budget?

### 2. Method

- Show the cache lifecycle: full-precision current residual, attention use,
  post-attention group sealing, packed affine storage, and reconstruction for
  the next step.
- Explain K per-channel grouping across tokens and V per-token grouping across
  channels with one compact diagram.
- State the paired deterministic protocol and the balanced JSON task families.

### 3. Evidence

Use at most four principal visuals:

1. Cache lifecycle and K/V grouping diagram.
2. Structural score versus measured cache bytes.
3. Paired failure/recovery counts, separated by input family.
4. One short FP16-versus-compressed JSON example with the exact structural
   break highlighted.

If Experiment B is valid, add its equal-budget result to visual 2 or 3 rather
than adding a fifth dense figure.

### 4. Conclusion and next step

- State the measured reliability-storage tradeoff.
- State whether prompt-schema protection beat the controls.
- Separate structural validity from semantic correctness.
- List parser-state signals and constrained decoding as future work, not as
  completed contributions.

## Claim Boundaries

Safe claims:

- a corrected, persistent, packed KIVI-style cache was evaluated under paired
  decoding conditions;
- quantization effects were measured on structured JSON generation;
- persistent payload, metadata, and residual storage were included;
- failures and recoveries were analyzed conditionally against FP16.

Claims to avoid:

- exact KIVI reproduction;
- production memory or latency speedup;
- official full StructEval leaderboard comparability;
- preservation of semantic correctness from path coverage alone;
- formal reliability guarantees;
- novelty or superiority of the allocation method without equal-budget data.

## Immediate Actions

1. Run and officially score the complete FP16 StructEval reproduction.
2. Benchmark the corrected compressed throughput before committing the A6000
   to the full matrix; the current unfused estimate is about 15 days per
   compressed condition.
3. Acquire the poster format and printing requirements today.
4. Preserve the completed JSON-10 gate as the first failure-mechanism example.
5. Reserve July 26 as an immovable poster-result freeze.
6. Build the poster from the corrected subset first; treat Experiment B as an optional
   improvement rather than a dependency.
