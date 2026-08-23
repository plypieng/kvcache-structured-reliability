# Matched-Budget Protection Pilot

## Purpose

The first protection experiment tests whether a small amount of extra cache
fidelity is more useful when it is assigned to structure-related positions.
It is a development pilot, not yet a claim of a new compression method.

## Frozen tasks

- 23 StructEval-T tasks where KIVI-4 previously changed an FP16 parseable
  output into a parse failure.
- 23 matched non-regression controls selected with the same input/output
  format and similar FP16 output length.
- The same 46-task manifest is used for every condition.

The manifest is
`toy_kv_experiments/data/structeval_full/manifests/protection_pilot_failure_control_46.json`.

## Conditions

All conditions use Mistral-7B-Instruct-v0.2, greedy decoding, the official
StructEval prompt/extraction path, real blockwise packed cache storage, K4/V4
base quantization, residual length 128, group size 32, and protected storage
at 8 bits. The protection budget is capped at 10.5 average bits per cache
scalar. This is a fixed upper bound; the uniform KIVI-4 baseline normally uses
less storage than the ceiling.

1. **Uniform KIVI-4:** no selected positions receive extra fidelity.
2. **Random protection:** all older positions are candidates; the same seeded
   random order is used for every task.
3. **Recency protection:** all older positions are candidates; the most recent
   eligible positions receive extra fidelity first.
4. **Structure-aware protection:** prompt-visible JSON syntax markers and
   schema/path identifiers receive extra fidelity by score.

The comparison is fair as an equal storage ceiling, not as an exact equality
of used bytes on every task. Result rows record the actual packed-cache bytes,
the ceiling bytes, protected positions, and StructEval scores.

## Interpretation rule

The failure set is intentionally enriched for KIVI-4 regressions, so it is
useful for debugging but not an unbiased benchmark estimate. The control set
tests whether any apparent gain is specific to failure-prone tasks. A useful
next signal is a structure-aware improvement over both random and recency
controls at the same storage ceiling, followed by a frozen unseen evaluation.
