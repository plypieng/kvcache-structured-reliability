# Paired StructEval-T analysis

The same 950 tasks are compared under FP16, KIVI-4, and KIVI-2.
Confidence intervals are paired bootstrap intervals with resampling
inside the generation and conversion categories for the StructEval-T score.

| Condition | T-score delta | 95% CI | FP16-only parse | Candidate-only parse | McNemar p | Path regressions |
|---|---:|---:|---:|---:|---:|---:|
| KIVI-4 | +0.0162 | [0.0026, 0.0297] | 23 | 60 | 5.974e-05 | 41 |
| KIVI-2 | -0.0777 | [-0.1069, -0.0491] | 125 | 118 | 0.7004 | 195 |

## Interpretation boundary

A candidate-only parse success and an FP16-only parse success are both
behavioral transitions under deterministic decoding. Aggregate gains do not
prove that quantization improves the model, and surface parser errors do not
identify the causal cache position. The exported annotation table must be
reviewed before making a delimiter- or tag-specific claim.

## Automated surface diagnostics

### KIVI-4

- `indentation_or_mapping_syntax`: 12
- `invalid_leading_token`: 10
- `extra_content_or_multiple_roots`: 1

### KIVI-2

- `missing_begin_marker`: 42
- `invalid_leading_token`: 30
- `indentation_or_mapping_syntax`: 13
- `missing_end_marker`: 10
- `other_parser_error`: 7
- `unbalanced_or_mismatched_delimiter`: 7
- `literal_escaped_layout_outside_string`: 6
- `invalid_key_syntax`: 4
- `extra_content_or_multiple_roots`: 3
- `invalid_token`: 3
